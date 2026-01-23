/**
 * UCX Maximum Bandwidth Benchmark
 *
 * This harness is tuned for absolute maximum throughput on Mellanox 100GbE
 * hardware. It establishes the UCX performance baseline that WarpForge
 * warpforge-io should aspire to match.
 *
 * Key optimizations:
 * - Zero-copy operations where possible
 * - Pre-registered memory regions
 * - Pipelined/overlapped operations
 * - CPU affinity for cache locality
 * - Optimal transport selection
 *
 * Build:
 *   gcc -O3 -march=native -o ucx_max_bandwidth ucx_max_bandwidth.c \
 *       -lucp -lucs -luct -lpthread
 *
 * Usage:
 *   Server: ./ucx_max_bandwidth -s
 *   Client: ./ucx_max_bandwidth -c <server-ip>
 *
 * Copyright 2026 WarpForge Project
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <errno.h>
#include <sched.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <ucp/api/ucp.h>
#include <ucs/type/status.h>

/* ========================================================================== */
/* Configuration                                                               */
/* ========================================================================== */

#define DEFAULT_PORT        18515
#define DEFAULT_MSG_SIZE    (1024 * 1024)   /* 1MB */
#define DEFAULT_ITERATIONS  10000
#define DEFAULT_WARMUP      1000
#define DEFAULT_WINDOW      64              /* Outstanding operations */

/* Memory alignment for RDMA */
#define ALIGN_SIZE          4096

/* ========================================================================== */
/* Data Structures                                                             */
/* ========================================================================== */

typedef struct {
    int is_server;
    char *server_addr;
    int port;
    size_t msg_size;
    int iterations;
    int warmup;
    int window;
    int cpu_affinity;
    int verbose;
} config_t;

typedef struct {
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;

    /* Pre-registered memory */
    void *send_buf;
    void *recv_buf;
    ucp_mem_h send_mem;
    ucp_mem_h recv_mem;
    void *send_rkey_buf;
    void *recv_rkey_buf;
    size_t send_rkey_len;
    size_t recv_rkey_len;

    /* Remote memory info */
    uint64_t remote_addr;
    ucp_rkey_h remote_rkey;
} ucp_resources_t;

typedef struct {
    uint64_t addr;
    size_t rkey_len;
    /* rkey buffer follows */
} mem_info_t;

/* ========================================================================== */
/* Utility Functions                                                           */
/* ========================================================================== */

static void die(const char *msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

static void die_status(const char *msg, ucs_status_t status) {
    fprintf(stderr, "ERROR: %s: %s\n", msg, ucs_status_string(status));
    exit(1);
}

static void *aligned_alloc_safe(size_t alignment, size_t size) {
    void *ptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        die("Memory allocation failed");
    }
    memset(ptr, 0, size);
    return ptr;
}

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void set_cpu_affinity(int cpu) {
    if (cpu < 0) return;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);

    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        fprintf(stderr, "Warning: Could not set CPU affinity to core %d\n", cpu);
    } else {
        printf("Pinned to CPU core %d\n", cpu);
    }
}

/* ========================================================================== */
/* TCP Helper for Connection Bootstrap                                         */
/* ========================================================================== */

static int tcp_connect(const char *host, int port) {
    struct addrinfo hints = {0}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    if (getaddrinfo(host, port_str, &hints, &res) != 0) {
        die("getaddrinfo failed");
    }

    int sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock < 0) {
        die("socket failed");
    }

    if (connect(sock, res->ai_addr, res->ai_addrlen) < 0) {
        die("connect failed");
    }

    freeaddrinfo(res);
    return sock;
}

static int tcp_accept(int port) {
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) {
        die("socket failed");
    }

    int opt = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        die("bind failed");
    }

    if (listen(server_sock, 1) < 0) {
        die("listen failed");
    }

    printf("Waiting for connection on port %d...\n", port);

    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &client_len);
    if (client_sock < 0) {
        die("accept failed");
    }

    printf("Client connected from %s\n", inet_ntoa(client_addr.sin_addr));

    close(server_sock);
    return client_sock;
}

static void tcp_send_all(int sock, const void *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(sock, (char*)buf + sent, len - sent, 0);
        if (n <= 0) die("tcp send failed");
        sent += n;
    }
}

static void tcp_recv_all(int sock, void *buf, size_t len) {
    size_t recvd = 0;
    while (recvd < len) {
        ssize_t n = recv(sock, (char*)buf + recvd, len - recvd, 0);
        if (n <= 0) die("tcp recv failed");
        recvd += n;
    }
}

/* ========================================================================== */
/* UCP Initialization                                                          */
/* ========================================================================== */

static void init_ucp_context(ucp_resources_t *res, size_t msg_size) {
    ucp_params_t params = {0};
    ucp_config_t *config;
    ucs_status_t status;

    /* Read UCX configuration from environment */
    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        die_status("ucp_config_read", status);
    }

    /* Request features for maximum performance */
    params.field_mask = UCP_PARAM_FIELD_FEATURES |
                        UCP_PARAM_FIELD_REQUEST_SIZE |
                        UCP_PARAM_FIELD_REQUEST_INIT;

    params.features = UCP_FEATURE_RMA |      /* RDMA read/write */
                      UCP_FEATURE_AM;         /* Active messages */

    params.request_size = 0;
    params.request_init = NULL;

    status = ucp_init(&params, config, &res->context);
    ucp_config_release(config);

    if (status != UCS_OK) {
        die_status("ucp_init", status);
    }

    /* Create worker */
    ucp_worker_params_t worker_params = {0};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(res->context, &worker_params, &res->worker);
    if (status != UCS_OK) {
        die_status("ucp_worker_create", status);
    }

    /* Allocate and register send buffer */
    res->send_buf = aligned_alloc_safe(ALIGN_SIZE, msg_size);

    ucp_mem_map_params_t mem_params = {0};
    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    mem_params.address = res->send_buf;
    mem_params.length = msg_size;
    mem_params.flags = UCP_MEM_MAP_NONBLOCK;

    status = ucp_mem_map(res->context, &mem_params, &res->send_mem);
    if (status != UCS_OK) {
        die_status("ucp_mem_map (send)", status);
    }

    /* Pack remote key for send buffer */
    status = ucp_rkey_pack(res->context, res->send_mem,
                           &res->send_rkey_buf, &res->send_rkey_len);
    if (status != UCS_OK) {
        die_status("ucp_rkey_pack (send)", status);
    }

    /* Allocate and register receive buffer */
    res->recv_buf = aligned_alloc_safe(ALIGN_SIZE, msg_size);

    mem_params.address = res->recv_buf;
    status = ucp_mem_map(res->context, &mem_params, &res->recv_mem);
    if (status != UCS_OK) {
        die_status("ucp_mem_map (recv)", status);
    }

    status = ucp_rkey_pack(res->context, res->recv_mem,
                           &res->recv_rkey_buf, &res->recv_rkey_len);
    if (status != UCS_OK) {
        die_status("ucp_rkey_pack (recv)", status);
    }
}

static void exchange_addresses(int sock, ucp_resources_t *res, int is_server) {
    ucs_status_t status;

    /* Get worker address */
    ucp_address_t *worker_addr;
    size_t worker_addr_len;
    status = ucp_worker_get_address(res->worker, &worker_addr, &worker_addr_len);
    if (status != UCS_OK) {
        die_status("ucp_worker_get_address", status);
    }

    /* Exchange worker addresses */
    uint64_t len = worker_addr_len;
    tcp_send_all(sock, &len, sizeof(len));
    tcp_send_all(sock, worker_addr, worker_addr_len);

    uint64_t peer_len;
    tcp_recv_all(sock, &peer_len, sizeof(peer_len));
    void *peer_addr = malloc(peer_len);
    tcp_recv_all(sock, peer_addr, peer_len);

    /* Create endpoint to peer */
    ucp_ep_params_t ep_params = {0};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = peer_addr;

    status = ucp_ep_create(res->worker, &ep_params, &res->ep);
    free(peer_addr);
    ucp_worker_release_address(res->worker, worker_addr);

    if (status != UCS_OK) {
        die_status("ucp_ep_create", status);
    }

    /* Exchange memory info (address + rkey) */
    /* Send our recv buffer info to peer (they will write to it) */
    size_t info_size = sizeof(mem_info_t) + res->recv_rkey_len;
    mem_info_t *my_info = malloc(info_size);
    my_info->addr = (uint64_t)res->recv_buf;
    my_info->rkey_len = res->recv_rkey_len;
    memcpy((char*)my_info + sizeof(mem_info_t), res->recv_rkey_buf, res->recv_rkey_len);

    tcp_send_all(sock, &info_size, sizeof(info_size));
    tcp_send_all(sock, my_info, info_size);
    free(my_info);

    /* Receive peer's recv buffer info */
    size_t peer_info_size;
    tcp_recv_all(sock, &peer_info_size, sizeof(peer_info_size));
    mem_info_t *peer_info = malloc(peer_info_size);
    tcp_recv_all(sock, peer_info, peer_info_size);

    res->remote_addr = peer_info->addr;

    /* Unpack remote key */
    status = ucp_ep_rkey_unpack(res->ep,
                                 (char*)peer_info + sizeof(mem_info_t),
                                 &res->remote_rkey);
    free(peer_info);

    if (status != UCS_OK) {
        die_status("ucp_ep_rkey_unpack", status);
    }

    printf("Connection established\n");
}

/* ========================================================================== */
/* Completion Handling                                                         */
/* ========================================================================== */

static volatile int completed_count = 0;

static void send_callback(void *request, ucs_status_t status, void *user_data) {
    if (status != UCS_OK) {
        fprintf(stderr, "Send failed: %s\n", ucs_status_string(status));
    }
    __sync_fetch_and_add(&completed_count, 1);
    ucp_request_free(request);
}

/* ========================================================================== */
/* Bandwidth Test                                                              */
/* ========================================================================== */

static void run_put_bandwidth(ucp_resources_t *res, config_t *cfg) {
    printf("\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  PUT BANDWIDTH TEST (Zero-Copy RDMA Write)\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("Message size:     %zu bytes\n", cfg->msg_size);
    printf("Iterations:       %d\n", cfg->iterations);
    printf("Warmup:           %d\n", cfg->warmup);
    printf("Window size:      %d\n", cfg->window);
    printf("\n");

    /* Initialize send buffer with pattern */
    memset(res->send_buf, 0xAB, cfg->msg_size);

    /* Warmup */
    printf("Warming up...\n");
    for (int i = 0; i < cfg->warmup; i++) {
        ucs_status_ptr_t status_ptr = ucp_put_nbx(
            res->ep,
            res->send_buf,
            cfg->msg_size,
            res->remote_addr,
            res->remote_rkey,
            NULL
        );

        if (UCS_PTR_IS_ERR(status_ptr)) {
            die_status("ucp_put_nbx (warmup)", UCS_PTR_STATUS(status_ptr));
        }

        if (status_ptr != NULL) {
            while (ucp_request_check_status(status_ptr) == UCS_INPROGRESS) {
                ucp_worker_progress(res->worker);
            }
            ucp_request_free(status_ptr);
        }
    }

    /* Flush to ensure warmup completes */
    ucs_status_t status = ucp_worker_flush(res->worker);
    if (status != UCS_OK) {
        die_status("ucp_worker_flush", status);
    }

    /* Timed benchmark with pipelining */
    printf("Running benchmark...\n");

    ucp_request_param_t param = {0};
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
    param.cb.send = send_callback;

    completed_count = 0;
    int outstanding = 0;

    double start_time = get_time_us();

    for (int i = 0; i < cfg->iterations; i++) {
        /* Post operation */
        ucs_status_ptr_t status_ptr = ucp_put_nbx(
            res->ep,
            res->send_buf,
            cfg->msg_size,
            res->remote_addr,
            res->remote_rkey,
            &param
        );

        if (UCS_PTR_IS_ERR(status_ptr)) {
            die_status("ucp_put_nbx", UCS_PTR_STATUS(status_ptr));
        }

        if (status_ptr == NULL) {
            /* Completed immediately */
            completed_count++;
        } else {
            outstanding++;
        }

        /* Limit outstanding operations */
        while (outstanding >= cfg->window) {
            ucp_worker_progress(res->worker);
            int new_completed = completed_count;
            outstanding -= (new_completed - (i + 1 - outstanding));
        }
    }

    /* Wait for all to complete */
    while (completed_count < cfg->iterations) {
        ucp_worker_progress(res->worker);
    }

    double end_time = get_time_us();
    double elapsed_us = end_time - start_time;

    /* Calculate results */
    double total_bytes = (double)cfg->msg_size * cfg->iterations;
    double bandwidth_gbps = (total_bytes * 8.0) / (elapsed_us * 1000.0);
    double msg_rate = cfg->iterations / (elapsed_us / 1e6);
    double latency_us = elapsed_us / cfg->iterations;

    printf("\n");
    printf("────────────────────────────────────────────────────────────\n");
    printf("  RESULTS\n");
    printf("────────────────────────────────────────────────────────────\n");
    printf("Total time:       %.2f ms\n", elapsed_us / 1000.0);
    printf("Bandwidth:        %.2f Gbps\n", bandwidth_gbps);
    printf("Message rate:     %.0f msg/s\n", msg_rate);
    printf("Avg latency:      %.2f μs\n", latency_us);
    printf("────────────────────────────────────────────────────────────\n");
}

/* ========================================================================== */
/* Cleanup                                                                     */
/* ========================================================================== */

static void cleanup(ucp_resources_t *res) {
    if (res->remote_rkey) {
        ucp_rkey_destroy(res->remote_rkey);
    }
    if (res->ep) {
        ucp_ep_destroy(res->ep);
    }
    if (res->send_rkey_buf) {
        ucp_rkey_buffer_release(res->send_rkey_buf);
    }
    if (res->recv_rkey_buf) {
        ucp_rkey_buffer_release(res->recv_rkey_buf);
    }
    if (res->send_mem) {
        ucp_mem_unmap(res->context, res->send_mem);
    }
    if (res->recv_mem) {
        ucp_mem_unmap(res->context, res->recv_mem);
    }
    if (res->worker) {
        ucp_worker_destroy(res->worker);
    }
    if (res->context) {
        ucp_cleanup(res->context);
    }
    free(res->send_buf);
    free(res->recv_buf);
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

static void usage(const char *prog) {
    printf("UCX Maximum Bandwidth Benchmark\n");
    printf("\n");
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("Options:\n");
    printf("  -s              Run as server\n");
    printf("  -c <host>       Run as client, connect to <host>\n");
    printf("  -p <port>       Port number (default: %d)\n", DEFAULT_PORT);
    printf("  -m <size>       Message size in bytes (default: %d)\n", DEFAULT_MSG_SIZE);
    printf("  -n <iters>      Number of iterations (default: %d)\n", DEFAULT_ITERATIONS);
    printf("  -w <warmup>     Warmup iterations (default: %d)\n", DEFAULT_WARMUP);
    printf("  -W <window>     Window size / outstanding ops (default: %d)\n", DEFAULT_WINDOW);
    printf("  -a <cpu>        Pin to CPU core\n");
    printf("  -v              Verbose output\n");
    printf("  -h              Show this help\n");
    printf("\n");
    printf("Environment Variables:\n");
    printf("  UCX_TLS         Transport layer selection\n");
    printf("  UCX_NET_DEVICES Network device selection\n");
    printf("\n");
    printf("Examples:\n");
    printf("  Server:  %s -s\n", prog);
    printf("  Client:  %s -c 192.168.2.1 -m 16777216\n", prog);
    printf("\n");
    printf("  High performance:\n");
    printf("  UCX_TLS=rc_mlx5 %s -c 192.168.2.1 -m 1048576 -W 128 -a 0\n", prog);
}

int main(int argc, char **argv) {
    config_t cfg = {
        .is_server = -1,
        .server_addr = NULL,
        .port = DEFAULT_PORT,
        .msg_size = DEFAULT_MSG_SIZE,
        .iterations = DEFAULT_ITERATIONS,
        .warmup = DEFAULT_WARMUP,
        .window = DEFAULT_WINDOW,
        .cpu_affinity = -1,
        .verbose = 0
    };

    int opt;
    while ((opt = getopt(argc, argv, "sc:p:m:n:w:W:a:vh")) != -1) {
        switch (opt) {
            case 's':
                cfg.is_server = 1;
                break;
            case 'c':
                cfg.is_server = 0;
                cfg.server_addr = optarg;
                break;
            case 'p':
                cfg.port = atoi(optarg);
                break;
            case 'm':
                cfg.msg_size = atol(optarg);
                break;
            case 'n':
                cfg.iterations = atoi(optarg);
                break;
            case 'w':
                cfg.warmup = atoi(optarg);
                break;
            case 'W':
                cfg.window = atoi(optarg);
                break;
            case 'a':
                cfg.cpu_affinity = atoi(optarg);
                break;
            case 'v':
                cfg.verbose = 1;
                break;
            case 'h':
            default:
                usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }

    if (cfg.is_server < 0) {
        fprintf(stderr, "Error: Must specify -s (server) or -c <host> (client)\n\n");
        usage(argv[0]);
        return 1;
    }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  UCX Maximum Bandwidth Benchmark\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("Mode:             %s\n", cfg.is_server ? "Server" : "Client");
    if (!cfg.is_server) {
        printf("Server:           %s\n", cfg.server_addr);
    }
    printf("Port:             %d\n", cfg.port);
    printf("Message size:     %zu bytes\n", cfg.msg_size);
    printf("Iterations:       %d\n", cfg.iterations);
    printf("Window:           %d\n", cfg.window);
    printf("\n");

    /* Set CPU affinity if requested */
    set_cpu_affinity(cfg.cpu_affinity);

    /* Initialize UCX */
    ucp_resources_t res = {0};
    init_ucp_context(&res, cfg.msg_size);

    /* Establish connection via TCP bootstrap */
    int sock;
    if (cfg.is_server) {
        sock = tcp_accept(cfg.port);
    } else {
        sock = tcp_connect(cfg.server_addr, cfg.port);
    }

    exchange_addresses(sock, &res, cfg.is_server);
    close(sock);

    /* Run bandwidth test (client sends, server receives) */
    if (!cfg.is_server) {
        run_put_bandwidth(&res, &cfg);
    } else {
        /* Server waits for client to finish */
        printf("Waiting for client benchmark...\n");

        /* Poll for incoming data (server is passive receiver) */
        for (int i = 0; i < cfg.iterations + cfg.warmup + 1000; i++) {
            ucp_worker_progress(res.worker);
            usleep(100);
        }

        printf("Server completed\n");
    }

    /* Synchronize before cleanup */
    sock = cfg.is_server ? tcp_accept(cfg.port) : tcp_connect(cfg.server_addr, cfg.port);
    char sync = 'X';
    tcp_send_all(sock, &sync, 1);
    tcp_recv_all(sock, &sync, 1);
    close(sock);

    cleanup(&res);

    printf("\nDone.\n");
    return 0;
}
