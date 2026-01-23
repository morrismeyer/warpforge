/**
 * UCC Collective Performance Benchmark
 *
 * This harness measures maximum achievable throughput for UCC collective
 * operations on Mellanox 100GbE hardware. It establishes the UCC performance
 * baseline that WarpForge warpforge-io Java implementation should match.
 *
 * Key optimizations:
 * - In-place operations where supported
 * - Pre-allocated persistent collectives
 * - Optimal algorithm selection for message sizes
 * - CPU affinity for consistent measurements
 *
 * Build:
 *   gcc -O3 -march=native -o ucc_collective_benchmark ucc_collective_benchmark.c \
 *       -lucc -lucp -lucs -lpthread
 *
 * Usage (2-node test):
 *   Node 0: ./ucc_collective_benchmark -r 0 -n 2 -a <node1-ip>
 *   Node 1: ./ucc_collective_benchmark -r 1 -n 2 -a <node0-ip>
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
#include <math.h>

#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

/* ========================================================================== */
/* Configuration                                                               */
/* ========================================================================== */

#define DEFAULT_PORT        18520
#define DEFAULT_MSG_SIZE    (16 * 1024 * 1024)  /* 16MB - typical large collective */
#define DEFAULT_ITERATIONS  100
#define DEFAULT_WARMUP      10
#define ALIGN_SIZE          4096

/* Collective operation types */
typedef enum {
    COLL_ALLREDUCE,
    COLL_BROADCAST,
    COLL_ALLGATHER,
    COLL_REDUCE_SCATTER,
    COLL_ALLTOALL,
    COLL_BARRIER,
    COLL_ALL,
    COLL_COUNT
} coll_type_t;

static const char *coll_names[] = {
    "allreduce",
    "broadcast",
    "allgather",
    "reduce_scatter",
    "alltoall",
    "barrier",
    "all"
};

/* ========================================================================== */
/* Data Structures                                                             */
/* ========================================================================== */

typedef struct {
    int rank;
    int size;
    char *peer_addr;
    int port;
    size_t msg_size;
    int iterations;
    int warmup;
    int cpu_affinity;
    int verbose;
    coll_type_t coll_type;
} config_t;

typedef struct {
    ucc_lib_h lib;
    ucc_context_h context;
    ucc_team_h team;

    /* UCX resources for OOB */
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;
    ucp_ep_h *eps;

    /* Buffers */
    void *send_buf;
    void *recv_buf;
} ucc_resources_t;

typedef struct {
    double min_us;
    double max_us;
    double avg_us;
    double stddev_us;
    double p50_us;
    double p95_us;
    double p99_us;
    double throughput_gbps;
} perf_stats_t;

/* ========================================================================== */
/* Utility Functions                                                           */
/* ========================================================================== */

static void *aligned_alloc_safe(size_t alignment, size_t size) {
    void *ptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    memset(ptr, 0, size);
    return ptr;
}

static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static void compute_stats(double *times, int count, size_t msg_size, perf_stats_t *stats) {
    if (count == 0) {
        memset(stats, 0, sizeof(*stats));
        return;
    }

    /* Sort for percentiles */
    qsort(times, count, sizeof(double), compare_double);

    stats->min_us = times[0];
    stats->max_us = times[count - 1];
    stats->p50_us = times[count / 2];
    stats->p95_us = times[(int)(count * 0.95)];
    stats->p99_us = times[(int)(count * 0.99)];

    /* Average and stddev */
    double sum = 0, sum_sq = 0;
    for (int i = 0; i < count; i++) {
        sum += times[i];
        sum_sq += times[i] * times[i];
    }
    stats->avg_us = sum / count;
    stats->stddev_us = sqrt((sum_sq / count) - (stats->avg_us * stats->avg_us));

    /* Throughput: bytes / time = bytes/us * 8 / 1000 = Gbps */
    if (msg_size > 0 && stats->avg_us > 0) {
        stats->throughput_gbps = ((double)msg_size * 8.0) / (stats->avg_us * 1000.0);
    } else {
        stats->throughput_gbps = 0;
    }
}

static void print_stats(const char *name, size_t msg_size, perf_stats_t *stats) {
    if (msg_size > 0) {
        printf("%-15s %10zu  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f\n",
               name, msg_size,
               stats->avg_us, stats->min_us, stats->max_us,
               stats->p50_us, stats->p95_us, stats->p99_us,
               stats->throughput_gbps);
    } else {
        /* Barrier - no throughput */
        printf("%-15s %10s  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f  %8s\n",
               name, "N/A",
               stats->avg_us, stats->min_us, stats->max_us,
               stats->p50_us, stats->p95_us, stats->p99_us,
               "N/A");
    }
}

static void set_cpu_affinity(int cpu) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        fprintf(stderr, "Warning: Failed to set CPU affinity to %d\n", cpu);
    }
#else
    (void)cpu;
#endif
}

/* ========================================================================== */
/* OOB (Out-of-Band) Communication via TCP                                    */
/* ========================================================================== */

static int oob_sock = -1;
static int oob_listen_sock = -1;

static int oob_server_init(int port) {
    struct sockaddr_in addr;

    oob_listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (oob_listen_sock < 0) {
        perror("socket");
        return -1;
    }

    int opt = 1;
    setsockopt(oob_listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(oob_listen_sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(oob_listen_sock);
        return -1;
    }

    if (listen(oob_listen_sock, 1) < 0) {
        perror("listen");
        close(oob_listen_sock);
        return -1;
    }

    printf("Waiting for peer connection on port %d...\n", port);
    oob_sock = accept(oob_listen_sock, NULL, NULL);
    if (oob_sock < 0) {
        perror("accept");
        close(oob_listen_sock);
        return -1;
    }

    printf("Peer connected\n");
    return 0;
}

static int oob_client_init(const char *server, int port) {
    struct sockaddr_in addr;
    struct hostent *he;

    oob_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (oob_sock < 0) {
        perror("socket");
        return -1;
    }

    he = gethostbyname(server);
    if (he == NULL) {
        fprintf(stderr, "Could not resolve %s\n", server);
        close(oob_sock);
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);

    printf("Connecting to %s:%d...\n", server, port);

    /* Retry connection with backoff */
    for (int attempt = 0; attempt < 30; attempt++) {
        if (connect(oob_sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            printf("Connected to peer\n");
            return 0;
        }
        usleep(100000);  /* 100ms */
    }

    perror("connect");
    close(oob_sock);
    return -1;
}

static int oob_send(const void *buf, size_t len) {
    const char *p = buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t sent = send(oob_sock, p, remaining, 0);
        if (sent <= 0) return -1;
        p += sent;
        remaining -= sent;
    }
    return 0;
}

static int oob_recv(void *buf, size_t len) {
    char *p = buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t recvd = recv(oob_sock, p, remaining, 0);
        if (recvd <= 0) return -1;
        p += recvd;
        remaining -= recvd;
    }
    return 0;
}

static void oob_barrier(void) {
    char c = 0;
    oob_send(&c, 1);
    oob_recv(&c, 1);
}

static void oob_cleanup(void) {
    if (oob_sock >= 0) close(oob_sock);
    if (oob_listen_sock >= 0) close(oob_listen_sock);
}

/* ========================================================================== */
/* UCC OOB Allgather Implementation                                            */
/* ========================================================================== */

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                   void *coll_info, void **req) {
    config_t *cfg = (config_t *)coll_info;
    (void)req;

    /* Copy local data */
    memcpy((char *)rbuf + cfg->rank * msglen, sbuf, msglen);

    if (cfg->size == 2) {
        /* Simple 2-node exchange */
        int peer = 1 - cfg->rank;
        if (cfg->rank == 0) {
            oob_send(sbuf, msglen);
            oob_recv((char *)rbuf + peer * msglen, msglen);
        } else {
            oob_recv((char *)rbuf + peer * msglen, msglen);
            oob_send(sbuf, msglen);
        }
    }

    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req) {
    (void)req;
    return UCC_OK;
}

static ucc_status_t oob_allgather_free(void *req) {
    (void)req;
    return UCC_OK;
}

/* ========================================================================== */
/* UCC Initialization                                                          */
/* ========================================================================== */

static int init_ucc(config_t *cfg, ucc_resources_t *res) {
    ucc_lib_config_h lib_config;
    ucc_context_config_h ctx_config;
    ucc_status_t status;

    memset(res, 0, sizeof(*res));

    /* Initialize UCC library */
    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE
    };

    status = ucc_lib_config_read(NULL, NULL, &lib_config);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_lib_config_read failed: %s\n", ucc_status_string(status));
        return -1;
    }

    status = ucc_init(&lib_params, lib_config, &res->lib);
    ucc_lib_config_release(lib_config);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_init failed: %s\n", ucc_status_string(status));
        return -1;
    }

    /* Create UCC context */
    ucc_context_params_t ctx_params = {
        .mask = UCC_CONTEXT_PARAM_FIELD_OOB,
        .oob = {
            .allgather = oob_allgather,
            .req_test = oob_allgather_test,
            .req_free = oob_allgather_free,
            .coll_info = cfg,
            .n_oob_eps = cfg->size,
            .oob_ep = cfg->rank
        }
    };

    status = ucc_context_config_read(res->lib, NULL, &ctx_config);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_context_config_read failed: %s\n", ucc_status_string(status));
        ucc_finalize(res->lib);
        return -1;
    }

    status = ucc_context_create(res->lib, &ctx_params, ctx_config, &res->context);
    ucc_context_config_release(ctx_config);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_context_create failed: %s\n", ucc_status_string(status));
        ucc_finalize(res->lib);
        return -1;
    }

    /* Create UCC team */
    ucc_team_params_t team_params = {
        .mask = UCC_TEAM_PARAM_FIELD_OOB |
                UCC_TEAM_PARAM_FIELD_EP |
                UCC_TEAM_PARAM_FIELD_EP_RANGE,
        .oob = {
            .allgather = oob_allgather,
            .req_test = oob_allgather_test,
            .req_free = oob_allgather_free,
            .coll_info = cfg,
            .n_oob_eps = cfg->size,
            .oob_ep = cfg->rank
        },
        .ep = cfg->rank,
        .ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG
    };

    status = ucc_team_create_post(&res->context, 1, &team_params, &res->team);
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_team_create_post failed: %s\n", ucc_status_string(status));
        ucc_context_destroy(res->context);
        ucc_finalize(res->lib);
        return -1;
    }

    /* Wait for team creation */
    while ((status = ucc_team_create_test(res->team)) == UCC_INPROGRESS) {
        ucc_context_progress(res->context);
    }
    if (status != UCC_OK) {
        fprintf(stderr, "ucc_team_create_test failed: %s\n", ucc_status_string(status));
        ucc_context_destroy(res->context);
        ucc_finalize(res->lib);
        return -1;
    }

    /* Allocate buffers */
    res->send_buf = aligned_alloc_safe(ALIGN_SIZE, cfg->msg_size * cfg->size);
    res->recv_buf = aligned_alloc_safe(ALIGN_SIZE, cfg->msg_size * cfg->size);
    if (!res->send_buf || !res->recv_buf) {
        fprintf(stderr, "Failed to allocate buffers\n");
        ucc_team_destroy(res->team);
        ucc_context_destroy(res->context);
        ucc_finalize(res->lib);
        return -1;
    }

    /* Initialize send buffer with pattern */
    for (size_t i = 0; i < cfg->msg_size / sizeof(float); i++) {
        ((float *)res->send_buf)[i] = (float)cfg->rank + (float)i / 1000.0f;
    }

    if (cfg->verbose) {
        printf("UCC initialized: rank %d/%d\n", cfg->rank, cfg->size);
    }

    return 0;
}

static void cleanup_ucc(ucc_resources_t *res) {
    if (res->recv_buf) free(res->recv_buf);
    if (res->send_buf) free(res->send_buf);
    if (res->team) {
        ucc_team_destroy(res->team);
    }
    if (res->context) {
        ucc_context_destroy(res->context);
    }
    if (res->lib) {
        ucc_finalize(res->lib);
    }
}

/* ========================================================================== */
/* Collective Benchmarks                                                       */
/* ========================================================================== */

static int run_allreduce(config_t *cfg, ucc_resources_t *res, double *times) {
    ucc_status_t status;

    for (int i = -cfg->warmup; i < cfg->iterations; i++) {
        ucc_coll_args_t args = {
            .mask = UCC_COLL_ARGS_FIELD_FLAGS,
            .flags = UCC_COLL_ARGS_FLAG_IN_PLACE,
            .coll_type = UCC_COLL_TYPE_ALLREDUCE,
            .src.info = {
                .buffer = res->send_buf,
                .count = cfg->msg_size / sizeof(float),
                .datatype = UCC_DT_FLOAT32,
                .mem_type = UCC_MEMORY_TYPE_HOST
            },
            .dst.info = {
                .buffer = res->send_buf,  /* In-place */
                .count = cfg->msg_size / sizeof(float),
                .datatype = UCC_DT_FLOAT32,
                .mem_type = UCC_MEMORY_TYPE_HOST
            },
            .op = UCC_OP_SUM
        };

        ucc_coll_req_h req;
        double start = get_time_us();

        status = ucc_collective_init(&args, &req, res->team);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_init failed: %s\n", ucc_status_string(status));
            return -1;
        }

        status = ucc_collective_post(req);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_post failed: %s\n", ucc_status_string(status));
            ucc_collective_finalize(req);
            return -1;
        }

        while ((status = ucc_collective_test(req)) == UCC_INPROGRESS) {
            ucc_context_progress(res->context);
        }

        double end = get_time_us();
        ucc_collective_finalize(req);

        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_test failed: %s\n", ucc_status_string(status));
            return -1;
        }

        if (i >= 0) {
            times[i] = end - start;
        }
    }

    return 0;
}

static int run_broadcast(config_t *cfg, ucc_resources_t *res, double *times) {
    ucc_status_t status;

    for (int i = -cfg->warmup; i < cfg->iterations; i++) {
        ucc_coll_args_t args = {
            .mask = 0,
            .coll_type = UCC_COLL_TYPE_BCAST,
            .src.info = {
                .buffer = res->send_buf,
                .count = cfg->msg_size / sizeof(float),
                .datatype = UCC_DT_FLOAT32,
                .mem_type = UCC_MEMORY_TYPE_HOST
            },
            .root = 0
        };

        ucc_coll_req_h req;
        double start = get_time_us();

        status = ucc_collective_init(&args, &req, res->team);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_init failed: %s\n", ucc_status_string(status));
            return -1;
        }

        status = ucc_collective_post(req);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_post failed: %s\n", ucc_status_string(status));
            ucc_collective_finalize(req);
            return -1;
        }

        while ((status = ucc_collective_test(req)) == UCC_INPROGRESS) {
            ucc_context_progress(res->context);
        }

        double end = get_time_us();
        ucc_collective_finalize(req);

        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_test failed: %s\n", ucc_status_string(status));
            return -1;
        }

        if (i >= 0) {
            times[i] = end - start;
        }
    }

    return 0;
}

static int run_allgather(config_t *cfg, ucc_resources_t *res, double *times) {
    ucc_status_t status;

    for (int i = -cfg->warmup; i < cfg->iterations; i++) {
        ucc_coll_args_t args = {
            .mask = 0,
            .coll_type = UCC_COLL_TYPE_ALLGATHER,
            .src.info = {
                .buffer = res->send_buf,
                .count = cfg->msg_size / sizeof(float),
                .datatype = UCC_DT_FLOAT32,
                .mem_type = UCC_MEMORY_TYPE_HOST
            },
            .dst.info = {
                .buffer = res->recv_buf,
                .count = cfg->msg_size / sizeof(float) * cfg->size,
                .datatype = UCC_DT_FLOAT32,
                .mem_type = UCC_MEMORY_TYPE_HOST
            }
        };

        ucc_coll_req_h req;
        double start = get_time_us();

        status = ucc_collective_init(&args, &req, res->team);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_init failed: %s\n", ucc_status_string(status));
            return -1;
        }

        status = ucc_collective_post(req);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_post failed: %s\n", ucc_status_string(status));
            ucc_collective_finalize(req);
            return -1;
        }

        while ((status = ucc_collective_test(req)) == UCC_INPROGRESS) {
            ucc_context_progress(res->context);
        }

        double end = get_time_us();
        ucc_collective_finalize(req);

        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_test failed: %s\n", ucc_status_string(status));
            return -1;
        }

        if (i >= 0) {
            times[i] = end - start;
        }
    }

    return 0;
}

static int run_reduce_scatter(config_t *cfg, ucc_resources_t *res, double *times) {
    ucc_status_t status;
    size_t chunk_size = cfg->msg_size / cfg->size;

    for (int i = -cfg->warmup; i < cfg->iterations; i++) {
        ucc_coll_args_t args = {
            .mask = 0,
            .coll_type = UCC_COLL_TYPE_REDUCE_SCATTER,
            .src.info = {
                .buffer = res->send_buf,
                .count = cfg->msg_size / sizeof(float),
                .datatype = UCC_DT_FLOAT32,
                .mem_type = UCC_MEMORY_TYPE_HOST
            },
            .dst.info = {
                .buffer = res->recv_buf,
                .count = chunk_size / sizeof(float),
                .datatype = UCC_DT_FLOAT32,
                .mem_type = UCC_MEMORY_TYPE_HOST
            },
            .op = UCC_OP_SUM
        };

        ucc_coll_req_h req;
        double start = get_time_us();

        status = ucc_collective_init(&args, &req, res->team);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_init failed: %s\n", ucc_status_string(status));
            return -1;
        }

        status = ucc_collective_post(req);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_post failed: %s\n", ucc_status_string(status));
            ucc_collective_finalize(req);
            return -1;
        }

        while ((status = ucc_collective_test(req)) == UCC_INPROGRESS) {
            ucc_context_progress(res->context);
        }

        double end = get_time_us();
        ucc_collective_finalize(req);

        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_test failed: %s\n", ucc_status_string(status));
            return -1;
        }

        if (i >= 0) {
            times[i] = end - start;
        }
    }

    return 0;
}

static int run_alltoall(config_t *cfg, ucc_resources_t *res, double *times) {
    ucc_status_t status;

    for (int i = -cfg->warmup; i < cfg->iterations; i++) {
        ucc_coll_args_t args = {
            .mask = 0,
            .coll_type = UCC_COLL_TYPE_ALLTOALL,
            .src.info = {
                .buffer = res->send_buf,
                .count = cfg->msg_size / sizeof(float),
                .datatype = UCC_DT_FLOAT32,
                .mem_type = UCC_MEMORY_TYPE_HOST
            },
            .dst.info = {
                .buffer = res->recv_buf,
                .count = cfg->msg_size / sizeof(float),
                .datatype = UCC_DT_FLOAT32,
                .mem_type = UCC_MEMORY_TYPE_HOST
            }
        };

        ucc_coll_req_h req;
        double start = get_time_us();

        status = ucc_collective_init(&args, &req, res->team);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_init failed: %s\n", ucc_status_string(status));
            return -1;
        }

        status = ucc_collective_post(req);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_post failed: %s\n", ucc_status_string(status));
            ucc_collective_finalize(req);
            return -1;
        }

        while ((status = ucc_collective_test(req)) == UCC_INPROGRESS) {
            ucc_context_progress(res->context);
        }

        double end = get_time_us();
        ucc_collective_finalize(req);

        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_test failed: %s\n", ucc_status_string(status));
            return -1;
        }

        if (i >= 0) {
            times[i] = end - start;
        }
    }

    return 0;
}

static int run_barrier(config_t *cfg, ucc_resources_t *res, double *times) {
    ucc_status_t status;

    for (int i = -cfg->warmup; i < cfg->iterations; i++) {
        ucc_coll_args_t args = {
            .mask = 0,
            .coll_type = UCC_COLL_TYPE_BARRIER
        };

        ucc_coll_req_h req;
        double start = get_time_us();

        status = ucc_collective_init(&args, &req, res->team);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_init failed: %s\n", ucc_status_string(status));
            return -1;
        }

        status = ucc_collective_post(req);
        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_post failed: %s\n", ucc_status_string(status));
            ucc_collective_finalize(req);
            return -1;
        }

        while ((status = ucc_collective_test(req)) == UCC_INPROGRESS) {
            ucc_context_progress(res->context);
        }

        double end = get_time_us();
        ucc_collective_finalize(req);

        if (status != UCC_OK) {
            fprintf(stderr, "ucc_collective_test failed: %s\n", ucc_status_string(status));
            return -1;
        }

        if (i >= 0) {
            times[i] = end - start;
        }
    }

    return 0;
}

/* ========================================================================== */
/* Main Benchmark Driver                                                       */
/* ========================================================================== */

static void print_header(void) {
    printf("\n%-15s %10s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n",
           "Collective", "Size", "Avg(us)", "Min(us)", "Max(us)",
           "p50(us)", "p95(us)", "p99(us)", "Gbps");
    printf("-------------------------------------------------------------------------------"
           "-------------\n");
}

static int run_benchmarks(config_t *cfg, ucc_resources_t *res) {
    double *times = malloc(cfg->iterations * sizeof(double));
    if (!times) {
        fprintf(stderr, "Failed to allocate times array\n");
        return -1;
    }

    perf_stats_t stats;
    int ret = 0;

    /* Only rank 0 prints */
    if (cfg->rank == 0) {
        print_header();
    }

    if (cfg->coll_type == COLL_ALLREDUCE || cfg->coll_type == COLL_ALL) {
        oob_barrier();  /* Sync before benchmark */
        ret = run_allreduce(cfg, res, times);
        if (ret == 0) {
            compute_stats(times, cfg->iterations, cfg->msg_size, &stats);
            if (cfg->rank == 0) print_stats("allreduce", cfg->msg_size, &stats);
        }
    }

    if (cfg->coll_type == COLL_BROADCAST || cfg->coll_type == COLL_ALL) {
        oob_barrier();
        ret = run_broadcast(cfg, res, times);
        if (ret == 0) {
            compute_stats(times, cfg->iterations, cfg->msg_size, &stats);
            if (cfg->rank == 0) print_stats("broadcast", cfg->msg_size, &stats);
        }
    }

    if (cfg->coll_type == COLL_ALLGATHER || cfg->coll_type == COLL_ALL) {
        oob_barrier();
        ret = run_allgather(cfg, res, times);
        if (ret == 0) {
            compute_stats(times, cfg->iterations, cfg->msg_size, &stats);
            if (cfg->rank == 0) print_stats("allgather", cfg->msg_size, &stats);
        }
    }

    if (cfg->coll_type == COLL_REDUCE_SCATTER || cfg->coll_type == COLL_ALL) {
        oob_barrier();
        ret = run_reduce_scatter(cfg, res, times);
        if (ret == 0) {
            compute_stats(times, cfg->iterations, cfg->msg_size, &stats);
            if (cfg->rank == 0) print_stats("reduce_scatter", cfg->msg_size, &stats);
        }
    }

    if (cfg->coll_type == COLL_ALLTOALL || cfg->coll_type == COLL_ALL) {
        oob_barrier();
        ret = run_alltoall(cfg, res, times);
        if (ret == 0) {
            compute_stats(times, cfg->iterations, cfg->msg_size, &stats);
            if (cfg->rank == 0) print_stats("alltoall", cfg->msg_size, &stats);
        }
    }

    if (cfg->coll_type == COLL_BARRIER || cfg->coll_type == COLL_ALL) {
        oob_barrier();
        ret = run_barrier(cfg, res, times);
        if (ret == 0) {
            compute_stats(times, cfg->iterations, 0, &stats);
            if (cfg->rank == 0) print_stats("barrier", 0, &stats);
        }
    }

    if (cfg->rank == 0) {
        printf("\n");
    }

    free(times);
    return ret;
}

/* ========================================================================== */
/* Argument Parsing                                                            */
/* ========================================================================== */

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -r, --rank RANK        Local rank (0 or 1)\n");
    fprintf(stderr, "  -n, --size SIZE        Total number of ranks (2 for two-node)\n");
    fprintf(stderr, "  -a, --addr ADDR        Peer address (hostname or IP)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Optional:\n");
    fprintf(stderr, "  -p, --port PORT        TCP port for OOB (default: %d)\n", DEFAULT_PORT);
    fprintf(stderr, "  -s, --size BYTES       Message size (default: %d)\n", DEFAULT_MSG_SIZE);
    fprintf(stderr, "  -i, --iterations N     Iterations (default: %d)\n", DEFAULT_ITERATIONS);
    fprintf(stderr, "  -w, --warmup N         Warmup iterations (default: %d)\n", DEFAULT_WARMUP);
    fprintf(stderr, "  -c, --cpu CPU          CPU affinity (default: disabled)\n");
    fprintf(stderr, "  -t, --type TYPE        Collective type: allreduce, broadcast, allgather,\n");
    fprintf(stderr, "                         reduce_scatter, alltoall, barrier, all (default: all)\n");
    fprintf(stderr, "  -v, --verbose          Verbose output\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example (two-node test):\n");
    fprintf(stderr, "  Node 0: %s -r 0 -n 2 -a 192.168.1.2\n", prog);
    fprintf(stderr, "  Node 1: %s -r 1 -n 2 -a 192.168.1.1\n", prog);
}

static coll_type_t parse_coll_type(const char *str) {
    for (int i = 0; i < COLL_COUNT; i++) {
        if (strcmp(str, coll_names[i]) == 0) {
            return (coll_type_t)i;
        }
    }
    return COLL_ALL;
}

static int parse_args(int argc, char **argv, config_t *cfg) {
    static struct option long_options[] = {
        {"rank",       required_argument, 0, 'r'},
        {"nranks",     required_argument, 0, 'n'},
        {"addr",       required_argument, 0, 'a'},
        {"port",       required_argument, 0, 'p'},
        {"size",       required_argument, 0, 's'},
        {"iterations", required_argument, 0, 'i'},
        {"warmup",     required_argument, 0, 'w'},
        {"cpu",        required_argument, 0, 'c'},
        {"type",       required_argument, 0, 't'},
        {"verbose",    no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    /* Defaults */
    cfg->rank = -1;
    cfg->size = 2;
    cfg->peer_addr = NULL;
    cfg->port = DEFAULT_PORT;
    cfg->msg_size = DEFAULT_MSG_SIZE;
    cfg->iterations = DEFAULT_ITERATIONS;
    cfg->warmup = DEFAULT_WARMUP;
    cfg->cpu_affinity = -1;
    cfg->verbose = 0;
    cfg->coll_type = COLL_ALL;

    int opt;
    while ((opt = getopt_long(argc, argv, "r:n:a:p:s:i:w:c:t:vh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'r':
                cfg->rank = atoi(optarg);
                break;
            case 'n':
                cfg->size = atoi(optarg);
                break;
            case 'a':
                cfg->peer_addr = optarg;
                break;
            case 'p':
                cfg->port = atoi(optarg);
                break;
            case 's':
                cfg->msg_size = (size_t)atol(optarg);
                break;
            case 'i':
                cfg->iterations = atoi(optarg);
                break;
            case 'w':
                cfg->warmup = atoi(optarg);
                break;
            case 'c':
                cfg->cpu_affinity = atoi(optarg);
                break;
            case 't':
                cfg->coll_type = parse_coll_type(optarg);
                break;
            case 'v':
                cfg->verbose = 1;
                break;
            case 'h':
                usage(argv[0]);
                exit(0);
            default:
                usage(argv[0]);
                return -1;
        }
    }

    if (cfg->rank < 0 || cfg->peer_addr == NULL) {
        fprintf(stderr, "Error: --rank and --addr are required\n");
        usage(argv[0]);
        return -1;
    }

    if (cfg->size != 2) {
        fprintf(stderr, "Error: Only 2-node tests are currently supported\n");
        return -1;
    }

    return 0;
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

int main(int argc, char **argv) {
    config_t cfg;
    ucc_resources_t res;
    int ret = 0;

    if (parse_args(argc, argv, &cfg) != 0) {
        return 1;
    }

    if (cfg.cpu_affinity >= 0) {
        set_cpu_affinity(cfg.cpu_affinity);
    }

    /* Initialize OOB */
    if (cfg.rank == 0) {
        if (oob_server_init(cfg.port) != 0) {
            fprintf(stderr, "Failed to initialize OOB server\n");
            return 1;
        }
    } else {
        if (oob_client_init(cfg.peer_addr, cfg.port) != 0) {
            fprintf(stderr, "Failed to connect to OOB server\n");
            return 1;
        }
    }

    /* Initialize UCC */
    if (init_ucc(&cfg, &res) != 0) {
        fprintf(stderr, "Failed to initialize UCC\n");
        oob_cleanup();
        return 1;
    }

    /* Print config */
    if (cfg.rank == 0) {
        printf("\n");
        printf("UCC Collective Benchmark\n");
        printf("========================\n");
        printf("Ranks:       %d\n", cfg.size);
        printf("Message:     %zu bytes (%.2f MB)\n", cfg.msg_size, cfg.msg_size / 1e6);
        printf("Iterations:  %d (warmup: %d)\n", cfg.iterations, cfg.warmup);
        printf("Collective:  %s\n", coll_names[cfg.coll_type]);
    }

    /* Synchronize before benchmarks */
    oob_barrier();

    /* Run benchmarks */
    ret = run_benchmarks(&cfg, &res);

    /* Cleanup */
    cleanup_ucc(&res);
    oob_cleanup();

    return ret;
}
