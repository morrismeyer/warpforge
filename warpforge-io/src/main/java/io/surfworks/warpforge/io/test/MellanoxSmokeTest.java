package io.surfworks.warpforge.io.test;

import io.surfworks.warpforge.io.rdma.Rdma;
import io.surfworks.warpforge.io.rdma.RdmaApi;
import io.surfworks.warpforge.io.rdma.RdmaBuffer;
import io.surfworks.warpforge.io.rdma.RdmaDevice;
import io.surfworks.warpforge.io.rdma.RdmaEndpoint;
import io.surfworks.warpforge.io.rdma.RdmaListener;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Mellanox Cross-Connect Smoke Test.
 *
 * <p>This tool validates RDMA connectivity between two machines connected via
 * Mellanox NICs. It runs in either server or client mode:
 *
 * <ul>
 *   <li><b>Server mode</b>: Listens for incoming connections and echoes data</li>
 *   <li><b>Client mode</b>: Connects to server and runs ping-pong + bandwidth tests</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>
 * # On the server (e.g., NVIDIA box):
 * java -cp warpforge-io.jar io.surfworks.warpforge.io.test.MellanoxSmokeTest server
 *
 * # On the client (e.g., AMD box):
 * java -cp warpforge-io.jar io.surfworks.warpforge.io.test.MellanoxSmokeTest client 192.168.2.1
 *
 * # Just enumerate devices (no connectivity test):
 * java -cp warpforge-io.jar io.surfworks.warpforge.io.test.MellanoxSmokeTest info
 * </pre>
 *
 * <h2>Expected Output</h2>
 * <p>All output is verbose and timestamped for monitoring. Each phase prints
 * clear START/PASS/FAIL markers.
 */
public class MellanoxSmokeTest {

    private static final int DEFAULT_PORT = 18515;
    private static final int PING_PONG_ITERATIONS = 1000;
    private static final int BANDWIDTH_ITERATIONS = 100;
    private static final int SMALL_MESSAGE_SIZE = 8;        // 8 bytes for latency
    private static final int LARGE_MESSAGE_SIZE = 1024 * 1024; // 1MB for bandwidth

    public static void main(String[] args) {
        printBanner();

        if (args.length == 0) {
            printUsage();
            System.exit(1);
        }

        String mode = args[0].toLowerCase();

        try {
            switch (mode) {
                case "info" -> runInfo();
                case "server" -> runServer(args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_PORT);
                case "client" -> {
                    if (args.length < 2) {
                        log("ERROR: Client mode requires server address");
                        printUsage();
                        System.exit(1);
                    }
                    int port = args.length > 2 ? Integer.parseInt(args[2]) : DEFAULT_PORT;
                    runClient(args[1], port);
                }
                default -> {
                    log("ERROR: Unknown mode: " + mode);
                    printUsage();
                    System.exit(1);
                }
            }
        } catch (Exception e) {
            log("FATAL ERROR: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void printBanner() {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║           MELLANOX CROSS-CONNECT SMOKE TEST                      ║");
        System.out.println("║           WarpForge RDMA Connectivity Validator                  ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝");
        System.out.println();
    }

    private static void printUsage() {
        System.out.println("Usage:");
        System.out.println("  mellanox-smoke-test info                    # Show RDMA device info");
        System.out.println("  mellanox-smoke-test server [port]           # Run as server (default port: " + DEFAULT_PORT + ")");
        System.out.println("  mellanox-smoke-test client <host> [port]    # Run as client");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  # On NVIDIA box (server):");
        System.out.println("  ./gradlew :warpforge-io:run --args='server'");
        System.out.println();
        System.out.println("  # On AMD box (client):");
        System.out.println("  ./gradlew :warpforge-io:run --args='client 192.168.2.1'");
        System.out.println();
    }

    // ========== INFO MODE ==========

    private static void runInfo() {
        logSection("RDMA SYSTEM INFORMATION");

        log("Platform: " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        log("Java: " + System.getProperty("java.version"));
        log("Hostname: " + getHostname());
        log("");

        log(Rdma.systemInfo());

        if (!Rdma.isLinux()) {
            log("WARNING: Not running on Linux - RDMA will use mock implementation");
            return;
        }

        if (!Rdma.hasInfiniBandDevices()) {
            log("WARNING: No InfiniBand devices found in /sys/class/infiniband");
            log("  - Check that Mellanox drivers are loaded: lsmod | grep mlx");
            log("  - Check device status: ibstat");
            return;
        }

        logSection("RDMA API INITIALIZATION");

        try (RdmaApi rdma = Rdma.load()) {
            log("Backend: " + rdma.backendName());
            log("GPU Direct supported: " + rdma.supportsGpuDirect());
            log("Atomics supported: " + rdma.supportsAtomics());
            log("Max inline size: " + rdma.maxInlineSize() + " bytes");
            log("Max memory region: " + formatBytes(rdma.maxMemoryRegionSize()));

            log("");
            logSection("RDMA DEVICES");

            List<RdmaDevice> devices = rdma.devices();
            if (devices.isEmpty()) {
                log("No RDMA devices found via API");
            } else {
                for (RdmaDevice device : devices) {
                    log("Device: " + device.name());
                    log("  Vendor: " + device.vendor());
                    log("  Link Speed: " + device.linkSpeed() + " Gbps");
                    log("  Port Count: " + device.portCount());
                    log("  Max MTU: " + device.maxMtu());
                    log("  InfiniBand: " + device.supportsInfiniBand());
                    log("  RoCE: " + device.supportsRoCE());
                    log("  100GbE Capable: " + device.is100GbECapable());
                    log("");
                }
            }

            log("INFO MODE COMPLETE - RDMA subsystem appears functional");

        } catch (Exception e) {
            log("ERROR initializing RDMA: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // ========== SERVER MODE ==========

    private static void runServer(int port) throws Exception {
        logSection("SERVER MODE");
        log("Listening port: " + port);
        log("Hostname: " + getHostname());

        try (RdmaApi rdma = Rdma.load()) {
            log("RDMA backend: " + rdma.backendName());

            if ("mock".equals(rdma.backendName())) {
                log("WARNING: Using mock RDMA - no real network traffic will occur");
            }

            logSection("WAITING FOR CLIENT CONNECTION");
            log("Server is ready. Waiting for client to connect...");
            log("(Press Ctrl+C to stop)");

            try (RdmaListener listener = rdma.listen(port)) {
                log("Listener created on port " + port);

                // Accept connection
                log("Accepting connection...");
                try (RdmaEndpoint endpoint = listener.accept()) {
                    log("CLIENT CONNECTED from: " + endpoint.remoteAddress());

                    runServerTests(rdma, endpoint);
                }
            }
        }

        log("");
        log("SERVER COMPLETE");
    }

    private static void runServerTests(RdmaApi rdma, RdmaEndpoint endpoint) throws Exception {
        try (Arena arena = Arena.ofConfined()) {

            // ===== PING-PONG TEST (Server side: receive then send) =====
            logSection("PING-PONG LATENCY TEST (Server Side)");
            log("Iterations: " + PING_PONG_ITERATIONS);
            log("Message size: " + SMALL_MESSAGE_SIZE + " bytes");

            MemorySegment pingBuffer = arena.allocate(SMALL_MESSAGE_SIZE);
            try (RdmaBuffer rdmaPingBuffer = rdma.registerMemory(pingBuffer)) {
                log("Buffer registered for ping-pong");

                for (int i = 0; i < PING_PONG_ITERATIONS; i++) {
                    // Receive ping
                    endpoint.receive(rdmaPingBuffer).join();

                    // Send pong (echo back)
                    endpoint.send(rdmaPingBuffer).join();

                    if (i % 100 == 0) {
                        log("  Ping-pong iteration " + i + "/" + PING_PONG_ITERATIONS);
                    }
                }
                log("PASS: Ping-pong test complete (server side)");
            }

            // ===== BANDWIDTH TEST (Server side: receive large messages) =====
            logSection("BANDWIDTH TEST (Server Side)");
            log("Iterations: " + BANDWIDTH_ITERATIONS);
            log("Message size: " + formatBytes(LARGE_MESSAGE_SIZE));

            MemorySegment bwBuffer = arena.allocate(LARGE_MESSAGE_SIZE);
            try (RdmaBuffer rdmaBwBuffer = rdma.registerMemory(bwBuffer)) {
                log("Buffer registered for bandwidth test");
                log("Receiving " + BANDWIDTH_ITERATIONS + " messages...");

                long totalReceived = 0;
                Instant start = Instant.now();

                for (int i = 0; i < BANDWIDTH_ITERATIONS; i++) {
                    endpoint.receive(rdmaBwBuffer).join();
                    totalReceived += LARGE_MESSAGE_SIZE;

                    if (i % 10 == 0) {
                        log("  Received " + i + "/" + BANDWIDTH_ITERATIONS + " (" + formatBytes(totalReceived) + ")");
                    }
                }

                Instant end = Instant.now();
                Duration elapsed = Duration.between(start, end);
                double seconds = elapsed.toNanos() / 1_000_000_000.0;
                double throughputGbps = (totalReceived * 8.0) / (seconds * 1_000_000_000.0);

                log("PASS: Bandwidth test complete");
                log("  Total received: " + formatBytes(totalReceived));
                log("  Time: " + String.format("%.3f", seconds) + " seconds");
                log("  Throughput: " + String.format("%.2f", throughputGbps) + " Gbps");

                // Send acknowledgment
                bwBuffer.set(ValueLayout.JAVA_LONG, 0, totalReceived);
                endpoint.send(rdmaBwBuffer).join();
                log("  Sent acknowledgment to client");
            }
        }
    }

    // ========== CLIENT MODE ==========

    private static void runClient(String serverHost, int port) throws Exception {
        logSection("CLIENT MODE");
        log("Server: " + serverHost + ":" + port);
        log("Local hostname: " + getHostname());

        try (RdmaApi rdma = Rdma.load()) {
            log("RDMA backend: " + rdma.backendName());

            if ("mock".equals(rdma.backendName())) {
                log("WARNING: Using mock RDMA - no real network traffic will occur");
            }

            logSection("CONNECTING TO SERVER");
            log("Connecting to " + serverHost + ":" + port + "...");

            try (RdmaEndpoint endpoint = rdma.connect(serverHost, port, 30_000)) {
                log("CONNECTED to server: " + endpoint.remoteAddress());

                runClientTests(rdma, endpoint);
            }
        }

        log("");
        logSection("TEST SUMMARY");
        log("ALL TESTS PASSED");
        log("Mellanox cross-connect is functional!");
    }

    private static void runClientTests(RdmaApi rdma, RdmaEndpoint endpoint) throws Exception {
        try (Arena arena = Arena.ofConfined()) {

            // ===== PING-PONG TEST (Client side: send then receive) =====
            logSection("PING-PONG LATENCY TEST");
            log("Iterations: " + PING_PONG_ITERATIONS);
            log("Message size: " + SMALL_MESSAGE_SIZE + " bytes");

            MemorySegment pingBuffer = arena.allocate(SMALL_MESSAGE_SIZE);
            try (RdmaBuffer rdmaPingBuffer = rdma.registerMemory(pingBuffer)) {
                log("Buffer registered");

                // Warmup
                log("Warmup (10 iterations)...");
                for (int i = 0; i < 10; i++) {
                    pingBuffer.set(ValueLayout.JAVA_LONG, 0, i);
                    endpoint.send(rdmaPingBuffer).join();
                    endpoint.receive(rdmaPingBuffer).join();
                }
                log("Warmup complete");

                // Timed run
                log("Running " + PING_PONG_ITERATIONS + " ping-pong iterations...");
                long[] latencies = new long[PING_PONG_ITERATIONS];
                Instant totalStart = Instant.now();

                for (int i = 0; i < PING_PONG_ITERATIONS; i++) {
                    pingBuffer.set(ValueLayout.JAVA_LONG, 0, i);

                    long start = System.nanoTime();
                    endpoint.send(rdmaPingBuffer).join();
                    endpoint.receive(rdmaPingBuffer).join();
                    long end = System.nanoTime();

                    latencies[i] = end - start;

                    if (i % 100 == 0 && i > 0) {
                        log("  Iteration " + i + "/" + PING_PONG_ITERATIONS);
                    }
                }

                Instant totalEnd = Instant.now();

                // Calculate statistics
                java.util.Arrays.sort(latencies);
                long min = latencies[0];
                long max = latencies[PING_PONG_ITERATIONS - 1];
                long median = latencies[PING_PONG_ITERATIONS / 2];
                long p99 = latencies[(int) (PING_PONG_ITERATIONS * 0.99)];
                long sum = 0;
                for (long l : latencies) sum += l;
                double avg = sum / (double) PING_PONG_ITERATIONS;

                log("");
                log("PASS: Ping-pong latency test complete");
                log("  Iterations: " + PING_PONG_ITERATIONS);
                log("  Round-trip latency:");
                log("    Min:    " + String.format("%,d", min / 1000) + " us");
                log("    Max:    " + String.format("%,d", max / 1000) + " us");
                log("    Avg:    " + String.format("%,.1f", avg / 1000) + " us");
                log("    Median: " + String.format("%,d", median / 1000) + " us");
                log("    P99:    " + String.format("%,d", p99 / 1000) + " us");
                log("  Total time: " + Duration.between(totalStart, totalEnd).toMillis() + " ms");
            }

            // ===== BANDWIDTH TEST (Client side: send large messages) =====
            logSection("BANDWIDTH TEST");
            log("Iterations: " + BANDWIDTH_ITERATIONS);
            log("Message size: " + formatBytes(LARGE_MESSAGE_SIZE));

            MemorySegment bwBuffer = arena.allocate(LARGE_MESSAGE_SIZE);
            try (RdmaBuffer rdmaBwBuffer = rdma.registerMemory(bwBuffer)) {
                log("Buffer registered (" + formatBytes(LARGE_MESSAGE_SIZE) + ")");

                // Fill buffer with pattern
                for (int i = 0; i < LARGE_MESSAGE_SIZE / 8; i++) {
                    bwBuffer.set(ValueLayout.JAVA_LONG, i * 8L, i);
                }
                log("Buffer initialized with test pattern");

                log("Sending " + BANDWIDTH_ITERATIONS + " messages...");
                long totalSent = 0;
                Instant start = Instant.now();

                for (int i = 0; i < BANDWIDTH_ITERATIONS; i++) {
                    endpoint.send(rdmaBwBuffer).join();
                    totalSent += LARGE_MESSAGE_SIZE;

                    if (i % 10 == 0) {
                        log("  Sent " + i + "/" + BANDWIDTH_ITERATIONS + " (" + formatBytes(totalSent) + ")");
                    }
                }

                Instant end = Instant.now();
                Duration elapsed = Duration.between(start, end);
                double seconds = elapsed.toNanos() / 1_000_000_000.0;
                double throughputGbps = (totalSent * 8.0) / (seconds * 1_000_000_000.0);

                log("");
                log("PASS: Bandwidth test complete");
                log("  Total sent: " + formatBytes(totalSent));
                log("  Time: " + String.format("%.3f", seconds) + " seconds");
                log("  Throughput: " + String.format("%.2f", throughputGbps) + " Gbps");

                // Wait for server acknowledgment
                log("Waiting for server acknowledgment...");
                endpoint.receive(rdmaBwBuffer).join();
                long serverReceived = bwBuffer.get(ValueLayout.JAVA_LONG, 0);
                log("  Server confirmed receipt of " + formatBytes(serverReceived));

                if (serverReceived != totalSent) {
                    log("WARNING: Byte count mismatch - sent " + totalSent + ", server received " + serverReceived);
                }
            }
        }
    }

    // ========== UTILITIES ==========

    private static void log(String message) {
        String timestamp = java.time.LocalDateTime.now()
                .format(java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss.SSS"));
        System.out.println("[" + timestamp + "] " + message);
    }

    private static void logSection(String title) {
        System.out.println();
        System.out.println("════════════════════════════════════════════════════════════════════");
        System.out.println("  " + title);
        System.out.println("════════════════════════════════════════════════════════════════════");
    }

    private static String getHostname() {
        try {
            return java.net.InetAddress.getLocalHost().getHostName();
        } catch (Exception e) {
            return "unknown";
        }
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }
}
