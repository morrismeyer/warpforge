package io.surfworks.warpforge.io.perf;

import io.surfworks.warpforge.io.rdma.Rdma;
import io.surfworks.warpforge.io.rdma.RdmaApi;
import io.surfworks.warpforge.io.rdma.RdmaBuffer;
import io.surfworks.warpforge.io.rdma.RdmaEndpoint;
import io.surfworks.warpforge.io.rdma.RdmaListener;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * RDMA Performance Tests for WarpForge.
 *
 * <p>Validates that the Java RDMA implementation achieves performance close to
 * the Linux baseline measured with native perftest tools (ib_write_bw, ib_send_lat, etc.).
 *
 * <h2>Hardware Baseline (from CONNECTX5.md)</h2>
 * <pre>
 * Hardware: Mellanox ConnectX-5 @ 100 Gbps link rate
 * Actual: PCIe 3.0 x8 bottleneck = 62.5 Gbps theoretical
 *
 * Bandwidth (1MB messages, 5000 iterations):
 *   - RDMA Write: 55.87 Gbps (89% of PCIe max)
 *   - RDMA Read:  55.97 Gbps (90% of PCIe max)
 *   - Send:       55.88 Gbps (89% of PCIe max)
 *
 * Latency (8-byte messages, 10000 iterations):
 *   - RDMA Write: 0.93 μs min, 0.99 μs typical, 1.00 μs avg, 1.15 μs P99
 *   - Send:       0.89 μs min, 0.92 μs typical, 0.92 μs avg, 0.95 μs P99
 * </pre>
 *
 * <h2>Performance Targets</h2>
 * <ul>
 *   <li>Bandwidth: ≥95% of Linux baseline (52.5 Gbps)</li>
 *   <li>Latency: ≤120% of Linux baseline (1.2 μs avg)</li>
 * </ul>
 *
 * <h2>Running</h2>
 * <pre>{@code
 * # Start server on remote node:
 * ./gradlew :warpforge-io:run --args='server'
 *
 * # Run performance tests:
 * ./gradlew :warpforge-io:rdmaPerfTest \
 *     -Drdma.server.host=10.0.0.1 \
 *     -Drdma.server.port=18515
 * }</pre>
 */
@Tag("rdma-perf")
@Tag("rdma")
@DisplayName("RDMA Performance Tests")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class RdmaPerformanceTest {

    // ===== Linux Baseline (from CONNECTX5.md) =====

    /** Linux baseline: RDMA Write bandwidth in Gbps */
    private static final double LINUX_WRITE_BW_GBPS = 55.87;

    /** Linux baseline: RDMA Read bandwidth in Gbps */
    private static final double LINUX_READ_BW_GBPS = 55.97;

    /** Linux baseline: Send bandwidth in Gbps */
    private static final double LINUX_SEND_BW_GBPS = 55.88;

    /** Linux baseline: RDMA Write latency in microseconds */
    private static final double LINUX_WRITE_LAT_US = 1.00;

    /** Linux baseline: Send latency in microseconds */
    private static final double LINUX_SEND_LAT_US = 0.92;

    // ===== Performance Targets =====

    /** Bandwidth target: achieve at least this % of Linux baseline */
    private static final double BANDWIDTH_TARGET_PERCENT = 95.0;

    /** Latency target: no more than this % above Linux baseline */
    private static final double LATENCY_OVERHEAD_MAX_PERCENT = 20.0;

    // ===== Test Configuration =====

    private static final int BANDWIDTH_MESSAGE_SIZE = 1024 * 1024; // 1MB
    private static final int BANDWIDTH_ITERATIONS = 5000;
    private static final int BANDWIDTH_WARMUP = 100;

    private static final int LATENCY_MESSAGE_SIZE = 8; // 8 bytes
    private static final int LATENCY_ITERATIONS = 10000;
    private static final int LATENCY_WARMUP = 1000;

    private static final int DEFAULT_PORT = 18515;

    // ===== Shared State =====

    private static RdmaApi rdma;
    private static Arena arena;
    private static String serverHost;
    private static int serverPort;
    private static boolean isServer;

    // Results for comparison
    private static double measuredWriteBwGbps;
    private static double measuredReadBwGbps;
    private static double measuredSendBwGbps;
    private static double measuredWriteLatUs;
    private static double measuredSendLatUs;

    @BeforeAll
    static void setUp() {
        serverHost = System.getProperty("rdma.server.host",
                System.getenv().getOrDefault("RDMA_SERVER_HOST", "10.0.0.1"));
        serverPort = Integer.getInteger("rdma.server.port", DEFAULT_PORT);
        isServer = Boolean.getBoolean("rdma.server.mode");

        // Skip if RDMA not available
        assumeTrue(Rdma.canUseRealRdma(),
                "Skipping: Real RDMA hardware not available");

        rdma = Rdma.load();
        arena = Arena.ofShared();

        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║           RDMA PERFORMANCE TEST SUITE                            ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  Backend: %-54s ║%n", rdma.backendName());
        System.out.printf("║  Server:  %-54s ║%n", serverHost + ":" + serverPort);
        System.out.printf("║  Mode:    %-54s ║%n", isServer ? "SERVER" : "CLIENT");
        System.out.println("╠══════════════════════════════════════════════════════════════════╣");
        System.out.println("║  Linux Baseline (PCIe 3.0 x8 limited):                          ║");
        System.out.printf("║    Write BW: %.2f Gbps                                          ║%n", LINUX_WRITE_BW_GBPS);
        System.out.printf("║    Read BW:  %.2f Gbps                                          ║%n", LINUX_READ_BW_GBPS);
        System.out.printf("║    Send BW:  %.2f Gbps                                          ║%n", LINUX_SEND_BW_GBPS);
        System.out.printf("║    Write Lat: %.2f μs                                            ║%n", LINUX_WRITE_LAT_US);
        System.out.printf("║    Send Lat:  %.2f μs                                            ║%n", LINUX_SEND_LAT_US);
        System.out.println("╚══════════════════════════════════════════════════════════════════╝");
    }

    @AfterAll
    static void tearDown() {
        if (arena != null) arena.close();
        if (rdma != null) rdma.close();

        // Print summary
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║           PERFORMANCE SUMMARY                                    ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════╣");
        if (measuredWriteBwGbps > 0) {
            double pct = (measuredWriteBwGbps / LINUX_WRITE_BW_GBPS) * 100;
            System.out.printf("║  Write BW:  %6.2f Gbps (%5.1f%% of Linux baseline)            ║%n",
                    measuredWriteBwGbps, pct);
        }
        if (measuredReadBwGbps > 0) {
            double pct = (measuredReadBwGbps / LINUX_READ_BW_GBPS) * 100;
            System.out.printf("║  Read BW:   %6.2f Gbps (%5.1f%% of Linux baseline)            ║%n",
                    measuredReadBwGbps, pct);
        }
        if (measuredSendBwGbps > 0) {
            double pct = (measuredSendBwGbps / LINUX_SEND_BW_GBPS) * 100;
            System.out.printf("║  Send BW:   %6.2f Gbps (%5.1f%% of Linux baseline)            ║%n",
                    measuredSendBwGbps, pct);
        }
        if (measuredWriteLatUs > 0) {
            double overhead = ((measuredWriteLatUs / LINUX_WRITE_LAT_US) - 1) * 100;
            System.out.printf("║  Write Lat: %6.2f μs   (%+5.1f%% vs Linux baseline)            ║%n",
                    measuredWriteLatUs, overhead);
        }
        if (measuredSendLatUs > 0) {
            double overhead = ((measuredSendLatUs / LINUX_SEND_LAT_US) - 1) * 100;
            System.out.printf("║  Send Lat:  %6.2f μs   (%+5.1f%% vs Linux baseline)            ║%n",
                    measuredSendLatUs, overhead);
        }
        System.out.println("╚══════════════════════════════════════════════════════════════════╝");
    }

    // ===== Bandwidth Tests =====

    @Test
    @Order(1)
    @DisplayName("Send Bandwidth (two-sided)")
    void testSendBandwidth() throws Exception {
        System.out.println("\n─── SEND BANDWIDTH TEST ───");
        System.out.printf("Message size: %d bytes, Iterations: %d%n",
                BANDWIDTH_MESSAGE_SIZE, BANDWIDTH_ITERATIONS);

        MemorySegment segment = arena.allocate(BANDWIDTH_MESSAGE_SIZE);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {

            if (isServer) {
                runBandwidthServer(buffer, BANDWIDTH_ITERATIONS);
            } else {
                // Client sends, measures bandwidth
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    System.out.println("Connected to " + endpoint.remoteAddress());

                    // Fill buffer with test pattern
                    for (int i = 0; i < BANDWIDTH_MESSAGE_SIZE / 8; i++) {
                        segment.setAtIndex(ValueLayout.JAVA_LONG, i, i);
                    }

                    // Warmup
                    System.out.println("Warmup: " + BANDWIDTH_WARMUP + " iterations...");
                    for (int i = 0; i < BANDWIDTH_WARMUP; i++) {
                        endpoint.send(buffer).join();
                    }

                    // Timed run
                    System.out.println("Measuring " + BANDWIDTH_ITERATIONS + " iterations...");
                    long totalBytes = 0;
                    long startNanos = System.nanoTime();

                    for (int i = 0; i < BANDWIDTH_ITERATIONS; i++) {
                        endpoint.send(buffer).join();
                        totalBytes += BANDWIDTH_MESSAGE_SIZE;
                    }

                    long elapsedNanos = System.nanoTime() - startNanos;
                    double seconds = elapsedNanos / 1_000_000_000.0;
                    double gbps = (totalBytes * 8.0) / elapsedNanos;

                    measuredSendBwGbps = gbps;

                    System.out.printf("Result: %.2f Gbps (%.2f GB/s)%n", gbps, totalBytes / seconds / 1e9);
                    System.out.printf("Linux baseline: %.2f Gbps%n", LINUX_SEND_BW_GBPS);

                    double targetGbps = LINUX_SEND_BW_GBPS * BANDWIDTH_TARGET_PERCENT / 100;
                    assertTrue(gbps >= targetGbps,
                            String.format("Send bandwidth %.2f Gbps below target %.2f Gbps (%.0f%% of baseline)",
                                    gbps, targetGbps, BANDWIDTH_TARGET_PERCENT));
                }
            }
        }
    }

    @Test
    @Order(2)
    @DisplayName("RDMA Write Bandwidth (one-sided)")
    void testWriteBandwidth() throws Exception {
        System.out.println("\n─── RDMA WRITE BANDWIDTH TEST ───");
        System.out.printf("Message size: %d bytes, Iterations: %d%n",
                BANDWIDTH_MESSAGE_SIZE, BANDWIDTH_ITERATIONS);

        MemorySegment localSegment = arena.allocate(BANDWIDTH_MESSAGE_SIZE);
        MemorySegment remoteSegment = arena.allocate(BANDWIDTH_MESSAGE_SIZE);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            if (isServer) {
                // Server exposes buffer for remote write
                runOneSidedServer(remoteBuffer);
            } else {
                // Client writes to remote buffer
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    System.out.println("Connected to " + endpoint.remoteAddress());

                    // Exchange remote buffer info (simplified - real impl needs OOB exchange)
                    long remoteAddr = remoteBuffer.address();
                    long remoteKey = remoteBuffer.remoteKey();

                    // Fill local buffer
                    localSegment.fill((byte) 0xAB);

                    // Warmup
                    System.out.println("Warmup: " + BANDWIDTH_WARMUP + " iterations...");
                    for (int i = 0; i < BANDWIDTH_WARMUP; i++) {
                        endpoint.write(localBuffer, remoteAddr, remoteKey).join();
                    }

                    // Timed run
                    System.out.println("Measuring " + BANDWIDTH_ITERATIONS + " iterations...");
                    long totalBytes = 0;
                    long startNanos = System.nanoTime();

                    for (int i = 0; i < BANDWIDTH_ITERATIONS; i++) {
                        endpoint.write(localBuffer, remoteAddr, remoteKey).join();
                        totalBytes += BANDWIDTH_MESSAGE_SIZE;
                    }

                    long elapsedNanos = System.nanoTime() - startNanos;
                    double gbps = (totalBytes * 8.0) / elapsedNanos;

                    measuredWriteBwGbps = gbps;

                    System.out.printf("Result: %.2f Gbps%n", gbps);
                    System.out.printf("Linux baseline: %.2f Gbps%n", LINUX_WRITE_BW_GBPS);

                    double targetGbps = LINUX_WRITE_BW_GBPS * BANDWIDTH_TARGET_PERCENT / 100;
                    assertTrue(gbps >= targetGbps,
                            String.format("Write bandwidth %.2f Gbps below target %.2f Gbps",
                                    gbps, targetGbps));
                }
            }
        }
    }

    @Test
    @Order(3)
    @DisplayName("RDMA Read Bandwidth (one-sided)")
    void testReadBandwidth() throws Exception {
        System.out.println("\n─── RDMA READ BANDWIDTH TEST ───");
        System.out.printf("Message size: %d bytes, Iterations: %d%n",
                BANDWIDTH_MESSAGE_SIZE, BANDWIDTH_ITERATIONS);

        MemorySegment localSegment = arena.allocate(BANDWIDTH_MESSAGE_SIZE);
        MemorySegment remoteSegment = arena.allocate(BANDWIDTH_MESSAGE_SIZE);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            if (isServer) {
                // Server exposes buffer for remote read
                remoteSegment.fill((byte) 0xCD);
                runOneSidedServer(remoteBuffer);
            } else {
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    System.out.println("Connected to " + endpoint.remoteAddress());

                    long remoteAddr = remoteBuffer.address();
                    long remoteKey = remoteBuffer.remoteKey();

                    // Warmup
                    System.out.println("Warmup: " + BANDWIDTH_WARMUP + " iterations...");
                    for (int i = 0; i < BANDWIDTH_WARMUP; i++) {
                        endpoint.read(localBuffer, remoteAddr, remoteKey).join();
                    }

                    // Timed run
                    System.out.println("Measuring " + BANDWIDTH_ITERATIONS + " iterations...");
                    long totalBytes = 0;
                    long startNanos = System.nanoTime();

                    for (int i = 0; i < BANDWIDTH_ITERATIONS; i++) {
                        endpoint.read(localBuffer, remoteAddr, remoteKey).join();
                        totalBytes += BANDWIDTH_MESSAGE_SIZE;
                    }

                    long elapsedNanos = System.nanoTime() - startNanos;
                    double gbps = (totalBytes * 8.0) / elapsedNanos;

                    measuredReadBwGbps = gbps;

                    System.out.printf("Result: %.2f Gbps%n", gbps);
                    System.out.printf("Linux baseline: %.2f Gbps%n", LINUX_READ_BW_GBPS);

                    double targetGbps = LINUX_READ_BW_GBPS * BANDWIDTH_TARGET_PERCENT / 100;
                    assertTrue(gbps >= targetGbps,
                            String.format("Read bandwidth %.2f Gbps below target %.2f Gbps",
                                    gbps, targetGbps));
                }
            }
        }
    }

    // ===== Latency Tests =====

    @Test
    @Order(4)
    @DisplayName("Send Latency (ping-pong)")
    void testSendLatency() throws Exception {
        System.out.println("\n─── SEND LATENCY TEST ───");
        System.out.printf("Message size: %d bytes, Iterations: %d%n",
                LATENCY_MESSAGE_SIZE, LATENCY_ITERATIONS);

        MemorySegment segment = arena.allocate(LATENCY_MESSAGE_SIZE);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {

            if (isServer) {
                // Server echoes back
                try (RdmaListener listener = rdma.listen(serverPort);
                     RdmaEndpoint endpoint = listener.accept(30_000)) {
                    System.out.println("Client connected: " + endpoint.remoteAddress());

                    // Echo loop
                    int total = LATENCY_WARMUP + LATENCY_ITERATIONS;
                    for (int i = 0; i < total; i++) {
                        endpoint.receive(buffer).join();
                        endpoint.send(buffer).join();
                    }
                }
            } else {
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    System.out.println("Connected to " + endpoint.remoteAddress());

                    // Warmup
                    System.out.println("Warmup: " + LATENCY_WARMUP + " iterations...");
                    for (int i = 0; i < LATENCY_WARMUP; i++) {
                        endpoint.send(buffer).join();
                        endpoint.receive(buffer).join();
                    }

                    // Timed run - measure RTT for each iteration
                    System.out.println("Measuring " + LATENCY_ITERATIONS + " iterations...");
                    long[] latencies = new long[LATENCY_ITERATIONS];

                    for (int i = 0; i < LATENCY_ITERATIONS; i++) {
                        long start = System.nanoTime();
                        endpoint.send(buffer).join();
                        endpoint.receive(buffer).join();
                        latencies[i] = System.nanoTime() - start;
                    }

                    // Calculate statistics
                    LatencyStats stats = calculateLatencyStats(latencies);
                    measuredSendLatUs = stats.avg / 1000.0; // RTT/2 for one-way

                    System.out.printf("RTT Latency:%n");
                    System.out.printf("  Min:     %.2f μs%n", stats.min / 1000.0);
                    System.out.printf("  Max:     %.2f μs%n", stats.max / 1000.0);
                    System.out.printf("  Avg:     %.2f μs%n", stats.avg / 1000.0);
                    System.out.printf("  Median:  %.2f μs%n", stats.median / 1000.0);
                    System.out.printf("  P99:     %.2f μs%n", stats.p99 / 1000.0);
                    System.out.printf("  P99.9:   %.2f μs%n", stats.p999 / 1000.0);
                    System.out.printf("Linux baseline (one-way): %.2f μs%n", LINUX_SEND_LAT_US);

                    // Compare one-way (RTT/2) to baseline
                    double oneWayUs = stats.avg / 2000.0;
                    double maxLatencyUs = LINUX_SEND_LAT_US * (1 + LATENCY_OVERHEAD_MAX_PERCENT / 100);
                    assertTrue(oneWayUs <= maxLatencyUs,
                            String.format("Send latency %.2f μs exceeds target %.2f μs",
                                    oneWayUs, maxLatencyUs));
                }
            }
        }
    }

    @Test
    @Order(5)
    @DisplayName("RDMA Write Latency")
    void testWriteLatency() throws Exception {
        System.out.println("\n─── RDMA WRITE LATENCY TEST ───");
        System.out.printf("Message size: %d bytes, Iterations: %d%n",
                LATENCY_MESSAGE_SIZE, LATENCY_ITERATIONS);

        MemorySegment localSegment = arena.allocate(LATENCY_MESSAGE_SIZE);
        MemorySegment remoteSegment = arena.allocate(LATENCY_MESSAGE_SIZE);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            if (isServer) {
                runOneSidedServer(remoteBuffer);
            } else {
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    System.out.println("Connected to " + endpoint.remoteAddress());

                    long remoteAddr = remoteBuffer.address();
                    long remoteKey = remoteBuffer.remoteKey();

                    // Warmup
                    System.out.println("Warmup: " + LATENCY_WARMUP + " iterations...");
                    for (int i = 0; i < LATENCY_WARMUP; i++) {
                        endpoint.write(localBuffer, remoteAddr, remoteKey).join();
                    }

                    // Timed run
                    System.out.println("Measuring " + LATENCY_ITERATIONS + " iterations...");
                    long[] latencies = new long[LATENCY_ITERATIONS];

                    for (int i = 0; i < LATENCY_ITERATIONS; i++) {
                        long start = System.nanoTime();
                        endpoint.write(localBuffer, remoteAddr, remoteKey).join();
                        latencies[i] = System.nanoTime() - start;
                    }

                    LatencyStats stats = calculateLatencyStats(latencies);
                    measuredWriteLatUs = stats.avg / 1000.0;

                    System.out.printf("Write Latency:%n");
                    System.out.printf("  Min:     %.2f μs%n", stats.min / 1000.0);
                    System.out.printf("  Max:     %.2f μs%n", stats.max / 1000.0);
                    System.out.printf("  Avg:     %.2f μs%n", stats.avg / 1000.0);
                    System.out.printf("  Median:  %.2f μs%n", stats.median / 1000.0);
                    System.out.printf("  P99:     %.2f μs%n", stats.p99 / 1000.0);
                    System.out.printf("  P99.9:   %.2f μs%n", stats.p999 / 1000.0);
                    System.out.printf("Linux baseline: %.2f μs%n", LINUX_WRITE_LAT_US);

                    double maxLatencyUs = LINUX_WRITE_LAT_US * (1 + LATENCY_OVERHEAD_MAX_PERCENT / 100);
                    assertTrue(stats.avg / 1000.0 <= maxLatencyUs,
                            String.format("Write latency %.2f μs exceeds target %.2f μs",
                                    stats.avg / 1000.0, maxLatencyUs));
                }
            }
        }
    }

    // ===== Helper Methods =====

    private void runBandwidthServer(RdmaBuffer buffer, int iterations) throws Exception {
        try (RdmaListener listener = rdma.listen(serverPort);
             RdmaEndpoint endpoint = listener.accept(30_000)) {
            System.out.println("Client connected: " + endpoint.remoteAddress());

            int total = BANDWIDTH_WARMUP + iterations;
            for (int i = 0; i < total; i++) {
                endpoint.receive(buffer).join();
                if (i % 1000 == 0) {
                    System.out.printf("Received %d/%d%n", i, total);
                }
            }
        }
    }

    private void runOneSidedServer(RdmaBuffer buffer) throws Exception {
        // One-sided operations don't need server-side handling,
        // but we need to keep the connection alive
        System.out.println("Buffer exposed for one-sided operations:");
        System.out.printf("  Address: 0x%x%n", buffer.address());
        System.out.printf("  RKey:    0x%x%n", buffer.remoteKey());

        // Wait for client to finish (simplified - real impl uses signaling)
        Thread.sleep(60_000);
    }

    private LatencyStats calculateLatencyStats(long[] latencies) {
        long[] sorted = latencies.clone();
        Arrays.sort(sorted);

        long sum = 0;
        for (long l : sorted) sum += l;

        int n = sorted.length;
        return new LatencyStats(
                sorted[0],                          // min
                sorted[n - 1],                      // max
                sum / (double) n,                   // avg
                sorted[n / 2],                      // median
                sorted[(int) (n * 0.99)],          // p99
                sorted[(int) (n * 0.999)]          // p99.9
        );
    }

    private record LatencyStats(
            long min,
            long max,
            double avg,
            long median,
            long p99,
            long p999
    ) {}
}
