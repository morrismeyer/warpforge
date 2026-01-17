package io.surfworks.warpforge.io.perf;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.io.buffer.RegisteredBuffer;
import io.surfworks.warpforge.io.collective.AllReduceOp;
import io.surfworks.warpforge.io.integration.RayIntegrationTestBase;
import io.surfworks.warpforge.io.rdma.RdmaBuffer;
import io.surfworks.warpforge.io.rdma.RdmaEndpoint;
import io.surfworks.warpforge.io.rdma.RdmaListener;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Java RDMA throughput tests targeting 95-98% of ibperf baseline.
 *
 * <p>These tests measure the actual throughput achieved by the Java
 * FFM-based RDMA implementation and compare it against the ibperf
 * baseline established in {@link IbPerfBaselineTest}.
 *
 * <h2>Target Performance</h2>
 * <ul>
 *   <li>RDMA Write: >= 95% of ib_write_bw baseline</li>
 *   <li>RDMA Read: >= 95% of ib_read_bw baseline</li>
 *   <li>AllReduce: >= 80% of theoretical ring bandwidth</li>
 * </ul>
 *
 * <h2>Running</h2>
 * <pre>{@code
 * # First run baseline tests
 * ./gradlew :warpforge-io:rdmaPerfTest --tests "*IbPerfBaselineTest*"
 *
 * # Then run Java throughput tests
 * ./gradlew :warpforge-io:rdmaPerfTest --tests "*JavaThroughputTest*"
 * }</pre>
 */
@Tag("rdma-perf")
@Tag("rdma")
@DisplayName("Java RDMA Throughput Tests")
class JavaThroughputTest extends RayIntegrationTestBase {

    private Arena arena;
    private RdmaEndpoint endpoint;

    // Performance targets (percentage of ibperf baseline)
    private static final double TARGET_WRITE_EFFICIENCY = 0.95; // 95%
    private static final double TARGET_READ_EFFICIENCY = 0.95;  // 95%
    private static final double TARGET_COLLECTIVE_EFFICIENCY = 0.80; // 80%

    // Test parameters
    private static final int MESSAGE_SIZE = 1024 * 1024; // 1MB
    private static final int WARMUP_ITERATIONS = 100;
    private static final int MEASURE_ITERATIONS = 1000;

    private static final int TEST_PORT = 18516;

    @BeforeEach
    void setUpTest() {
        arena = Arena.ofConfined();

        // Establish connection between ranks
        if (rank == 0) {
            try (RdmaListener listener = rdma.listen(TEST_PORT)) {
                sync();
                endpoint = listener.accept(30000);
            }
        } else {
            sync();
            endpoint = rdma.connect(masterAddress, TEST_PORT, 30000);
        }

        sync();
    }

    @AfterEach
    void tearDownTest() {
        if (endpoint != null) endpoint.close();
        if (arena != null) arena.close();
    }

    // ===== RDMA Write Throughput =====

    @Test
    @DisplayName("Java RDMA Write Throughput (target: 95% of ibperf)")
    void measureWriteThroughput() throws Exception {
        MemorySegment localSegment = arena.allocate(MESSAGE_SIZE);
        MemorySegment remoteSegment = arena.allocate(MESSAGE_SIZE);

        // Fill local buffer
        localSegment.fill((byte) 0xAB);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            long remoteAddr = remoteBuffer.address();
            long remoteKey = remoteBuffer.remoteKey();

            sync();

            if (rank == 0) {
                // Warmup
                for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                    endpoint.write(localBuffer, remoteAddr, remoteKey).get(10, TimeUnit.SECONDS);
                }
                endpoint.flush();

                // Measure
                long startNanos = System.nanoTime();
                for (int i = 0; i < MEASURE_ITERATIONS; i++) {
                    endpoint.write(localBuffer, remoteAddr, remoteKey).get(10, TimeUnit.SECONDS);
                }
                endpoint.flush();
                long elapsedNanos = System.nanoTime() - startNanos;

                // Calculate throughput
                double totalBytes = (double) MESSAGE_SIZE * MEASURE_ITERATIONS;
                double seconds = elapsedNanos / 1e9;
                double gbps = (totalBytes * 8) / elapsedNanos;
                double mbps = totalBytes / seconds / (1024 * 1024);

                log("Java RDMA Write: %.2f Gbps (%.2f MB/s)", gbps, mbps);

                // Compare to baseline if available
                if (IbPerfBaselineTest.baselineWriteBandwidthGbps > 0) {
                    double efficiency = gbps / IbPerfBaselineTest.baselineWriteBandwidthGbps;
                    log("Efficiency vs ibperf: %.1f%% (target: %.0f%%)",
                            efficiency * 100, TARGET_WRITE_EFFICIENCY * 100);

                    assertTrue(efficiency >= TARGET_WRITE_EFFICIENCY,
                            String.format("Write throughput %.2f Gbps is only %.1f%% of baseline %.2f Gbps " +
                                            "(target: %.0f%%)",
                                    gbps, efficiency * 100, IbPerfBaselineTest.baselineWriteBandwidthGbps,
                                    TARGET_WRITE_EFFICIENCY * 100));
                } else {
                    // No baseline, just verify minimum performance
                    assertTrue(gbps >= 90.0,
                            "Write throughput should be at least 90 Gbps on 100GbE, got " + gbps);
                }
            }

            sync();
        }
    }

    @Test
    @DisplayName("Java RDMA Read Throughput (target: 95% of ibperf)")
    void measureReadThroughput() throws Exception {
        MemorySegment localSegment = arena.allocate(MESSAGE_SIZE);
        MemorySegment remoteSegment = arena.allocate(MESSAGE_SIZE);

        // Remote fills buffer
        if (rank == 1) {
            remoteSegment.fill((byte) 0xCD);
        }

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            long remoteAddr = remoteBuffer.address();
            long remoteKey = remoteBuffer.remoteKey();

            sync();

            if (rank == 0) {
                // Warmup
                for (int i = 0; i < WARMUP_ITERATIONS; i++) {
                    endpoint.read(localBuffer, remoteAddr, remoteKey).get(10, TimeUnit.SECONDS);
                }
                endpoint.flush();

                // Measure
                long startNanos = System.nanoTime();
                for (int i = 0; i < MEASURE_ITERATIONS; i++) {
                    endpoint.read(localBuffer, remoteAddr, remoteKey).get(10, TimeUnit.SECONDS);
                }
                endpoint.flush();
                long elapsedNanos = System.nanoTime() - startNanos;

                double totalBytes = (double) MESSAGE_SIZE * MEASURE_ITERATIONS;
                double gbps = (totalBytes * 8) / elapsedNanos;
                double mbps = totalBytes / (elapsedNanos / 1e9) / (1024 * 1024);

                log("Java RDMA Read: %.2f Gbps (%.2f MB/s)", gbps, mbps);

                if (IbPerfBaselineTest.baselineReadBandwidthGbps > 0) {
                    double efficiency = gbps / IbPerfBaselineTest.baselineReadBandwidthGbps;
                    log("Efficiency vs ibperf: %.1f%% (target: %.0f%%)",
                            efficiency * 100, TARGET_READ_EFFICIENCY * 100);

                    assertTrue(efficiency >= TARGET_READ_EFFICIENCY,
                            String.format("Read throughput %.2f Gbps is only %.1f%% of baseline",
                                    gbps, efficiency * 100));
                } else {
                    assertTrue(gbps >= 90.0,
                            "Read throughput should be at least 90 Gbps, got " + gbps);
                }
            }

            sync();
        }
    }

    // ===== RDMA Send/Receive Throughput =====

    @Test
    @DisplayName("Java RDMA Send/Receive Throughput")
    void measureSendRecvThroughput() throws Exception {
        MemorySegment segment = arena.allocate(MESSAGE_SIZE);

        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            sync();

            // Ping-pong: rank 0 sends, rank 1 receives, then vice versa
            long startNanos = System.nanoTime();
            int totalIterations = MEASURE_ITERATIONS;

            for (int i = 0; i < totalIterations; i++) {
                if (rank == 0) {
                    endpoint.send(buffer).get(10, TimeUnit.SECONDS);
                    endpoint.receive(buffer).get(10, TimeUnit.SECONDS);
                } else {
                    endpoint.receive(buffer).get(10, TimeUnit.SECONDS);
                    endpoint.send(buffer).get(10, TimeUnit.SECONDS);
                }
            }

            long elapsedNanos = System.nanoTime() - startNanos;

            double totalBytes = (double) MESSAGE_SIZE * totalIterations * 2; // Both directions
            double gbps = (totalBytes * 8) / elapsedNanos;

            log("Java Send/Recv: %.2f Gbps (ping-pong)", gbps);

            sync();
        }
    }

    // ===== Write Latency =====

    @Test
    @DisplayName("Java RDMA Write Latency")
    void measureWriteLatency() throws Exception {
        final int LATENCY_SIZE = 8;
        final int LATENCY_ITERATIONS = 10000;

        MemorySegment localSegment = arena.allocate(LATENCY_SIZE);
        MemorySegment remoteSegment = arena.allocate(LATENCY_SIZE);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            long remoteAddr = remoteBuffer.address();
            long remoteKey = remoteBuffer.remoteKey();

            sync();

            if (rank == 0) {
                // Warmup
                for (int i = 0; i < 1000; i++) {
                    endpoint.write(localBuffer, remoteAddr, remoteKey).get(5, TimeUnit.SECONDS);
                }

                // Measure individual latencies
                long[] latencies = new long[LATENCY_ITERATIONS];
                for (int i = 0; i < LATENCY_ITERATIONS; i++) {
                    long start = System.nanoTime();
                    endpoint.write(localBuffer, remoteAddr, remoteKey).get(5, TimeUnit.SECONDS);
                    latencies[i] = System.nanoTime() - start;
                }

                // Calculate statistics
                java.util.Arrays.sort(latencies);
                double avgNs = java.util.Arrays.stream(latencies).average().orElse(0);
                double p50Ns = latencies[LATENCY_ITERATIONS / 2];
                double p99Ns = latencies[(int) (LATENCY_ITERATIONS * 0.99)];

                double avgUs = avgNs / 1000;
                double p50Us = p50Ns / 1000;
                double p99Us = p99Ns / 1000;

                log("Java Write Latency: avg=%.2f us, p50=%.2f us, p99=%.2f us",
                        avgUs, p50Us, p99Us);

                if (IbPerfBaselineTest.baselineWriteLatencyUs > 0) {
                    double overhead = avgUs - IbPerfBaselineTest.baselineWriteLatencyUs;
                    log("Latency overhead vs ibperf: %.2f us", overhead);

                    // Java overhead should be < 2us
                    assertTrue(overhead < 2.0,
                            String.format("Latency overhead %.2f us exceeds 2us limit", overhead));
                }

                // Absolute limit: < 5us for 100GbE
                assertTrue(p50Us < 5.0,
                        "P50 latency should be < 5us, got " + p50Us);
            }

            sync();
        }
    }

    // ===== Collective Throughput =====

    @Test
    @DisplayName("AllReduce Throughput (target: 80% theoretical)")
    void measureAllReduceThroughput() throws Exception {
        final int TENSOR_SIZE = 64 * 1024 * 1024; // 64M floats = 256MB
        final int ALLREDUCE_ITERATIONS = 50;

        try (Tensor input = Tensor.zeros(ScalarType.F32, TENSOR_SIZE)) {
            // Fill with test data
            float[] data = new float[TENSOR_SIZE];
            java.util.Arrays.fill(data, 1.0f);
            input.copyFrom(data);

            // Warmup
            for (int i = 0; i < 5; i++) {
                collective.allReduce(input, AllReduceOp.SUM).get().close();
            }

            // Measure
            long startNanos = System.nanoTime();
            for (int i = 0; i < ALLREDUCE_ITERATIONS; i++) {
                Tensor result = collective.allReduce(input, AllReduceOp.SUM)
                        .get(60, TimeUnit.SECONDS);
                result.close();
            }
            long elapsedNanos = System.nanoTime() - startNanos;

            // Calculate throughput
            // For ring allreduce: data moves 2x (once for reduce, once for broadcast)
            double totalBytes = (double) TENSOR_SIZE * 4 * ALLREDUCE_ITERATIONS * 2;
            double gbps = (totalBytes * 8) / elapsedNanos;
            double effectiveGbps = gbps / 2; // Effective throughput accounting for algorithm

            log("AllReduce: %.2f Gbps effective (%.2f Gbps raw)",
                    effectiveGbps, gbps);

            // For 100GbE ring with 2 nodes, theoretical max is ~100 Gbps
            // We target 80% efficiency
            double theoreticalMax = 100.0;
            double efficiency = effectiveGbps / theoreticalMax;

            log("Efficiency: %.1f%% (target: %.0f%%)",
                    efficiency * 100, TARGET_COLLECTIVE_EFFICIENCY * 100);

            assertTrue(efficiency >= TARGET_COLLECTIVE_EFFICIENCY,
                    String.format("AllReduce efficiency %.1f%% below target %.0f%%",
                            efficiency * 100, TARGET_COLLECTIVE_EFFICIENCY * 100));
        }
    }

    // ===== Zero-Copy Verification =====

    @Test
    @DisplayName("Verify zero-copy path with RegisteredBuffer")
    void verifyZeroCopy() throws Exception {
        // Use RegisteredBuffer to verify zero-copy integration with Tensor

        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(
                rdma, ScalarType.F32, MESSAGE_SIZE / 4)) {

            // Fill via tensor interface
            Tensor tensor = buffer.tensor();
            float[] data = new float[MESSAGE_SIZE / 4];
            java.util.Arrays.fill(data, 42.0f);
            tensor.copyFrom(data);

            // Verify same memory is used
            assertEquals(tensor.data().address(), buffer.segment().address(),
                    "Tensor and RegisteredBuffer should share memory");

            log("RegisteredBuffer zero-copy verified: same address 0x%x",
                    buffer.address());

            sync();

            // Use for RDMA operation
            if (rank == 0) {
                RdmaBuffer rdmaBuffer = buffer.rdmaBuffer();
                endpoint.send(rdmaBuffer).get(10, TimeUnit.SECONDS);
                log("Zero-copy send completed");
            }

            sync();
        }
    }

    // ===== Sustained Throughput =====

    @Test
    @DisplayName("Sustained Write Throughput (60 seconds)")
    void measureSustainedThroughput() throws Exception {
        final long DURATION_SECONDS = 60;

        MemorySegment localSegment = arena.allocate(MESSAGE_SIZE);
        MemorySegment remoteSegment = arena.allocate(MESSAGE_SIZE);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            long remoteAddr = remoteBuffer.address();
            long remoteKey = remoteBuffer.remoteKey();

            sync();

            if (rank == 0) {
                long startNanos = System.nanoTime();
                long endNanos = startNanos + DURATION_SECONDS * 1_000_000_000L;
                long iterations = 0;
                long totalBytes = 0;

                while (System.nanoTime() < endNanos) {
                    endpoint.write(localBuffer, remoteAddr, remoteKey).get(10, TimeUnit.SECONDS);
                    iterations++;
                    totalBytes += MESSAGE_SIZE;
                }

                long elapsedNanos = System.nanoTime() - startNanos;
                double seconds = elapsedNanos / 1e9;
                double gbps = (totalBytes * 8.0) / elapsedNanos;

                log("Sustained throughput over %.0f seconds: %.2f Gbps (%d iterations)",
                        seconds, gbps, iterations);

                // Sustained throughput should maintain at least 90% of peak
                assertTrue(gbps >= 85.0,
                        "Sustained throughput should be >= 85 Gbps, got " + gbps);
            }

            sync();
        }
    }
}
