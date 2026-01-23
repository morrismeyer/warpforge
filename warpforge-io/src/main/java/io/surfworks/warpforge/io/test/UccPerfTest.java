package io.surfworks.warpforge.io.test;

import java.util.Arrays;
import java.util.Locale;

import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.io.collective.AllReduceOp;
import io.surfworks.warpforge.io.collective.Collective;
import io.surfworks.warpforge.io.collective.CollectiveApi;
import io.surfworks.warpforge.io.collective.CollectiveConfig;

/**
 * Comprehensive UCC collective operations performance test.
 *
 * <p>This test measures throughput and latency for all collective operations
 * across various message sizes. It runs over the Mellanox 100GbE cross-connect
 * between NVIDIA and AMD GPU boxes.
 *
 * <h2>Usage</h2>
 * <pre>
 * # On mark1nvidia (10.0.0.1) - run as rank 0 (master):
 * ./gradlew :warpforge-io:uccPerfTest --args='--rank 0 --world-size 2 --master 10.0.0.1 --port 29500 --size 1048576 --iterations 100'
 *
 * # On mark1amd (10.0.0.2) - run as rank 1 (worker):
 * ./gradlew :warpforge-io:uccPerfTest --args='--rank 1 --world-size 2 --master 10.0.0.1 --port 29500 --size 1048576 --iterations 100'
 * </pre>
 *
 * <h2>Metrics Collected</h2>
 * <ul>
 *   <li>Throughput (Gbps) - calculated from message size and elapsed time</li>
 *   <li>Latency (ms) - average time per operation</li>
 *   <li>Operations per second</li>
 * </ul>
 */
public class UccPerfTest {

    private static final long BYTES_PER_FLOAT = 4;
    private static final double BYTES_TO_GBITS = 8.0 / 1_000_000_000.0;

    public static void main(String[] args) {
        // Default configuration
        int rank = 0;
        int worldSize = 2;
        String masterAddr = "10.0.0.1";
        int masterPort = 29500;
        long sizeBytes = 1024 * 1024; // 1MB default
        int iterations = 100;
        int warmup = 10;
        boolean allOps = true;
        String specificOp = null;

        // Parse arguments
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--rank", "-r" -> rank = Integer.parseInt(args[++i]);
                case "--world-size", "-w" -> worldSize = Integer.parseInt(args[++i]);
                case "--master", "-m" -> masterAddr = args[++i];
                case "--port", "-p" -> masterPort = Integer.parseInt(args[++i]);
                case "--size", "-s" -> sizeBytes = Long.parseLong(args[++i]);
                case "--iterations", "-i" -> iterations = Integer.parseInt(args[++i]);
                case "--warmup" -> warmup = Integer.parseInt(args[++i]);
                case "--op" -> {
                    allOps = false;
                    specificOp = args[++i];
                }
                case "--help", "-h" -> {
                    printUsage();
                    return;
                }
            }
        }

        // Calculate element count (assuming float32)
        long elementCount = sizeBytes / BYTES_PER_FLOAT;

        System.out.println("=".repeat(70));
        System.out.println("UCC Collective Performance Test");
        System.out.println("=".repeat(70));
        System.out.printf("Rank: %d / %d%n", rank, worldSize);
        System.out.printf("Master: %s:%d%n", masterAddr, masterPort);
        System.out.printf("Message size: %s (%d floats)%n", formatBytes(sizeBytes), elementCount);
        System.out.printf("Iterations: %d (warmup: %d)%n", iterations, warmup);
        System.out.println(Collective.systemInfo());
        System.out.println();

        // Create configuration
        CollectiveConfig config = CollectiveConfig.builder(worldSize, rank)
                .masterAddress(masterAddr)
                .masterPort(masterPort)
                .build();

        // Run benchmarks
        try (CollectiveApi collective = Collective.loadUcc(config)) {
            System.out.printf("[Rank %d] UCC initialized, backend: %s%n", rank, collective.backendName());
            System.out.println();

            // Initial barrier to sync all ranks
            collective.barrier().join();

            if (allOps) {
                // Run all operations
                benchmarkAllReduce(collective, rank, worldSize, elementCount, iterations, warmup);
                benchmarkAllGather(collective, rank, worldSize, elementCount, iterations, warmup);
                benchmarkBroadcast(collective, rank, worldSize, elementCount, iterations, warmup);
                benchmarkReduceScatter(collective, rank, worldSize, elementCount, iterations, warmup);
                benchmarkAllToAll(collective, rank, worldSize, elementCount, iterations, warmup);
                benchmarkBarrier(collective, rank, iterations, warmup);
            } else {
                // Run specific operation
                switch (specificOp.toLowerCase()) {
                    case "allreduce" -> benchmarkAllReduce(collective, rank, worldSize, elementCount, iterations, warmup);
                    case "allgather" -> benchmarkAllGather(collective, rank, worldSize, elementCount, iterations, warmup);
                    case "broadcast" -> benchmarkBroadcast(collective, rank, worldSize, elementCount, iterations, warmup);
                    case "reducescatter" -> benchmarkReduceScatter(collective, rank, worldSize, elementCount, iterations, warmup);
                    case "alltoall" -> benchmarkAllToAll(collective, rank, worldSize, elementCount, iterations, warmup);
                    case "barrier" -> benchmarkBarrier(collective, rank, iterations, warmup);
                    default -> {
                        System.err.println("Unknown operation: " + specificOp);
                        System.exit(1);
                    }
                }
            }

            // Final barrier
            collective.barrier().join();

            System.out.println();
            System.out.println("=".repeat(70));
            System.out.printf("[Rank %d] All benchmarks completed%n", rank);
            System.out.println("Stats: " + collective.stats());
            System.out.println("=".repeat(70));

        } catch (Exception e) {
            System.err.printf("[Rank %d] FAILED: %s%n", rank, e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    // ========================================================================
    // Benchmark Methods
    // ========================================================================

    private static void benchmarkAllReduce(CollectiveApi collective, int rank, int worldSize,
                                           long elementCount, int iterations, int warmup) throws Exception {
        System.out.println("-".repeat(70));
        System.out.printf("[Rank %d] Benchmark: AllReduce SUM%n", rank);
        System.out.println("-".repeat(70));

        // Allocate tensors
        float[] data = new float[(int) elementCount];
        Arrays.fill(data, rank + 1.0f);
        Tensor input = Tensor.fromFloatArray(data, (int) elementCount);
        long sizeBytes = elementCount * BYTES_PER_FLOAT;

        // Warmup
        System.out.printf("[Rank %d] Warmup (%d iterations)...%n", rank, warmup);
        for (int i = 0; i < warmup; i++) {
            Tensor result = collective.allReduce(input, AllReduceOp.SUM).join();
            result.close();
        }

        // Barrier before timed runs
        collective.barrier().join();

        // Timed iterations
        System.out.printf("[Rank %d] Running %d iterations...%n", rank, iterations);
        long[] latencies = new long[iterations];
        long totalStart = System.nanoTime();

        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            Tensor result = collective.allReduce(input, AllReduceOp.SUM).join();
            latencies[i] = System.nanoTime() - start;
            result.close();
        }

        long totalElapsed = System.nanoTime() - totalStart;

        // Verify correctness
        Tensor verifyResult = collective.allReduce(input, AllReduceOp.SUM).join();
        float expected = worldSize * (worldSize + 1) / 2.0f;
        float actual = verifyResult.getFloatFlat(0);
        verifyResult.close();

        boolean correct = Math.abs(actual - expected) < 0.001f;

        // Report results
        printResults("AllReduce", rank, sizeBytes, iterations, totalElapsed, latencies, correct);

        input.close();
    }

    private static void benchmarkAllGather(CollectiveApi collective, int rank, int worldSize,
                                           long elementCount, int iterations, int warmup) throws Exception {
        System.out.println("-".repeat(70));
        System.out.printf("[Rank %d] Benchmark: AllGather%n", rank);
        System.out.println("-".repeat(70));

        // Each rank contributes elementCount floats
        float[] data = new float[(int) elementCount];
        for (int i = 0; i < elementCount; i++) {
            data[i] = rank * 1000 + i;
        }
        Tensor input = Tensor.fromFloatArray(data, (int) elementCount);
        long sizeBytes = elementCount * BYTES_PER_FLOAT;
        long totalSizeBytes = sizeBytes * worldSize;

        // Warmup
        System.out.printf("[Rank %d] Warmup (%d iterations)...%n", rank, warmup);
        for (int i = 0; i < warmup; i++) {
            Tensor result = collective.allGather(input).join();
            result.close();
        }

        // Barrier before timed runs
        collective.barrier().join();

        // Timed iterations
        System.out.printf("[Rank %d] Running %d iterations...%n", rank, iterations);
        long[] latencies = new long[iterations];
        long totalStart = System.nanoTime();

        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            Tensor result = collective.allGather(input).join();
            latencies[i] = System.nanoTime() - start;
            result.close();
        }

        long totalElapsed = System.nanoTime() - totalStart;

        // Verify correctness
        Tensor verifyResult = collective.allGather(input).join();
        boolean correct = verifyResult.numElements() == elementCount * worldSize;
        if (correct) {
            for (int r = 0; r < worldSize; r++) {
                float expected = r * 1000;
                float actual = verifyResult.getFloatFlat((int) (r * elementCount));
                if (Math.abs(actual - expected) > 0.001f) {
                    correct = false;
                    break;
                }
            }
        }
        verifyResult.close();

        // Report results (use total size for bandwidth calculation)
        printResults("AllGather", rank, totalSizeBytes, iterations, totalElapsed, latencies, correct);

        input.close();
    }

    private static void benchmarkBroadcast(CollectiveApi collective, int rank, int worldSize,
                                           long elementCount, int iterations, int warmup) throws Exception {
        System.out.println("-".repeat(70));
        System.out.printf("[Rank %d] Benchmark: Broadcast from rank 0%n", rank);
        System.out.println("-".repeat(70));

        float[] data = new float[(int) elementCount];
        if (rank == 0) {
            for (int i = 0; i < elementCount; i++) {
                data[i] = 42.0f + i;
            }
        }
        Tensor input = Tensor.fromFloatArray(data, (int) elementCount);
        long sizeBytes = elementCount * BYTES_PER_FLOAT;

        // Warmup
        System.out.printf("[Rank %d] Warmup (%d iterations)...%n", rank, warmup);
        for (int i = 0; i < warmup; i++) {
            Tensor result = collective.broadcast(input, 0).join();
            result.close();
        }

        // Barrier before timed runs
        collective.barrier().join();

        // Timed iterations
        System.out.printf("[Rank %d] Running %d iterations...%n", rank, iterations);
        long[] latencies = new long[iterations];
        long totalStart = System.nanoTime();

        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            Tensor result = collective.broadcast(input, 0).join();
            latencies[i] = System.nanoTime() - start;
            result.close();
        }

        long totalElapsed = System.nanoTime() - totalStart;

        // Verify correctness
        Tensor verifyResult = collective.broadcast(input, 0).join();
        boolean correct = Math.abs(verifyResult.getFloatFlat(0) - 42.0f) < 0.001f;
        verifyResult.close();

        printResults("Broadcast", rank, sizeBytes, iterations, totalElapsed, latencies, correct);

        input.close();
    }

    private static void benchmarkReduceScatter(CollectiveApi collective, int rank, int worldSize,
                                               long elementCount, int iterations, int warmup) throws Exception {
        System.out.println("-".repeat(70));
        System.out.printf("[Rank %d] Benchmark: ReduceScatter SUM%n", rank);
        System.out.println("-".repeat(70));

        // Each rank provides worldSize * chunk elements
        long totalElements = elementCount * worldSize;
        float[] data = new float[(int) totalElements];
        Arrays.fill(data, 1.0f);
        Tensor input = Tensor.fromFloatArray(data, (int) totalElements);
        long sizeBytes = totalElements * BYTES_PER_FLOAT;

        // Warmup
        System.out.printf("[Rank %d] Warmup (%d iterations)...%n", rank, warmup);
        for (int i = 0; i < warmup; i++) {
            Tensor result = collective.reduceScatter(input, AllReduceOp.SUM).join();
            result.close();
        }

        // Barrier before timed runs
        collective.barrier().join();

        // Timed iterations
        System.out.printf("[Rank %d] Running %d iterations...%n", rank, iterations);
        long[] latencies = new long[iterations];
        long totalStart = System.nanoTime();

        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            Tensor result = collective.reduceScatter(input, AllReduceOp.SUM).join();
            latencies[i] = System.nanoTime() - start;
            result.close();
        }

        long totalElapsed = System.nanoTime() - totalStart;

        // Verify correctness
        Tensor verifyResult = collective.reduceScatter(input, AllReduceOp.SUM).join();
        float expected = worldSize;
        float actual = verifyResult.getFloatFlat(0);
        boolean correct = Math.abs(actual - expected) < 0.001f;
        verifyResult.close();

        printResults("ReduceScatter", rank, sizeBytes, iterations, totalElapsed, latencies, correct);

        input.close();
    }

    private static void benchmarkAllToAll(CollectiveApi collective, int rank, int worldSize,
                                          long elementCount, int iterations, int warmup) throws Exception {
        System.out.println("-".repeat(70));
        System.out.printf("[Rank %d] Benchmark: AllToAll%n", rank);
        System.out.println("-".repeat(70));

        // Each rank sends/receives elementCount elements to/from each other rank
        long totalElements = elementCount * worldSize;
        float[] data = new float[(int) totalElements];
        for (int i = 0; i < totalElements; i++) {
            data[i] = rank * 10000 + i;
        }
        Tensor input = Tensor.fromFloatArray(data, (int) totalElements);
        long sizeBytes = totalElements * BYTES_PER_FLOAT;

        // Warmup
        System.out.printf("[Rank %d] Warmup (%d iterations)...%n", rank, warmup);
        for (int i = 0; i < warmup; i++) {
            Tensor result = collective.allToAll(input).join();
            result.close();
        }

        // Barrier before timed runs
        collective.barrier().join();

        // Timed iterations
        System.out.printf("[Rank %d] Running %d iterations...%n", rank, iterations);
        long[] latencies = new long[iterations];
        long totalStart = System.nanoTime();

        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            Tensor result = collective.allToAll(input).join();
            latencies[i] = System.nanoTime() - start;
            result.close();
        }

        long totalElapsed = System.nanoTime() - totalStart;

        // AllToAll correctness is complex to verify, just check output size
        Tensor verifyResult = collective.allToAll(input).join();
        boolean correct = verifyResult.numElements() == totalElements;
        verifyResult.close();

        printResults("AllToAll", rank, sizeBytes, iterations, totalElapsed, latencies, correct);

        input.close();
    }

    private static void benchmarkBarrier(CollectiveApi collective, int rank,
                                         int iterations, int warmup) throws Exception {
        System.out.println("-".repeat(70));
        System.out.printf("[Rank %d] Benchmark: Barrier%n", rank);
        System.out.println("-".repeat(70));

        // Warmup
        System.out.printf("[Rank %d] Warmup (%d iterations)...%n", rank, warmup);
        for (int i = 0; i < warmup; i++) {
            collective.barrier().join();
        }

        // Barrier before timed runs
        collective.barrier().join();

        // Timed iterations
        System.out.printf("[Rank %d] Running %d iterations...%n", rank, iterations);
        long[] latencies = new long[iterations];
        long totalStart = System.nanoTime();

        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            collective.barrier().join();
            latencies[i] = System.nanoTime() - start;
        }

        long totalElapsed = System.nanoTime() - totalStart;

        // Report results (no bandwidth for barrier, just latency)
        printBarrierResults(rank, iterations, totalElapsed, latencies);
    }

    // ========================================================================
    // Result Reporting
    // ========================================================================

    private static void printResults(String operation, int rank, long sizeBytes,
                                     int iterations, long totalElapsedNs,
                                     long[] latencies, boolean correct) {
        // Calculate statistics
        double totalSeconds = totalElapsedNs / 1_000_000_000.0;
        double avgLatencyMs = (totalElapsedNs / (double) iterations) / 1_000_000.0;
        double minLatencyMs = Arrays.stream(latencies).min().orElse(0) / 1_000_000.0;
        double maxLatencyMs = Arrays.stream(latencies).max().orElse(0) / 1_000_000.0;

        // Calculate percentiles
        long[] sorted = latencies.clone();
        Arrays.sort(sorted);
        double p50LatencyMs = sorted[iterations / 2] / 1_000_000.0;
        double p95LatencyMs = sorted[(int) (iterations * 0.95)] / 1_000_000.0;
        double p99LatencyMs = sorted[(int) (iterations * 0.99)] / 1_000_000.0;

        // Calculate throughput
        double totalDataBytes = (double) sizeBytes * iterations;
        double throughputGbps = (totalDataBytes * BYTES_TO_GBITS) / totalSeconds;
        double opsPerSecond = iterations / totalSeconds;

        // Print results
        System.out.println();
        System.out.printf("[Rank %d] %s Results:%n", rank, operation);
        System.out.printf("  Correctness:    %s%n", correct ? "PASS" : "FAIL");
        System.out.printf("  Message size:   %s%n", formatBytes(sizeBytes));
        System.out.printf("  Iterations:     %d%n", iterations);
        System.out.printf("  Total time:     %.3f s%n", totalSeconds);
        System.out.printf("  Throughput:     %.2f Gbps%n", throughputGbps);
        System.out.printf("  Ops/second:     %.1f%n", opsPerSecond);
        System.out.println();
        System.out.printf("  Latency (ms):%n");
        System.out.printf("    Average:      %.3f ms%n", avgLatencyMs);
        System.out.printf("    Min:          %.3f ms%n", minLatencyMs);
        System.out.printf("    Max:          %.3f ms%n", maxLatencyMs);
        System.out.printf("    P50:          %.3f ms%n", p50LatencyMs);
        System.out.printf("    P95:          %.3f ms%n", p95LatencyMs);
        System.out.printf("    P99:          %.3f ms%n", p99LatencyMs);
        System.out.println();

        // Machine-parseable output line for shell script
        System.out.printf("PERF_RESULT: %s size=%d bw=%.2f Gbps lat=%.3f ms%n",
                operation, sizeBytes, throughputGbps, avgLatencyMs);
    }

    private static void printBarrierResults(int rank, int iterations,
                                            long totalElapsedNs, long[] latencies) {
        // Calculate statistics
        double totalSeconds = totalElapsedNs / 1_000_000_000.0;
        double avgLatencyUs = (totalElapsedNs / (double) iterations) / 1_000.0;
        double minLatencyUs = Arrays.stream(latencies).min().orElse(0) / 1_000.0;
        double maxLatencyUs = Arrays.stream(latencies).max().orElse(0) / 1_000.0;

        // Calculate percentiles
        long[] sorted = latencies.clone();
        Arrays.sort(sorted);
        double p50LatencyUs = sorted[iterations / 2] / 1_000.0;
        double p95LatencyUs = sorted[(int) (iterations * 0.95)] / 1_000.0;
        double p99LatencyUs = sorted[(int) (iterations * 0.99)] / 1_000.0;

        double opsPerSecond = iterations / totalSeconds;

        // Print results
        System.out.println();
        System.out.printf("[Rank %d] Barrier Results:%n", rank);
        System.out.printf("  Iterations:     %d%n", iterations);
        System.out.printf("  Total time:     %.3f s%n", totalSeconds);
        System.out.printf("  Ops/second:     %.1f%n", opsPerSecond);
        System.out.println();
        System.out.printf("  Latency (us):%n");
        System.out.printf("    Average:      %.1f us%n", avgLatencyUs);
        System.out.printf("    Min:          %.1f us%n", minLatencyUs);
        System.out.printf("    Max:          %.1f us%n", maxLatencyUs);
        System.out.printf("    P50:          %.1f us%n", p50LatencyUs);
        System.out.printf("    P95:          %.1f us%n", p95LatencyUs);
        System.out.printf("    P99:          %.1f us%n", p99LatencyUs);
        System.out.println();

        // Machine-parseable output line
        System.out.printf("PERF_RESULT: Barrier ops/s=%.1f lat=%.1f us%n", opsPerSecond, avgLatencyUs);
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    private static String formatBytes(long bytes) {
        if (bytes >= 1024L * 1024L * 1024L) {
            return String.format(Locale.US, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
        } else if (bytes >= 1024L * 1024L) {
            return String.format(Locale.US, "%.2f MB", bytes / (1024.0 * 1024.0));
        } else if (bytes >= 1024L) {
            return String.format(Locale.US, "%.2f KB", bytes / 1024.0);
        } else {
            return bytes + " B";
        }
    }

    private static void printUsage() {
        System.out.println("Usage: UccPerfTest [options]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --rank, -r <n>        Rank of this process (0 or 1)");
        System.out.println("  --world-size, -w <n>  Total number of processes (default: 2)");
        System.out.println("  --master, -m <addr>   Master address (default: 10.0.0.1)");
        System.out.println("  --port, -p <port>     Master port (default: 29500)");
        System.out.println("  --size, -s <bytes>    Message size in bytes (default: 1048576)");
        System.out.println("  --iterations, -i <n>  Number of iterations (default: 100)");
        System.out.println("  --warmup <n>          Number of warmup iterations (default: 10)");
        System.out.println("  --op <name>           Run only specific operation");
        System.out.println("                        (allreduce, allgather, broadcast, reducescatter, alltoall, barrier)");
        System.out.println("  --help, -h            Show this help");
        System.out.println();
        System.out.println("Example:");
        System.out.println("  # On mark1nvidia (master):");
        System.out.println("  ./gradlew :warpforge-io:uccPerfTest --args='--rank 0 --size 16777216 --iterations 1000'");
        System.out.println();
        System.out.println("  # On mark1amd (worker):");
        System.out.println("  ./gradlew :warpforge-io:uccPerfTest --args='--rank 1 --size 16777216 --iterations 1000'");
    }
}
