package io.surfworks.warpforge.io.test;

import java.util.Arrays;

import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.io.collective.AllReduceOp;
import io.surfworks.warpforge.io.collective.Collective;
import io.surfworks.warpforge.io.collective.CollectiveApi;
import io.surfworks.warpforge.io.collective.CollectiveConfig;

/**
 * Two-node UCC collective operations test.
 *
 * <p>This is a standalone test runner for verifying UCC collective operations
 * between two GPU nodes (mark1nvidia and mark1amd) over the ConnectX-5 cross-connect.
 *
 * <h2>Usage</h2>
 * <pre>
 * # On mark1nvidia (10.0.0.1) - run as rank 0 (master):
 * ./gradlew :warpforge-io:twoNodeTest --args='--rank 0 --world-size 2 --master 10.0.0.1 --port 29500'
 *
 * # On mark1amd (10.0.0.2) - run as rank 1 (worker):
 * ./gradlew :warpforge-io:twoNodeTest --args='--rank 1 --world-size 2 --master 10.0.0.1 --port 29500'
 * </pre>
 *
 * <h2>Network Topology</h2>
 * <pre>
 * mark1nvidia (10.0.0.1) <-- QSFP28 100GbE --> mark1amd (10.0.0.2)
 * </pre>
 */
public class TwoNodeCollectiveTest {

    public static void main(String[] args) {
        // Parse arguments
        int rank = 0;
        int worldSize = 2;
        String masterAddr = "10.0.0.1";
        int masterPort = 29500;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--rank", "-r" -> rank = Integer.parseInt(args[++i]);
                case "--world-size", "-w" -> worldSize = Integer.parseInt(args[++i]);
                case "--master", "-m" -> masterAddr = args[++i];
                case "--port", "-p" -> masterPort = Integer.parseInt(args[++i]);
                case "--help", "-h" -> {
                    printUsage();
                    return;
                }
            }
        }

        System.out.println("=".repeat(60));
        System.out.println("Two-Node UCC Collective Test");
        System.out.println("=".repeat(60));
        System.out.printf("Rank: %d / %d%n", rank, worldSize);
        System.out.printf("Master: %s:%d%n", masterAddr, masterPort);
        System.out.println(Collective.systemInfo());

        // Create configuration
        CollectiveConfig config = CollectiveConfig.builder(worldSize, rank)
                .masterAddress(masterAddr)
                .masterPort(masterPort)
                .build();

        // Run tests
        try (CollectiveApi collective = Collective.loadUcc(config)) {
            System.out.printf("[Rank %d] UCC initialized, backend: %s%n", rank, collective.backendName());

            // Test 1: Barrier
            testBarrier(collective, rank);

            // Test 2: AllReduce SUM
            testAllReduceSum(collective, rank, worldSize);

            // Test 3: AllGather
            testAllGather(collective, rank, worldSize);

            // Test 4: Broadcast
            testBroadcast(collective, rank);

            // Test 5: Large AllReduce (performance)
            testLargeAllReduce(collective, rank, worldSize);

            // Final barrier
            System.out.printf("[Rank %d] All tests completed, syncing...%n", rank);
            collective.barrier().join();

            System.out.println("=".repeat(60));
            System.out.printf("[Rank %d] SUCCESS - All tests passed!%n", rank);
            System.out.println("Stats: " + collective.stats());
            System.out.println("=".repeat(60));

        } catch (Exception e) {
            System.err.printf("[Rank %d] FAILED: %s%n", rank, e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void testBarrier(CollectiveApi collective, int rank) throws Exception {
        System.out.printf("[Rank %d] Test 1: Barrier...%n", rank);
        long start = System.nanoTime();
        collective.barrier().join();
        long elapsed = System.nanoTime() - start;
        System.out.printf("[Rank %d] Barrier completed in %.3f ms%n", rank, elapsed / 1e6);
    }

    private static void testAllReduceSum(CollectiveApi collective, int rank, int worldSize) throws Exception {
        System.out.printf("[Rank %d] Test 2: AllReduce SUM...%n", rank);

        int size = 1024;
        float[] data = new float[size];
        Arrays.fill(data, rank + 1.0f); // rank 0 -> 1.0, rank 1 -> 2.0

        Tensor input = Tensor.fromFloatArray(data, size);
        long start = System.nanoTime();
        Tensor result = collective.allReduce(input, AllReduceOp.SUM).join();
        long elapsed = System.nanoTime() - start;

        // Verify: sum should be 1 + 2 + ... + worldSize = worldSize * (worldSize + 1) / 2
        float expected = worldSize * (worldSize + 1) / 2.0f;
        float actual = result.getFloatFlat(0);

        if (Math.abs(actual - expected) > 0.001f) {
            throw new AssertionError(String.format(
                "AllReduce SUM failed: expected %.1f, got %.1f", expected, actual));
        }

        System.out.printf("[Rank %d] AllReduce SUM verified: %.1f (expected %.1f) in %.3f ms%n",
                rank, actual, expected, elapsed / 1e6);

        input.close();
        result.close();
    }

    private static void testAllGather(CollectiveApi collective, int rank, int worldSize) throws Exception {
        System.out.printf("[Rank %d] Test 3: AllGather...%n", rank);

        int sizePerRank = 128;
        float[] data = new float[sizePerRank];
        for (int i = 0; i < sizePerRank; i++) {
            data[i] = rank * 1000 + i;
        }

        Tensor input = Tensor.fromFloatArray(data, sizePerRank);
        long start = System.nanoTime();
        Tensor result = collective.allGather(input).join();
        long elapsed = System.nanoTime() - start;

        // Verify we have data from all ranks
        long expectedElements = (long) sizePerRank * worldSize;
        if (result.numElements() != expectedElements) {
            throw new AssertionError(String.format(
                "AllGather size mismatch: expected %d, got %d", expectedElements, result.numElements()));
        }

        // Verify first element from each rank
        for (int r = 0; r < worldSize; r++) {
            float expected = r * 1000;
            float actual = result.getFloatFlat(r * sizePerRank);
            if (Math.abs(actual - expected) > 0.001f) {
                throw new AssertionError(String.format(
                    "AllGather data mismatch at rank %d: expected %.1f, got %.1f", r, expected, actual));
            }
        }

        System.out.printf("[Rank %d] AllGather verified: %d -> %d elements in %.3f ms%n",
                rank, sizePerRank, result.numElements(), elapsed / 1e6);

        input.close();
        result.close();
    }

    private static void testBroadcast(CollectiveApi collective, int rank) throws Exception {
        System.out.printf("[Rank %d] Test 4: Broadcast from rank 0...%n", rank);

        int size = 256;
        float[] data = new float[size];
        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                data[i] = 42.0f + i;
            }
        }

        Tensor input = Tensor.fromFloatArray(data, size);
        long start = System.nanoTime();
        Tensor result = collective.broadcast(input, 0).join();
        long elapsed = System.nanoTime() - start;

        // All ranks should have rank 0's data
        float expected = 42.0f;
        float actual = result.getFloatFlat(0);
        if (Math.abs(actual - expected) > 0.001f) {
            throw new AssertionError(String.format(
                "Broadcast failed: expected %.1f, got %.1f", expected, actual));
        }

        System.out.printf("[Rank %d] Broadcast verified: first element = %.1f in %.3f ms%n",
                rank, actual, elapsed / 1e6);

        input.close();
        result.close();
    }

    private static void testLargeAllReduce(CollectiveApi collective, int rank, int worldSize) throws Exception {
        System.out.printf("[Rank %d] Test 5: Large AllReduce (64MB)...%n", rank);

        int size = 16 * 1024 * 1024; // 16M floats = 64MB
        float[] data = new float[size];
        Arrays.fill(data, 1.0f);

        Tensor input = Tensor.fromFloatArray(data, size);

        // Warmup
        collective.allReduce(input, AllReduceOp.SUM).join().close();

        // Timed run
        long start = System.nanoTime();
        Tensor result = collective.allReduce(input, AllReduceOp.SUM).join();
        long elapsed = System.nanoTime() - start;

        // Verify
        float expected = worldSize;
        float actual = result.getFloatFlat(0);
        if (Math.abs(actual - expected) > 0.001f) {
            throw new AssertionError(String.format(
                "Large AllReduce failed: expected %.1f, got %.1f", expected, actual));
        }

        // Calculate throughput
        double seconds = elapsed / 1e9;
        double mbytes = size * 4.0 / (1024 * 1024);
        double throughputMBps = mbytes / seconds;
        double throughputGbps = throughputMBps * 8 / 1000;

        System.out.printf("[Rank %d] Large AllReduce: %.1f MB in %.3f s = %.2f MB/s (%.2f Gbps)%n",
                rank, mbytes, seconds, throughputMBps, throughputGbps);

        input.close();
        result.close();
    }

    private static void printUsage() {
        System.out.println("Usage: TwoNodeCollectiveTest [options]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --rank, -r <n>        Rank of this process (0 or 1)");
        System.out.println("  --world-size, -w <n>  Total number of processes (default: 2)");
        System.out.println("  --master, -m <addr>   Master address (default: 10.0.0.1)");
        System.out.println("  --port, -p <port>     Master port (default: 29500)");
        System.out.println("  --help, -h            Show this help");
        System.out.println();
        System.out.println("Example:");
        System.out.println("  # On mark1nvidia (master):");
        System.out.println("  ./gradlew :warpforge-io:twoNodeTest --args='--rank 0'");
        System.out.println();
        System.out.println("  # On mark1amd (worker):");
        System.out.println("  ./gradlew :warpforge-io:twoNodeTest --args='--rank 1'");
    }
}
