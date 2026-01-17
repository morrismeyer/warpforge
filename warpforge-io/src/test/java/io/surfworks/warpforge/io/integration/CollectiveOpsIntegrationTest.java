package io.surfworks.warpforge.io.integration;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.io.collective.AllReduceOp;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Ray-launched integration tests for all collective operations.
 *
 * <p>These tests verify that each collective operation works correctly
 * across the Mark 1 lab nodes using UCC over RDMA.
 *
 * <h2>Collective Operations Tested</h2>
 * <ul>
 *   <li>AllReduce (SUM, MAX, MIN, AVG, PROD) - {@link #testAllReduce*}</li>
 *   <li>AllGather - {@link #testAllGather()}</li>
 *   <li>Broadcast - {@link #testBroadcast()}</li>
 *   <li>ReduceScatter - {@link #testReduceScatter()}</li>
 *   <li>AllToAll - {@link #testAllToAll()}</li>
 *   <li>Reduce - {@link #testReduce()}</li>
 *   <li>Scatter - {@link #testScatter()}</li>
 *   <li>Gather - {@link #testGather()}</li>
 *   <li>Barrier - {@link #testBarrier()}</li>
 *   <li>Send/Recv - {@link #testPointToPoint()}</li>
 * </ul>
 *
 * <h2>Running</h2>
 * <pre>{@code
 * ./gradlew :warpforge-io:rayIntegrationTest \
 *     --tests "*CollectiveOpsIntegrationTest*" \
 *     -Drdma.world.size=2
 * }</pre>
 */
@Tag("ray-integration")
@Tag("rdma")
@DisplayName("Collective Operations Integration Tests")
class CollectiveOpsIntegrationTest extends RayIntegrationTestBase {

    // ===== AllReduce Tests =====

    @Test
    @DisplayName("AllReduce SUM: Sum values across ranks")
    void testAllReduceSum() throws Exception {
        final int SIZE = 1024;

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE)) {
            // Each rank fills with its rank value
            float[] data = new float[SIZE];
            for (int i = 0; i < SIZE; i++) {
                data[i] = rank + 1; // rank 0 -> 1, rank 1 -> 2
            }
            input.copyFrom(data);

            Tensor result = collective.allReduce(input, AllReduceOp.SUM)
                    .get(30, TimeUnit.SECONDS);

            // Expected: sum of all ranks' values = 1 + 2 + ... + worldSize
            float expectedSum = (worldSize * (worldSize + 1)) / 2.0f;

            float[] resultData = new float[SIZE];
            result.copyTo(resultData);

            for (int i = 0; i < SIZE; i++) {
                assertEquals(expectedSum, resultData[i], 0.001f,
                        "AllReduce SUM mismatch at index " + i);
            }

            log("AllReduce SUM verified: each element = %.1f", expectedSum);
            result.close();
        }
    }

    @Test
    @DisplayName("AllReduce MAX: Maximum across ranks")
    void testAllReduceMax() throws Exception {
        final int SIZE = 256;

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE)) {
            float[] data = new float[SIZE];
            for (int i = 0; i < SIZE; i++) {
                data[i] = rank * 10 + i; // Different values per rank
            }
            input.copyFrom(data);

            Tensor result = collective.allReduce(input, AllReduceOp.MAX)
                    .get(30, TimeUnit.SECONDS);

            float[] resultData = new float[SIZE];
            result.copyTo(resultData);

            // Max should be from highest rank
            for (int i = 0; i < SIZE; i++) {
                float expected = (worldSize - 1) * 10 + i;
                assertEquals(expected, resultData[i], 0.001f,
                        "AllReduce MAX mismatch at index " + i);
            }

            log("AllReduce MAX verified");
            result.close();
        }
    }

    @Test
    @DisplayName("AllReduce MIN: Minimum across ranks")
    void testAllReduceMin() throws Exception {
        final int SIZE = 256;

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE)) {
            float[] data = new float[SIZE];
            for (int i = 0; i < SIZE; i++) {
                data[i] = rank * 10 + i;
            }
            input.copyFrom(data);

            Tensor result = collective.allReduce(input, AllReduceOp.MIN)
                    .get(30, TimeUnit.SECONDS);

            float[] resultData = new float[SIZE];
            result.copyTo(resultData);

            // Min should be from rank 0
            for (int i = 0; i < SIZE; i++) {
                float expected = i; // rank 0's values
                assertEquals(expected, resultData[i], 0.001f,
                        "AllReduce MIN mismatch at index " + i);
            }

            log("AllReduce MIN verified");
            result.close();
        }
    }

    @Test
    @DisplayName("AllReduce AVG: Average across ranks")
    void testAllReduceAvg() throws Exception {
        final int SIZE = 256;

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE)) {
            float[] data = new float[SIZE];
            for (int i = 0; i < SIZE; i++) {
                data[i] = rank; // rank 0 -> 0, rank 1 -> 1, etc.
            }
            input.copyFrom(data);

            Tensor result = collective.allReduce(input, AllReduceOp.AVG)
                    .get(30, TimeUnit.SECONDS);

            float[] resultData = new float[SIZE];
            result.copyTo(resultData);

            // Average of 0, 1, 2, ..., (worldSize-1) = (worldSize-1)/2
            float expectedAvg = (worldSize - 1) / 2.0f;

            for (int i = 0; i < SIZE; i++) {
                assertEquals(expectedAvg, resultData[i], 0.001f,
                        "AllReduce AVG mismatch at index " + i);
            }

            log("AllReduce AVG verified: %.2f", expectedAvg);
            result.close();
        }
    }

    @Test
    @DisplayName("AllReduce PROD: Product across ranks")
    void testAllReduceProd() throws Exception {
        final int SIZE = 64;

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE)) {
            float[] data = new float[SIZE];
            for (int i = 0; i < SIZE; i++) {
                data[i] = rank + 1; // rank 0 -> 1, rank 1 -> 2
            }
            input.copyFrom(data);

            Tensor result = collective.allReduce(input, AllReduceOp.PROD)
                    .get(30, TimeUnit.SECONDS);

            float[] resultData = new float[SIZE];
            result.copyTo(resultData);

            // Product = 1 * 2 * ... * worldSize = worldSize!
            float expectedProd = 1;
            for (int r = 1; r <= worldSize; r++) {
                expectedProd *= r;
            }

            for (int i = 0; i < SIZE; i++) {
                assertEquals(expectedProd, resultData[i], 0.001f,
                        "AllReduce PROD mismatch at index " + i);
            }

            log("AllReduce PROD verified: %.1f", expectedProd);
            result.close();
        }
    }

    @Test
    @DisplayName("AllReduce in-place")
    void testAllReduceInPlace() throws Exception {
        final int SIZE = 512;

        try (Tensor tensor = Tensor.zeros(ScalarType.F32, SIZE)) {
            float[] data = new float[SIZE];
            for (int i = 0; i < SIZE; i++) {
                data[i] = 1.0f;
            }
            tensor.copyFrom(data);

            collective.allReduceInPlace(tensor, AllReduceOp.SUM)
                    .get(30, TimeUnit.SECONDS);

            float[] resultData = new float[SIZE];
            tensor.copyTo(resultData);

            for (int i = 0; i < SIZE; i++) {
                assertEquals(worldSize, resultData[i], 0.001f);
            }

            log("AllReduce in-place verified");
        }
    }

    // ===== AllGather Test =====

    @Test
    @DisplayName("AllGather: Gather and distribute tensors")
    void testAllGather() throws Exception {
        final int SIZE_PER_RANK = 128;

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE_PER_RANK)) {
            // Each rank fills with unique values
            float[] data = new float[SIZE_PER_RANK];
            for (int i = 0; i < SIZE_PER_RANK; i++) {
                data[i] = rank * 1000 + i;
            }
            input.copyFrom(data);

            Tensor result = collective.allGather(input).get(30, TimeUnit.SECONDS);

            // Result should have SIZE_PER_RANK * worldSize elements
            assertEquals(SIZE_PER_RANK * worldSize, result.numElements());

            float[] resultData = new float[(int) result.numElements()];
            result.copyTo(resultData);

            // Verify each rank's contribution
            for (int r = 0; r < worldSize; r++) {
                for (int i = 0; i < SIZE_PER_RANK; i++) {
                    float expected = r * 1000 + i;
                    int idx = r * SIZE_PER_RANK + i;
                    assertEquals(expected, resultData[idx], 0.001f,
                            "AllGather mismatch at rank " + r + " index " + i);
                }
            }

            log("AllGather verified: %d -> %d elements", SIZE_PER_RANK, result.numElements());
            result.close();
        }
    }

    // ===== Broadcast Test =====

    @Test
    @DisplayName("Broadcast: Send from root to all ranks")
    void testBroadcast() throws Exception {
        final int SIZE = 256;
        final float MAGIC = 42.5f;

        try (Tensor tensor = Tensor.zeros(ScalarType.F32, SIZE)) {
            // Only root fills the tensor
            if (isMaster()) {
                float[] data = new float[SIZE];
                for (int i = 0; i < SIZE; i++) {
                    data[i] = MAGIC + i;
                }
                tensor.copyFrom(data);
            }

            Tensor result = collective.broadcast(tensor, 0).get(30, TimeUnit.SECONDS);

            float[] resultData = new float[SIZE];
            result.copyTo(resultData);

            // All ranks should have root's data
            for (int i = 0; i < SIZE; i++) {
                assertEquals(MAGIC + i, resultData[i], 0.001f,
                        "Broadcast mismatch at index " + i);
            }

            log("Broadcast verified");
            result.close();
        }
    }

    // ===== ReduceScatter Test =====

    @Test
    @DisplayName("ReduceScatter: Reduce then scatter")
    void testReduceScatter() throws Exception {
        final int SIZE = 256 * worldSize; // Must be divisible by worldSize

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE)) {
            float[] data = new float[SIZE];
            for (int i = 0; i < SIZE; i++) {
                data[i] = 1.0f; // Each rank contributes 1
            }
            input.copyFrom(data);

            Tensor result = collective.reduceScatter(input, AllReduceOp.SUM)
                    .get(30, TimeUnit.SECONDS);

            // Each rank gets SIZE/worldSize elements
            assertEquals(SIZE / worldSize, result.numElements());

            float[] resultData = new float[(int) result.numElements()];
            result.copyTo(resultData);

            // After sum reduction, each element should equal worldSize
            for (int i = 0; i < result.numElements(); i++) {
                assertEquals(worldSize, resultData[i], 0.001f,
                        "ReduceScatter mismatch at index " + i);
            }

            log("ReduceScatter verified: %d -> %d elements", SIZE, result.numElements());
            result.close();
        }
    }

    // ===== AllToAll Test =====

    @Test
    @DisplayName("AllToAll: Exchange data between all ranks")
    void testAllToAll() throws Exception {
        final int CHUNK_SIZE = 64;
        final int SIZE = CHUNK_SIZE * worldSize;

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE)) {
            // Each rank prepares chunks for each destination
            float[] data = new float[SIZE];
            for (int dest = 0; dest < worldSize; dest++) {
                for (int i = 0; i < CHUNK_SIZE; i++) {
                    // Encode: src_rank * 1000 + dest_rank * 100 + offset
                    data[dest * CHUNK_SIZE + i] = rank * 1000 + dest * 100 + i;
                }
            }
            input.copyFrom(data);

            Tensor result = collective.allToAll(input).get(30, TimeUnit.SECONDS);

            float[] resultData = new float[SIZE];
            result.copyTo(resultData);

            // Verify: we should receive chunk from each source rank
            for (int src = 0; src < worldSize; src++) {
                for (int i = 0; i < CHUNK_SIZE; i++) {
                    // Data from src for us: src * 1000 + rank * 100 + i
                    float expected = src * 1000 + rank * 100 + i;
                    int idx = src * CHUNK_SIZE + i;
                    assertEquals(expected, resultData[idx], 0.001f,
                            "AllToAll mismatch from rank " + src + " at index " + i);
                }
            }

            log("AllToAll verified");
            result.close();
        }
    }

    // ===== Reduce Test =====

    @Test
    @DisplayName("Reduce: Combine to root only")
    void testReduce() throws Exception {
        final int SIZE = 128;

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE)) {
            float[] data = new float[SIZE];
            for (int i = 0; i < SIZE; i++) {
                data[i] = rank + 1;
            }
            input.copyFrom(data);

            Tensor result = collective.reduce(input, AllReduceOp.SUM, 0)
                    .get(30, TimeUnit.SECONDS);

            if (isMaster()) {
                float expectedSum = (worldSize * (worldSize + 1)) / 2.0f;

                float[] resultData = new float[SIZE];
                result.copyTo(resultData);

                for (int i = 0; i < SIZE; i++) {
                    assertEquals(expectedSum, resultData[i], 0.001f);
                }

                log("Reduce to root verified: sum = %.1f", expectedSum);
            }

            result.close();
        }
    }

    // ===== Scatter Test =====

    @Test
    @DisplayName("Scatter: Distribute from root")
    void testScatter() throws Exception {
        final int CHUNK_SIZE = 64;
        final int TOTAL_SIZE = CHUNK_SIZE * worldSize;

        try (Tensor input = Tensor.zeros(ScalarType.F32, TOTAL_SIZE)) {
            // Root prepares data for each rank
            if (isMaster()) {
                float[] data = new float[TOTAL_SIZE];
                for (int r = 0; r < worldSize; r++) {
                    for (int i = 0; i < CHUNK_SIZE; i++) {
                        data[r * CHUNK_SIZE + i] = r * 100 + i;
                    }
                }
                input.copyFrom(data);
            }

            Tensor result = collective.scatter(input, 0).get(30, TimeUnit.SECONDS);

            assertEquals(CHUNK_SIZE, result.numElements());

            float[] resultData = new float[CHUNK_SIZE];
            result.copyTo(resultData);

            // Each rank should get its chunk
            for (int i = 0; i < CHUNK_SIZE; i++) {
                assertEquals(rank * 100 + i, resultData[i], 0.001f);
            }

            log("Scatter verified: received chunk for rank %d", rank);
            result.close();
        }
    }

    // ===== Gather Test =====

    @Test
    @DisplayName("Gather: Collect to root")
    void testGather() throws Exception {
        final int SIZE = 64;

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE)) {
            // Each rank fills with unique data
            float[] data = new float[SIZE];
            for (int i = 0; i < SIZE; i++) {
                data[i] = rank * 1000 + i;
            }
            input.copyFrom(data);

            Tensor result = collective.gather(input, 0).get(30, TimeUnit.SECONDS);

            if (isMaster()) {
                assertEquals(SIZE * worldSize, result.numElements());

                float[] resultData = new float[(int) result.numElements()];
                result.copyTo(resultData);

                for (int r = 0; r < worldSize; r++) {
                    for (int i = 0; i < SIZE; i++) {
                        float expected = r * 1000 + i;
                        int idx = r * SIZE + i;
                        assertEquals(expected, resultData[idx], 0.001f);
                    }
                }

                log("Gather to root verified: collected from %d ranks", worldSize);
            }

            result.close();
        }
    }

    // ===== Barrier Test =====

    @Test
    @DisplayName("Barrier: Synchronize all ranks")
    void testBarrier() throws Exception {
        long startTime = System.currentTimeMillis();

        // Each rank waits a different amount
        Thread.sleep(rank * 100);

        collective.barrier().get(30, TimeUnit.SECONDS);

        long elapsed = System.currentTimeMillis() - startTime;
        log("Barrier completed after %d ms", elapsed);

        // All ranks should finish at approximately the same time
        // (within some tolerance)
        sync();
    }

    // ===== Point-to-Point Test =====

    @Test
    @DisplayName("Send/Recv: Point-to-point communication")
    void testPointToPoint() throws Exception {
        final int SIZE = 256;
        final int TAG = 42;

        try (Tensor sendTensor = Tensor.zeros(ScalarType.F32, SIZE);
             Tensor recvTensor = Tensor.zeros(ScalarType.F32, SIZE)) {

            if (rank == 0) {
                // Rank 0 sends to rank 1
                float[] data = new float[SIZE];
                for (int i = 0; i < SIZE; i++) {
                    data[i] = i * 2.5f;
                }
                sendTensor.copyFrom(data);

                collective.send(sendTensor, 1, TAG).get(30, TimeUnit.SECONDS);
                log("Sent %d elements to rank 1", SIZE);
            } else if (rank == 1) {
                // Rank 1 receives from rank 0
                collective.recv(recvTensor, 0, TAG).get(30, TimeUnit.SECONDS);

                float[] data = new float[SIZE];
                recvTensor.copyTo(data);

                for (int i = 0; i < SIZE; i++) {
                    assertEquals(i * 2.5f, data[i], 0.001f);
                }

                log("Received %d elements from rank 0", SIZE);
            }
        }
    }

    // ===== Large Tensor Test =====

    @Test
    @DisplayName("Large AllReduce: 256MB tensor")
    void testLargeAllReduce() throws Exception {
        final int SIZE = 64 * 1024 * 1024; // 64M floats = 256MB

        try (Tensor input = Tensor.zeros(ScalarType.F32, SIZE)) {
            // Fill with 1s
            float[] data = new float[SIZE];
            java.util.Arrays.fill(data, 1.0f);
            input.copyFrom(data);

            long start = System.nanoTime();

            Tensor result = collective.allReduce(input, AllReduceOp.SUM)
                    .get(120, TimeUnit.SECONDS);

            long elapsed = System.nanoTime() - start;
            double seconds = elapsed / 1e9;
            double gbps = (SIZE * 4.0 * 8 * 2) / elapsed; // 2x for ring

            log("Large AllReduce: %d MB in %.2f s (%.2f Gbps effective)",
                    SIZE * 4 / (1024 * 1024), seconds, gbps);

            // Verify first and last elements
            assertEquals(worldSize, result.getFloatFlat(0), 0.001f);
            assertEquals(worldSize, result.getFloatFlat(SIZE - 1), 0.001f);

            result.close();
        }
    }
}
