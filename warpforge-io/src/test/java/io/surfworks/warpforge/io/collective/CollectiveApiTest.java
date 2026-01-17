package io.surfworks.warpforge.io.collective;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import org.junit.jupiter.api.*;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for CollectiveApi interface using mock implementation.
 * These tests verify all collective operations run correctly on NUC without actual distributed communication.
 */
@Tag("unit")
@DisplayName("Collective API Unit Tests")
class CollectiveApiTest {

    private CollectiveApi collective;

    @BeforeEach
    void setUp() {
        // Create mock collective with world size 2, rank 0
        collective = Collective.loadMock(2, 0);
    }

    @AfterEach
    void tearDown() {
        if (collective != null) collective.close();
    }

    @Test
    @DisplayName("Should return mock backend name")
    void testBackendName() {
        assertEquals("mock", collective.backendName());
    }

    @Test
    @DisplayName("Should return correct configuration")
    void testConfig() {
        assertEquals(2, collective.worldSize());
        assertEquals(0, collective.rank());
    }

    // ===== AllReduce Tests =====

    @Test
    @DisplayName("Should perform allreduce with SUM")
    void testAllReduceSum() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 4, 4)) {
            // Fill with test data
            float[] data = new float[16];
            for (int i = 0; i < 16; i++) data[i] = i;
            input.copyFrom(data);

            Tensor result = collective.allReduce(input, AllReduceOp.SUM).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            assertEquals(input.spec().byteSize(), result.spec().byteSize());
            assertArrayEquals(input.shape(), result.shape());
            result.close();
        }
    }

    @Test
    @DisplayName("Should perform allreduce with MAX")
    void testAllReduceMax() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) {
            Tensor result = collective.allReduce(input, AllReduceOp.MAX).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            result.close();
        }
    }

    @Test
    @DisplayName("Should perform allreduce with MIN")
    void testAllReduceMin() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) {
            Tensor result = collective.allReduce(input, AllReduceOp.MIN).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            result.close();
        }
    }

    @Test
    @DisplayName("Should perform allreduce with AVG")
    void testAllReduceAvg() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) {
            Tensor result = collective.allReduce(input, AllReduceOp.AVG).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            result.close();
        }
    }

    @Test
    @DisplayName("Should perform allreduce with PROD")
    void testAllReduceProd() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) {
            Tensor result = collective.allReduce(input, AllReduceOp.PROD).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            result.close();
        }
    }

    @Test
    @DisplayName("Should perform in-place allreduce")
    void testAllReduceInPlace() throws Exception {
        try (Tensor tensor = Tensor.zeros(ScalarType.F32, 8)) {
            collective.allReduceInPlace(tensor, AllReduceOp.SUM).get(5, TimeUnit.SECONDS);
            // In mock mode, tensor should be unchanged
        }
    }

    // ===== AllGather Tests =====

    @Test
    @DisplayName("Should perform allgather")
    void testAllGather() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 4)) {
            Tensor result = collective.allGather(input).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            // Result should be worldSize times larger in first dimension
            assertEquals(4 * collective.worldSize(), result.shape()[0]);
            result.close();
        }
    }

    @Test
    @DisplayName("Should perform allgather with explicit output")
    void testAllGatherWithOutput() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 4);
             Tensor output = Tensor.zeros(ScalarType.F32, 8)) {
            collective.allGather(input, output).get(5, TimeUnit.SECONDS);
        }
    }

    // ===== Broadcast Tests =====

    @Test
    @DisplayName("Should perform broadcast from root 0")
    void testBroadcastFromRoot0() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) {
            Tensor result = collective.broadcast(input, 0).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            assertArrayEquals(input.shape(), result.shape());
            result.close();
        }
    }

    @Test
    @DisplayName("Should perform broadcast from root 1")
    void testBroadcastFromRoot1() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) {
            Tensor result = collective.broadcast(input, 1).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            result.close();
        }
    }

    @Test
    @DisplayName("Should perform in-place broadcast")
    void testBroadcastInPlace() throws Exception {
        try (Tensor tensor = Tensor.zeros(ScalarType.F32, 8)) {
            collective.broadcastInPlace(tensor, 0).get(5, TimeUnit.SECONDS);
        }
    }

    @Test
    @DisplayName("Should reject invalid root rank")
    void testBroadcastInvalidRoot() {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) {
            assertThrows(CollectiveException.class, () ->
                    collective.broadcast(input, 99).get(5, TimeUnit.SECONDS));
        }
    }

    // ===== ReduceScatter Tests =====

    @Test
    @DisplayName("Should perform reduce-scatter")
    void testReduceScatter() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) { // Must be divisible by worldSize
            Tensor result = collective.reduceScatter(input, AllReduceOp.SUM).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            // Result should be worldSize times smaller in first dimension
            assertEquals(8 / collective.worldSize(), result.shape()[0]);
            result.close();
        }
    }

    @Test
    @DisplayName("Should perform reduce-scatter with explicit output")
    void testReduceScatterWithOutput() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8);
             Tensor output = Tensor.zeros(ScalarType.F32, 4)) {
            collective.reduceScatter(input, output, AllReduceOp.SUM).get(5, TimeUnit.SECONDS);
        }
    }

    // ===== AllToAll Tests =====

    @Test
    @DisplayName("Should perform all-to-all exchange")
    void testAllToAll() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) {
            Tensor result = collective.allToAll(input).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            assertArrayEquals(input.shape(), result.shape());
            result.close();
        }
    }

    @Test
    @DisplayName("Should perform all-to-all with explicit output")
    void testAllToAllWithOutput() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8);
             Tensor output = Tensor.zeros(ScalarType.F32, 8)) {
            collective.allToAll(input, output).get(5, TimeUnit.SECONDS);
        }
    }

    // ===== Reduce Tests =====

    @Test
    @DisplayName("Should perform reduce to root")
    void testReduce() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) {
            Tensor result = collective.reduce(input, AllReduceOp.SUM, 0).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            result.close();
        }
    }

    // ===== Scatter Tests =====

    @Test
    @DisplayName("Should perform scatter from root")
    void testScatter() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 8)) {
            Tensor result = collective.scatter(input, 0).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            assertEquals(8 / collective.worldSize(), result.shape()[0]);
            result.close();
        }
    }

    // ===== Gather Tests =====

    @Test
    @DisplayName("Should perform gather to root")
    void testGather() throws Exception {
        try (Tensor input = Tensor.zeros(ScalarType.F32, 4)) {
            Tensor result = collective.gather(input, 0).get(5, TimeUnit.SECONDS);
            assertNotNull(result);
            // On root, result should be worldSize times larger
            if (collective.rank() == 0) {
                assertEquals(4 * collective.worldSize(), result.shape()[0]);
            }
            result.close();
        }
    }

    // ===== Barrier Tests =====

    @Test
    @DisplayName("Should perform barrier synchronization")
    void testBarrier() throws Exception {
        collective.barrier().get(5, TimeUnit.SECONDS);
        // In mock mode, this completes immediately
    }

    // ===== Point-to-Point Tests =====

    @Test
    @DisplayName("Should send to rank")
    void testSend() throws Exception {
        try (Tensor tensor = Tensor.zeros(ScalarType.F32, 8)) {
            collective.send(tensor, 1, 0).get(5, TimeUnit.SECONDS);
        }
    }

    @Test
    @DisplayName("Should receive from rank")
    void testRecv() throws Exception {
        try (Tensor tensor = Tensor.zeros(ScalarType.F32, 8)) {
            collective.recv(tensor, 1, 0).get(5, TimeUnit.SECONDS);
        }
    }

    // ===== Statistics Tests =====

    @Test
    @DisplayName("Should track collective statistics")
    void testStats() throws Exception {
        try (Tensor tensor = Tensor.zeros(ScalarType.F32, 8)) {
            collective.allReduce(tensor, AllReduceOp.SUM).get();
            collective.broadcast(tensor, 0).get();
            collective.barrier().get();

            CollectiveApi.CollectiveStats stats = collective.stats();
            assertEquals(1, stats.allReduceCount());
            assertEquals(1, stats.broadcastCount());
            assertEquals(1, stats.barrierCount());
            assertTrue(stats.totalOperations() >= 3);
        }
    }

    // ===== Data Type Tests =====

    @Test
    @DisplayName("Should work with different scalar types")
    void testDifferentDtypes() throws Exception {
        // Float32
        try (Tensor f32 = Tensor.zeros(ScalarType.F32, 4)) {
            collective.allReduce(f32, AllReduceOp.SUM).get().close();
        }

        // Float64
        try (Tensor f64 = Tensor.zeros(ScalarType.F64, 4)) {
            collective.allReduce(f64, AllReduceOp.SUM).get().close();
        }

        // Int32
        try (Tensor i32 = Tensor.zeros(ScalarType.I32, 4)) {
            collective.allReduce(i32, AllReduceOp.SUM).get().close();
        }

        // Int64
        try (Tensor i64 = Tensor.zeros(ScalarType.I64, 4)) {
            collective.allReduce(i64, AllReduceOp.SUM).get().close();
        }
    }

    // ===== Lifecycle Tests =====

    @Test
    @DisplayName("Should throw when using closed context")
    void testClosedContextThrows() {
        collective.close();
        try (Tensor tensor = Tensor.zeros(ScalarType.F32, 8)) {
            assertThrows(CollectiveException.class, () ->
                    collective.allReduce(tensor, AllReduceOp.SUM));
        }
    }
}
