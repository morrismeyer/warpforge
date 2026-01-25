package io.surfworks.warpforge.core.tensor.typed;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.device.Nvidia;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Tests for TensorArena scoped memory management.
 */
@DisplayName("TensorArena")
class TensorArenaTest {

    private static final float EPSILON = 1e-6f;

    @Nested
    @DisplayName("Factory Methods")
    class FactoryMethods {

        @Test
        @DisplayName("ofConfined creates confined arena")
        void ofConfinedCreatesConfinedArena() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                assertNotNull(arena);
                assertNotNull(arena.underlying());
                assertFalse(arena.isClosed());
            }
        }

        @Test
        @DisplayName("ofShared creates shared arena")
        void ofSharedCreatesSharedArena() {
            try (TensorArena arena = TensorArena.ofShared()) {
                assertNotNull(arena);
                assertFalse(arena.isClosed());
            }
        }

        @Test
        @DisplayName("ofAuto creates auto arena")
        void ofAutoCreatesAutoArena() {
            TensorArena arena = TensorArena.ofAuto();
            assertNotNull(arena);
            assertFalse(arena.isClosed());
            // Auto arenas are GC-managed, don't need explicit close
        }
    }

    @Nested
    @DisplayName("Allocation Methods")
    class AllocationMethods {

        @Test
        @DisplayName("zeros allocates zero tensor")
        void zerosAllocatesZeroTensor() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                TypedTensor<Matrix, F32, Cpu> tensor = arena.zeros(
                        new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);

                assertEquals(2, tensor.rank());
                assertEquals(12, tensor.elementCount());

                float[] data = tensor.underlying().toFloatArray();
                for (float v : data) {
                    assertEquals(0.0f, v);
                }
            }
        }

        @Test
        @DisplayName("full allocates filled tensor")
        void fullAllocatesFilledTensor() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                TypedTensor<Vector, F32, Cpu> tensor = arena.full(
                        2.5f, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);

                float[] data = tensor.underlying().toFloatArray();
                for (float v : data) {
                    assertEquals(2.5f, v, EPSILON);
                }
            }
        }

        @Test
        @DisplayName("fromFloatArray copies data")
        void fromFloatArrayCopiesData() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                float[] source = {1, 2, 3, 4, 5, 6};
                TypedTensor<Matrix, F32, Cpu> tensor = arena.fromFloatArray(
                        source, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);

                assertArrayEquals(source, tensor.underlying().toFloatArray());
            }
        }

        @Test
        @DisplayName("fromDoubleArray copies data")
        void fromDoubleArrayCopiesData() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                double[] source = {1.1, 2.2, 3.3};
                TypedTensor<Vector, F64, Cpu> tensor = arena.fromDoubleArray(
                        source, new Vector(3), F64.INSTANCE, Cpu.INSTANCE);

                assertArrayEquals(source, tensor.underlying().toDoubleArray());
            }
        }

        @Test
        @DisplayName("fromFloatArray rejects shape mismatch")
        void fromFloatArrayRejectsShapeMismatch() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                float[] source = {1, 2, 3};

                assertThrows(IllegalArgumentException.class, () ->
                        arena.fromFloatArray(source, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE));
            }
        }

        @Test
        @DisplayName("copy creates independent tensor")
        void copyCopiesData() {
            try (TensorArena arena = TensorArena.ofConfined();
                 TypedTensor<Vector, F32, Cpu> original = TypedTensor.fromFloatArray(
                         new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<Vector, F32, Cpu> copy = arena.copy(original);

                assertNotSame(original.underlying(), copy.underlying());
                assertArrayEquals(original.underlying().toFloatArray(),
                                  copy.underlying().toFloatArray());
            }
        }
    }

    @Nested
    @DisplayName("Tensor Counting")
    class TensorCounting {

        @Test
        @DisplayName("tensorCount starts at zero")
        void tensorCountStartsAtZero() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                assertEquals(0, arena.tensorCount());
            }
        }

        @Test
        @DisplayName("tensorCount increases with allocations")
        void tensorCountIncreases() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                arena.zeros(new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                assertEquals(1, arena.tensorCount());

                arena.zeros(new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                assertEquals(2, arena.tensorCount());

                arena.zeros(new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                assertEquals(3, arena.tensorCount());
            }
        }
    }

    @Nested
    @DisplayName("Lifecycle")
    class Lifecycle {

        @Test
        @DisplayName("close marks arena as closed")
        void closeMarksArenaAsClosed() {
            TensorArena arena = TensorArena.ofConfined();
            assertFalse(arena.isClosed());

            arena.close();
            assertTrue(arena.isClosed());
        }

        @Test
        @DisplayName("allocation after close throws")
        void allocationAfterCloseThrows() {
            TensorArena arena = TensorArena.ofConfined();
            arena.close();

            assertThrows(IllegalStateException.class, () ->
                    arena.zeros(new Vector(5), F32.INSTANCE, Cpu.INSTANCE));
        }

        @Test
        @DisplayName("double close is safe")
        void doubleCloseIsSafe() {
            TensorArena arena = TensorArena.ofConfined();
            arena.close();
            arena.close(); // Should not throw
            assertTrue(arena.isClosed());
        }

        @Test
        @DisplayName("try-with-resources closes arena")
        void tryWithResourcesClosesArena() {
            TensorArena arena;
            try (TensorArena a = TensorArena.ofConfined()) {
                arena = a;
                assertFalse(arena.isClosed());
            }
            assertTrue(arena.isClosed());
        }
    }

    @Nested
    @DisplayName("GPU Validation")
    class GpuValidation {

        @Test
        @DisplayName("zeros rejects GPU device")
        void zerosRejectsGpu() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                assertThrows(UnsupportedOperationException.class, () ->
                        arena.zeros(new Vector(5), F32.INSTANCE, Nvidia.DEFAULT));
            }
        }

        @Test
        @DisplayName("full rejects GPU device")
        void fullRejectsGpu() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                assertThrows(UnsupportedOperationException.class, () ->
                        arena.full(1.0f, new Vector(5), F32.INSTANCE, Nvidia.DEFAULT));
            }
        }
    }

    @Nested
    @DisplayName("Integration")
    class Integration {

        @Test
        @DisplayName("multiple tensors share arena lifetime")
        void multipleTensorsShareArenaLifetime() {
            TypedTensor<Matrix, F32, Cpu> a;
            TypedTensor<Matrix, F32, Cpu> b;
            TypedTensor<Vector, F32, Cpu> c;

            try (TensorArena arena = TensorArena.ofConfined()) {
                a = arena.zeros(new Matrix(10, 10), F32.INSTANCE, Cpu.INSTANCE);
                b = arena.zeros(new Matrix(10, 10), F32.INSTANCE, Cpu.INSTANCE);
                c = arena.zeros(new Vector(100), F32.INSTANCE, Cpu.INSTANCE);

                assertEquals(3, arena.tensorCount());

                // Tensors are usable within the scope
                assertNotNull(a.underlying().data());
                assertNotNull(b.underlying().data());
                assertNotNull(c.underlying().data());
            }

            // After scope, arena is closed - tensors should not be accessed
            // (accessing them would be undefined behavior in native memory)
        }

        @Test
        @DisplayName("arena with typed operations")
        void arenaWithTypedOperations() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                var a = arena.fromFloatArray(
                        new float[]{1, 2, 3, 4}, new Matrix(2, 2), F32.INSTANCE, Cpu.INSTANCE);
                var b = arena.fromFloatArray(
                        new float[]{5, 6, 7, 8}, new Matrix(2, 2), F32.INSTANCE, Cpu.INSTANCE);

                // Operations create new tensors (outside the arena)
                try (var result = io.surfworks.warpforge.core.tensor.typed.ops.MatrixOps.matmul(a, b)) {
                    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
                    float[] expected = {19, 22, 43, 50};
                    assertArrayEquals(expected, result.underlying().toFloatArray(), EPSILON);
                }
            }
        }
    }

    @Nested
    @DisplayName("toString")
    class ToStringTests {

        @Test
        @DisplayName("toString shows tensor count")
        void toStringShowsTensorCount() {
            try (TensorArena arena = TensorArena.ofConfined()) {
                arena.zeros(new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                arena.zeros(new Vector(5), F32.INSTANCE, Cpu.INSTANCE);

                String str = arena.toString();
                assertTrue(str.contains("tensors=2"));
                assertTrue(str.contains("closed=false"));
            }
        }

        @Test
        @DisplayName("toString shows closed state")
        void toStringShowsClosedState() {
            TensorArena arena = TensorArena.ofConfined();
            arena.close();

            String str = arena.toString();
            assertTrue(str.contains("closed=true"));
        }
    }
}
