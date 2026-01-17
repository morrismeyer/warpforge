package io.surfworks.warpforge.core.tensor;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorTest {

    @Nested
    @DisplayName("Factory Methods")
    class FactoryMethods {

        @Test
        void zerosCreatesZeroFilledTensor() {
            try (Tensor t = Tensor.zeros(2, 3)) {
                assertEquals(2, t.rank());
                assertArrayEquals(new int[]{2, 3}, t.shape());
                assertEquals(6, t.elementCount());
                assertEquals(ScalarType.F32, t.dtype());

                // All elements should be zero
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 3; j++) {
                        assertEquals(0.0f, t.getFloat(i, j), 1e-9f);
                    }
                }
            }
        }

        @Test
        void zerosWithDtypeRespectsDtype() {
            try (Tensor t = Tensor.zeros(ScalarType.F64, 4)) {
                assertEquals(ScalarType.F64, t.dtype());
                assertEquals(4 * 8, t.spec().byteSize()); // 4 elements * 8 bytes
            }
        }

        @Test
        void fullCreatesFilledTensor() {
            try (Tensor t = Tensor.full(3.14f, 2, 2)) {
                assertEquals(4, t.elementCount());
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        assertEquals(3.14f, t.getFloat(i, j), 1e-6f);
                    }
                }
            }
        }

        @Test
        void fromFloatArrayCreatesCorrectTensor() {
            float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            try (Tensor t = Tensor.fromFloatArray(data, 2, 3)) {
                assertArrayEquals(new int[]{2, 3}, t.shape());
                assertEquals(1.0f, t.getFloat(0, 0), 1e-9f);
                assertEquals(2.0f, t.getFloat(0, 1), 1e-9f);
                assertEquals(3.0f, t.getFloat(0, 2), 1e-9f);
                assertEquals(4.0f, t.getFloat(1, 0), 1e-9f);
                assertEquals(5.0f, t.getFloat(1, 1), 1e-9f);
                assertEquals(6.0f, t.getFloat(1, 2), 1e-9f);
            }
        }

        @Test
        void fromFloatArrayThrowsOnMismatch() {
            float[] data = {1.0f, 2.0f, 3.0f};
            assertThrows(IllegalArgumentException.class, () ->
                Tensor.fromFloatArray(data, 2, 3) // expects 6 elements
            );
        }

        @Test
        void fromDoubleArrayCreatesCorrectTensor() {
            double[] data = {1.0, 2.0, 3.0, 4.0};
            try (Tensor t = Tensor.fromDoubleArray(data, 2, 2)) {
                assertEquals(ScalarType.F64, t.dtype());
                assertEquals(1.0, t.getDouble(0, 0), 1e-15);
                assertEquals(4.0, t.getDouble(1, 1), 1e-15);
            }
        }

        @Test
        void fromIntArrayCreatesCorrectTensor() {
            int[] data = {1, 2, 3, 4};
            try (Tensor t = Tensor.fromIntArray(data, 4)) {
                assertEquals(ScalarType.I32, t.dtype());
                assertEquals(1, t.getInt(0));
                assertEquals(4, t.getInt(3));
            }
        }
    }

    @Nested
    @DisplayName("Element Access")
    class ElementAccess {

        @Test
        void getSetFloat() {
            try (Tensor t = Tensor.zeros(3, 3)) {
                t.setFloat(42.0f, 1, 2);
                assertEquals(42.0f, t.getFloat(1, 2), 1e-9f);
                assertEquals(0.0f, t.getFloat(0, 0), 1e-9f);
            }
        }

        @Test
        void getSetFloatFlat() {
            try (Tensor t = Tensor.zeros(2, 3)) {
                t.setFloatFlat(5, 99.0f);
                assertEquals(99.0f, t.getFloatFlat(5), 1e-9f);
                // Index 5 in 2x3 tensor is [1][2]
                assertEquals(99.0f, t.getFloat(1, 2), 1e-9f);
            }
        }

        @Test
        void getSetDouble() {
            try (Tensor t = Tensor.zeros(ScalarType.F64, 2, 2)) {
                t.setDouble(Math.PI, 0, 1);
                assertEquals(Math.PI, t.getDouble(0, 1), 1e-15);
            }
        }

        @Test
        void getSetInt() {
            int[] data = {0, 0, 0, 0};
            try (Tensor t = Tensor.fromIntArray(data, 2, 2)) {
                t.setInt(123, 1, 0);
                assertEquals(123, t.getInt(1, 0));
            }
        }

        @Test
        void indexOutOfBoundsThrows() {
            try (Tensor t = Tensor.zeros(2, 3)) {
                assertThrows(IndexOutOfBoundsException.class, () -> t.getFloat(2, 0));
                assertThrows(IndexOutOfBoundsException.class, () -> t.getFloat(0, 3));
                assertThrows(IndexOutOfBoundsException.class, () -> t.getFloat(-1, 0));
            }
        }

        @Test
        void wrongNumberOfIndicesThrows() {
            try (Tensor t = Tensor.zeros(2, 3)) {
                assertThrows(IllegalArgumentException.class, () -> t.getFloat(0));
                assertThrows(IllegalArgumentException.class, () -> t.getFloat(0, 0, 0));
            }
        }
    }

    @Nested
    @DisplayName("Bulk Operations")
    class BulkOperations {

        @Test
        void toFloatArrayReturnsCorrectData() {
            float[] original = {1.0f, 2.0f, 3.0f, 4.0f};
            try (Tensor t = Tensor.fromFloatArray(original, 2, 2)) {
                float[] result = t.toFloatArray();
                assertArrayEquals(original, result, 1e-9f);
            }
        }

        @Test
        void toDoubleArrayReturnsCorrectData() {
            double[] original = {1.0, 2.0, 3.0};
            try (Tensor t = Tensor.fromDoubleArray(original, 3)) {
                double[] result = t.toDoubleArray();
                assertArrayEquals(original, result, 1e-15);
            }
        }

        @Test
        void copyFromUpdatesData() {
            try (Tensor t = Tensor.zeros(2, 2)) {
                float[] newData = {5.0f, 6.0f, 7.0f, 8.0f};
                t.copyFrom(newData);
                assertArrayEquals(newData, t.toFloatArray(), 1e-9f);
            }
        }

        @Test
        void copyFromThrowsOnSizeMismatch() {
            try (Tensor t = Tensor.zeros(2, 2)) {
                float[] wrongSize = {1.0f, 2.0f};
                assertThrows(IllegalArgumentException.class, () -> t.copyFrom(wrongSize));
            }
        }

        @Test
        void copyCreatesIndependentTensor() {
            float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
            try (Tensor original = Tensor.fromFloatArray(data, 2, 2)) {
                try (Tensor copied = original.copy()) {
                    // Modify original
                    original.setFloat(99.0f, 0, 0);

                    // Copy should be unaffected
                    assertEquals(1.0f, copied.getFloat(0, 0), 1e-9f);
                    assertEquals(99.0f, original.getFloat(0, 0), 1e-9f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Reshape")
    class ReshapeTests {

        @Test
        void reshapeCreatesViewWithNewShape() {
            float[] data = {1, 2, 3, 4, 5, 6};
            try (Tensor t = Tensor.fromFloatArray(data, 2, 3)) {
                Tensor reshaped = t.reshape(3, 2);
                assertArrayEquals(new int[]{3, 2}, reshaped.shape());
                assertEquals(6, reshaped.elementCount());

                // Data should be shared
                t.setFloat(99.0f, 0, 0);
                assertEquals(99.0f, reshaped.getFloat(0, 0), 1e-9f);
            }
        }

        @Test
        void reshapeToFlatArray() {
            try (Tensor t = Tensor.zeros(2, 3, 4)) {
                Tensor flat = t.reshape(24);
                assertEquals(1, flat.rank());
                assertEquals(24, flat.elementCount());
            }
        }

        @Test
        void reshapeThrowsOnElementCountMismatch() {
            try (Tensor t = Tensor.zeros(2, 3)) {
                assertThrows(IllegalArgumentException.class, () -> t.reshape(2, 2));
            }
        }
    }

    @Nested
    @DisplayName("TensorSpec")
    class TensorSpecTests {

        @Test
        void rowMajorStrides() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            // Row-major: last dimension varies fastest
            assertArrayEquals(new long[]{12, 4, 1}, spec.strides());
        }

        @Test
        void columnMajorStrides() {
            long[] strides = TensorSpec.computeColumnMajorStrides(new int[]{2, 3, 4});
            // Column-major: first dimension varies fastest
            assertArrayEquals(new long[]{1, 2, 6}, strides);
        }

        @Test
        void flatIndexRowMajor() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            // Row-major: [row][col] -> row * 3 + col
            assertEquals(0, spec.flatIndex(0, 0));
            assertEquals(1, spec.flatIndex(0, 1));
            assertEquals(2, spec.flatIndex(0, 2));
            assertEquals(3, spec.flatIndex(1, 0));
            assertEquals(5, spec.flatIndex(1, 2));
        }

        @Test
        void scalarTensorHasOneElement() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32);
            assertEquals(0, spec.rank());
            assertEquals(1, spec.elementCount());
        }

        @Test
        void isContiguousForRowMajor() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            assertTrue(spec.isContiguous());
        }

        @Test
        void broadcastableShapes() {
            TensorSpec a = TensorSpec.of(ScalarType.F32, 1, 3);
            TensorSpec b = TensorSpec.of(ScalarType.F32, 2, 3);
            assertTrue(a.isBroadcastableWith(b));

            TensorSpec c = TensorSpec.of(ScalarType.F32, 2, 4);
            assertFalse(b.isBroadcastableWith(c));
        }
    }

    @Nested
    @DisplayName("ScalarType")
    class ScalarTypeTests {

        @Test
        void byteSizes() {
            assertEquals(4, ScalarType.F32.byteSize());
            assertEquals(8, ScalarType.F64.byteSize());
            assertEquals(4, ScalarType.I32.byteSize());
            assertEquals(8, ScalarType.I64.byteSize());
            assertEquals(2, ScalarType.F16.byteSize());
        }

        @Test
        void isFloating() {
            assertTrue(ScalarType.F32.isFloating());
            assertTrue(ScalarType.F64.isFloating());
            assertFalse(ScalarType.I32.isFloating());
        }

        @Test
        void isInteger() {
            assertTrue(ScalarType.I32.isInteger());
            assertTrue(ScalarType.I64.isInteger());
            assertFalse(ScalarType.F32.isInteger());
        }

        @Test
        void npyDtypeRoundTrip() {
            assertEquals(ScalarType.F32, ScalarType.fromNpyDtype("<f4"));
            assertEquals(ScalarType.F64, ScalarType.fromNpyDtype("<f8"));
            assertEquals(ScalarType.I32, ScalarType.fromNpyDtype("<i4"));
            assertEquals(ScalarType.I64, ScalarType.fromNpyDtype(">i8")); // big-endian
        }
    }

    @Test
    @DisplayName("Tensor toString includes shape and dtype")
    void toStringIncludesMetadata() {
        try (Tensor t = Tensor.zeros(2, 3)) {
            String str = t.toString();
            assertTrue(str.contains("[2, 3]"));
            assertTrue(str.contains("F32"));
        }
    }

    @Test
    @DisplayName("Tensor is AutoCloseable")
    void autoCloseableWorks() {
        Tensor t = Tensor.zeros(10);
        assertDoesNotThrow(t::close);
    }
}
