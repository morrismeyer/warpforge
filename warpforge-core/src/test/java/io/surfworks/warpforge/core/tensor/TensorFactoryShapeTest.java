package io.surfworks.warpforge.core.tensor;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Comprehensive tests for Tensor factory method shape validation.
 * These tests document expected behavior for shape-related errors and
 * serve as a specification for future compile-time type safety.
 */
@DisplayName("Tensor Factory Shape Validation")
class TensorFactoryShapeTest {

    @Nested
    @DisplayName("fromFloatArray - Data Length vs Shape Mismatch")
    class FromFloatArrayMismatch {

        @Test
        @DisplayName("Throws when data is too short for shape")
        void dataTooShort() {
            float[] data = new float[]{1, 2, 3}; // 3 elements
            var ex = assertThrows(IllegalArgumentException.class,
                () -> Tensor.fromFloatArray(data, 2, 3)); // expects 6 elements
            assertTrue(ex.getMessage().contains("3"),
                "Error message should include actual data length");
            assertTrue(ex.getMessage().contains("6") || ex.getMessage().contains("[2, 3]"),
                "Error message should include expected count or shape");
        }

        @Test
        @DisplayName("Throws when data is too long for shape")
        void dataTooLong() {
            float[] data = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; // 10 elements
            var ex = assertThrows(IllegalArgumentException.class,
                () -> Tensor.fromFloatArray(data, 2, 3)); // expects 6 elements
            assertTrue(ex.getMessage().contains("10"),
                "Error message should include actual data length");
        }

        @Test
        @DisplayName("Succeeds when data length exactly matches shape")
        void exactMatch() {
            float[] data = new float[]{1, 2, 3, 4, 5, 6};
            try (Tensor t = Tensor.fromFloatArray(data, 2, 3)) {
                assertEquals(6, t.elementCount());
                assertArrayEquals(new int[]{2, 3}, t.shape());
            }
        }

        @Test
        @DisplayName("Throws when data length is 1 but shape is larger")
        void singleElementForLargerShape() {
            float[] data = new float[]{42.0f};
            assertThrows(IllegalArgumentException.class,
                () -> Tensor.fromFloatArray(data, 2, 3));
        }

        @Test
        @DisplayName("Succeeds with empty array and zero-sized dimension")
        void emptyArrayZeroShape() {
            float[] data = new float[0];
            try (Tensor t = Tensor.fromFloatArray(data, 0)) {
                assertEquals(0, t.elementCount());
            }
        }

        @Test
        @DisplayName("Succeeds with empty array and multi-dimensional zero shape")
        void emptyArrayMultiZeroShape() {
            float[] data = new float[0];
            try (Tensor t = Tensor.fromFloatArray(data, 2, 0, 3)) {
                assertEquals(0, t.elementCount());
            }
        }
    }

    @Nested
    @DisplayName("fromDoubleArray - Data Length vs Shape Mismatch")
    class FromDoubleArrayMismatch {

        @Test
        @DisplayName("Throws when data is too short for shape")
        void dataTooShort() {
            double[] data = new double[]{1.0, 2.0}; // 2 elements
            assertThrows(IllegalArgumentException.class,
                () -> Tensor.fromDoubleArray(data, 2, 2)); // expects 4 elements
        }

        @Test
        @DisplayName("Throws when data is too long for shape")
        void dataTooLong() {
            double[] data = new double[]{1.0, 2.0, 3.0, 4.0, 5.0}; // 5 elements
            assertThrows(IllegalArgumentException.class,
                () -> Tensor.fromDoubleArray(data, 2, 2)); // expects 4 elements
        }

        @Test
        @DisplayName("Succeeds when data length exactly matches shape")
        void exactMatch() {
            double[] data = new double[]{1.0, 2.0, 3.0, 4.0};
            try (Tensor t = Tensor.fromDoubleArray(data, 2, 2)) {
                assertEquals(4, t.elementCount());
                assertEquals(ScalarType.F64, t.dtype());
            }
        }
    }

    @Nested
    @DisplayName("fromIntArray - Data Length vs Shape Mismatch")
    class FromIntArrayMismatch {

        @Test
        @DisplayName("Throws when data is too short for shape")
        void dataTooShort() {
            int[] data = new int[]{1, 2, 3}; // 3 elements
            assertThrows(IllegalArgumentException.class,
                () -> Tensor.fromIntArray(data, 4)); // expects 4 elements
        }

        @Test
        @DisplayName("Throws when data is too long for shape")
        void dataTooLong() {
            int[] data = new int[]{1, 2, 3, 4, 5}; // 5 elements
            assertThrows(IllegalArgumentException.class,
                () -> Tensor.fromIntArray(data, 4)); // expects 4 elements
        }

        @Test
        @DisplayName("Succeeds when data length exactly matches shape")
        void exactMatch() {
            int[] data = new int[]{1, 2, 3, 4};
            try (Tensor t = Tensor.fromIntArray(data, 4)) {
                assertEquals(4, t.elementCount());
                assertEquals(ScalarType.I32, t.dtype());
            }
        }
    }

    @Nested
    @DisplayName("fromMemorySegment - Segment Size Validation")
    class FromMemorySegmentValidation {

        @Test
        @DisplayName("Throws when segment is too small for spec")
        void segmentTooSmall() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 10); // needs 40 bytes
            try (var arena = java.lang.foreign.Arena.ofConfined()) {
                var segment = arena.allocate(20); // only 20 bytes
                var ex = assertThrows(IllegalArgumentException.class,
                    () -> Tensor.fromMemorySegment(segment, spec));
                assertTrue(ex.getMessage().contains("20"),
                    "Error message should include actual segment size");
                assertTrue(ex.getMessage().contains("40"),
                    "Error message should include required size");
            }
        }

        @Test
        @DisplayName("Succeeds when segment is exactly the right size")
        void segmentExactSize() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 10); // needs 40 bytes
            try (var arena = java.lang.foreign.Arena.ofConfined()) {
                var segment = arena.allocate(40);
                assertDoesNotThrow(() -> Tensor.fromMemorySegment(segment, spec));
            }
        }

        @Test
        @DisplayName("Succeeds when segment is larger than needed")
        void segmentLarger() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 10); // needs 40 bytes
            try (var arena = java.lang.foreign.Arena.ofConfined()) {
                var segment = arena.allocate(100); // extra space
                assertDoesNotThrow(() -> Tensor.fromMemorySegment(segment, spec));
            }
        }
    }

    @Nested
    @DisplayName("reshape - Element Count Mismatch")
    class ReshapeMismatch {

        @Test
        @DisplayName("Throws when new shape has fewer elements")
        void newShapeFewerElements() {
            try (Tensor t = Tensor.zeros(2, 3)) { // 6 elements
                var ex = assertThrows(IllegalArgumentException.class,
                    () -> t.reshape(2, 2)); // 4 elements
                assertTrue(ex.getMessage().contains("6"),
                    "Error message should include original element count");
                assertTrue(ex.getMessage().contains("4"),
                    "Error message should include new element count");
            }
        }

        @Test
        @DisplayName("Throws when new shape has more elements")
        void newShapeMoreElements() {
            try (Tensor t = Tensor.zeros(2, 3)) { // 6 elements
                assertThrows(IllegalArgumentException.class,
                    () -> t.reshape(3, 3)); // 9 elements
            }
        }

        @Test
        @DisplayName("Succeeds when element counts match")
        void elementCountsMatch() {
            try (Tensor t = Tensor.zeros(2, 3)) { // 6 elements
                Tensor reshaped = t.reshape(3, 2); // still 6 elements
                assertEquals(6, reshaped.elementCount());
                assertArrayEquals(new int[]{3, 2}, reshaped.shape());
            }
        }

        @Test
        @DisplayName("Reshape to 1D succeeds")
        void reshapeTo1D() {
            try (Tensor t = Tensor.zeros(2, 3, 4)) { // 24 elements
                Tensor flat = t.reshape(24);
                assertEquals(1, flat.rank());
                assertEquals(24, flat.elementCount());
            }
        }

        @Test
        @DisplayName("Reshape from 1D to multi-dimensional succeeds")
        void reshapeFrom1D() {
            try (Tensor t = Tensor.zeros(24)) {
                Tensor shaped = t.reshape(2, 3, 4);
                assertEquals(3, shaped.rank());
                assertArrayEquals(new int[]{2, 3, 4}, shaped.shape());
            }
        }

        @Test
        @DisplayName("Reshape scalar to [1] succeeds")
        void reshapeScalarTo1() {
            // Create a scalar-like tensor with 1 element
            try (Tensor t = Tensor.zeros(1)) {
                Tensor reshaped = t.reshape(1);
                assertEquals(1, reshaped.elementCount());
            }
        }

        @Test
        @DisplayName("Reshape preserves data")
        void reshapePreservesData() {
            float[] data = {1, 2, 3, 4, 5, 6};
            try (Tensor t = Tensor.fromFloatArray(data, 2, 3)) {
                Tensor reshaped = t.reshape(3, 2);
                // Data should be same when flattened
                assertArrayEquals(data, reshaped.toFloatArray(), 1e-6f);
            }
        }

        @Test
        @DisplayName("Multiple reshapes maintain element count constraint")
        void multipleReshapes() {
            try (Tensor t = Tensor.zeros(24)) {
                Tensor r1 = t.reshape(2, 12);
                Tensor r2 = r1.reshape(3, 8);
                Tensor r3 = r2.reshape(4, 6);
                Tensor r4 = r3.reshape(24);
                assertEquals(24, r4.elementCount());
            }
        }
    }

    @Nested
    @DisplayName("zeros Factory")
    class ZerosFactory {

        @Test
        @DisplayName("Creates tensor with correct shape")
        void correctShape() {
            try (Tensor t = Tensor.zeros(2, 3, 4)) {
                assertArrayEquals(new int[]{2, 3, 4}, t.shape());
                assertEquals(24, t.elementCount());
            }
        }

        @Test
        @DisplayName("Creates tensor with all zeros")
        void allZeros() {
            try (Tensor t = Tensor.zeros(3, 3)) {
                float[] data = t.toFloatArray();
                for (float v : data) {
                    assertEquals(0.0f, v, 1e-9f);
                }
            }
        }

        @Test
        @DisplayName("Creates scalar tensor")
        void scalarTensor() {
            try (Tensor t = Tensor.zeros()) {
                assertEquals(0, t.rank());
                assertEquals(1, t.elementCount());
            }
        }

        @Test
        @DisplayName("Creates 1D tensor")
        void oneDimensional() {
            try (Tensor t = Tensor.zeros(10)) {
                assertEquals(1, t.rank());
                assertEquals(10, t.elementCount());
            }
        }

        @Test
        @DisplayName("Creates high-dimensional tensor")
        void highDimensional() {
            try (Tensor t = Tensor.zeros(2, 2, 2, 2, 2)) {
                assertEquals(5, t.rank());
                assertEquals(32, t.elementCount());
            }
        }

        @Test
        @DisplayName("Respects dtype parameter")
        void respectsDtype() {
            try (Tensor f64 = Tensor.zeros(ScalarType.F64, 4)) {
                assertEquals(ScalarType.F64, f64.dtype());
                assertEquals(32, f64.spec().byteSize()); // 4 * 8 bytes
            }
        }
    }

    @Nested
    @DisplayName("full Factory")
    class FullFactory {

        @Test
        @DisplayName("Creates tensor filled with value")
        void filledWithValue() {
            try (Tensor t = Tensor.full(3.14f, 2, 3)) {
                float[] data = t.toFloatArray();
                for (float v : data) {
                    assertEquals(3.14f, v, 1e-6f);
                }
            }
        }

        @Test
        @DisplayName("Creates tensor with correct shape")
        void correctShape() {
            try (Tensor t = Tensor.full(1.0f, 4, 5, 6)) {
                assertArrayEquals(new int[]{4, 5, 6}, t.shape());
            }
        }

        @Test
        @DisplayName("Handles negative fill value")
        void negativeFillValue() {
            try (Tensor t = Tensor.full(-42.0f, 3)) {
                assertEquals(-42.0f, t.getFloat(0), 1e-6f);
                assertEquals(-42.0f, t.getFloat(2), 1e-6f);
            }
        }

        @Test
        @DisplayName("Handles special float values")
        void specialFloatValues() {
            try (Tensor inf = Tensor.full(Float.POSITIVE_INFINITY, 2)) {
                assertEquals(Float.POSITIVE_INFINITY, inf.getFloat(0));
            }
            try (Tensor nan = Tensor.full(Float.NaN, 2)) {
                assertTrue(Float.isNaN(nan.getFloat(0)));
            }
        }
    }

    @Nested
    @DisplayName("copyFrom/copyTo - Size Validation")
    class CopyValidation {

        @Test
        @DisplayName("copyFrom throws when source is too short")
        void copyFromTooShort() {
            try (Tensor t = Tensor.zeros(2, 3)) { // 6 elements
                float[] source = new float[]{1, 2, 3}; // only 3 elements
                var ex = assertThrows(IllegalArgumentException.class,
                    () -> t.copyFrom(source));
                assertTrue(ex.getMessage().contains("3"),
                    "Error message should include source length");
                assertTrue(ex.getMessage().contains("6"),
                    "Error message should include tensor size");
            }
        }

        @Test
        @DisplayName("copyFrom throws when source is too long")
        void copyFromTooLong() {
            try (Tensor t = Tensor.zeros(2, 3)) { // 6 elements
                float[] source = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; // 10 elements
                assertThrows(IllegalArgumentException.class,
                    () -> t.copyFrom(source));
            }
        }

        @Test
        @DisplayName("copyTo throws when destination is too short")
        void copyToTooShort() {
            try (Tensor t = Tensor.zeros(2, 3)) { // 6 elements
                float[] dest = new float[3]; // only 3 slots
                assertThrows(IllegalArgumentException.class,
                    () -> t.copyTo(dest));
            }
        }

        @Test
        @DisplayName("copyTo throws when destination is too long")
        void copyToTooLong() {
            try (Tensor t = Tensor.zeros(2, 3)) { // 6 elements
                float[] dest = new float[10]; // 10 slots
                assertThrows(IllegalArgumentException.class,
                    () -> t.copyTo(dest));
            }
        }

        @Test
        @DisplayName("copyFrom succeeds with exact size match")
        void copyFromExactMatch() {
            try (Tensor t = Tensor.zeros(2, 3)) {
                float[] source = new float[]{1, 2, 3, 4, 5, 6};
                assertDoesNotThrow(() -> t.copyFrom(source));
                assertArrayEquals(source, t.toFloatArray(), 1e-6f);
            }
        }

        @Test
        @DisplayName("copyTo succeeds with exact size match")
        void copyToExactMatch() {
            float[] data = {1, 2, 3, 4, 5, 6};
            try (Tensor t = Tensor.fromFloatArray(data, 2, 3)) {
                float[] dest = new float[6];
                assertDoesNotThrow(() -> t.copyTo(dest));
                assertArrayEquals(data, dest, 1e-6f);
            }
        }
    }

    @Nested
    @DisplayName("allocate Factory")
    class AllocateFactory {

        @Test
        @DisplayName("Creates tensor with correct spec")
        void correctSpec() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            try (var arena = java.lang.foreign.Arena.ofConfined()) {
                Tensor t = Tensor.allocate(spec, arena);
                assertEquals(2, t.rank());
                assertArrayEquals(new int[]{2, 3}, t.shape());
                assertEquals(ScalarType.F32, t.dtype());
            }
        }

        @Test
        @DisplayName("Tensor is zero-initialized")
        void zeroInitialized() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 4);
            try (var arena = java.lang.foreign.Arena.ofConfined()) {
                Tensor t = Tensor.allocate(spec, arena);
                for (int i = 0; i < 4; i++) {
                    assertEquals(0.0f, t.getFloat(i), 1e-9f);
                }
            }
        }
    }
}
