package io.surfworks.warpforge.core.tensor;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Comprehensive tests for TensorSpec shape validation.
 * These tests document expected behavior for edge cases and serve as a
 * specification for future compile-time type safety.
 */
@DisplayName("TensorSpec Shape Validation")
class TensorSpecShapeTest {

    @Nested
    @DisplayName("Index Validation")
    class IndexValidation {

        @Test
        @DisplayName("flatIndex throws IndexOutOfBoundsException for negative index in first dimension")
        void negativeIndexFirstDimension() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            var ex = assertThrows(IndexOutOfBoundsException.class,
                () -> spec.flatIndex(-1, 0));
            assertTrue(ex.getMessage().contains("dimension 0"),
                "Error message should indicate which dimension failed");
            assertTrue(ex.getMessage().contains("-1"),
                "Error message should include the invalid index");
        }

        @Test
        @DisplayName("flatIndex throws IndexOutOfBoundsException for negative index in last dimension")
        void negativeIndexLastDimension() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            var ex = assertThrows(IndexOutOfBoundsException.class,
                () -> spec.flatIndex(0, -1));
            assertTrue(ex.getMessage().contains("dimension 1"),
                "Error message should indicate which dimension failed");
        }

        @Test
        @DisplayName("flatIndex throws IndexOutOfBoundsException for index equal to dimension size")
        void indexEqualsToDimensionSize() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            // Index 2 is out of bounds for dimension with size 2
            var ex = assertThrows(IndexOutOfBoundsException.class,
                () -> spec.flatIndex(2, 0));
            assertTrue(ex.getMessage().contains("size 2"),
                "Error message should include the dimension size");
        }

        @Test
        @DisplayName("flatIndex throws IndexOutOfBoundsException for index greater than dimension size")
        void indexGreaterThanDimensionSize() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            assertThrows(IndexOutOfBoundsException.class,
                () -> spec.flatIndex(0, 100));
        }

        @Test
        @DisplayName("flatIndex throws IllegalArgumentException for wrong number of indices (too few)")
        void tooFewIndices() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            var ex = assertThrows(IllegalArgumentException.class,
                () -> spec.flatIndex(0, 0));
            assertTrue(ex.getMessage().contains("Expected 3"),
                "Error message should indicate expected count");
            assertTrue(ex.getMessage().contains("got 2"),
                "Error message should indicate actual count");
        }

        @Test
        @DisplayName("flatIndex throws IllegalArgumentException for wrong number of indices (too many)")
        void tooManyIndices() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            var ex = assertThrows(IllegalArgumentException.class,
                () -> spec.flatIndex(0, 0, 0, 0));
            assertTrue(ex.getMessage().contains("Expected 2"),
                "Error message should indicate expected count");
            assertTrue(ex.getMessage().contains("got 4"),
                "Error message should indicate actual count");
        }

        @Test
        @DisplayName("flatIndex accepts zero indices for scalar tensor")
        void scalarTensorZeroIndices() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32); // Scalar
            assertEquals(0, spec.flatIndex());
        }

        @Test
        @DisplayName("flatIndex validates each dimension independently")
        void eachDimensionValidatedIndependently() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 10, 5, 3);
            // Valid: all within bounds
            assertDoesNotThrow(() -> spec.flatIndex(9, 4, 2));
            // Invalid: first dimension out of bounds
            assertThrows(IndexOutOfBoundsException.class, () -> spec.flatIndex(10, 0, 0));
            // Invalid: middle dimension out of bounds
            assertThrows(IndexOutOfBoundsException.class, () -> spec.flatIndex(0, 5, 0));
            // Invalid: last dimension out of bounds
            assertThrows(IndexOutOfBoundsException.class, () -> spec.flatIndex(0, 0, 3));
        }

        @Test
        @DisplayName("flatIndex works with int[] overload")
        void intArrayOverload() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            assertEquals(spec.flatIndex(1L, 2L), spec.flatIndex(1, 2));
            assertThrows(IndexOutOfBoundsException.class, () -> spec.flatIndex(new int[]{-1, 0}));
        }
    }

    @Nested
    @DisplayName("Empty and Scalar Shapes")
    class EmptyAndScalarShapes {

        @Test
        @DisplayName("Empty shape array creates scalar tensor")
        void emptyShapeCreatesScalar() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32);
            assertEquals(0, spec.rank());
            assertEquals(1, spec.elementCount());
            assertArrayEquals(new int[]{}, spec.shape());
            assertArrayEquals(new long[]{}, spec.strides());
        }

        @Test
        @DisplayName("Scalar tensor has correct byte size")
        void scalarByteSize() {
            TensorSpec f32Scalar = TensorSpec.of(ScalarType.F32);
            assertEquals(4, f32Scalar.byteSize());

            TensorSpec f64Scalar = TensorSpec.of(ScalarType.F64);
            assertEquals(8, f64Scalar.byteSize());
        }

        @Test
        @DisplayName("Scalar tensor is always contiguous")
        void scalarIsContiguous() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32);
            assertTrue(spec.isContiguous());
        }
    }

    @Nested
    @DisplayName("Zero-Sized Dimensions")
    class ZeroSizedDimensions {

        @Test
        @DisplayName("Tensor with zero-sized dimension has zero elements")
        void zeroSizedDimensionZeroElements() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 0, 3);
            assertEquals(3, spec.rank());
            assertEquals(0, spec.elementCount());
        }

        @Test
        @DisplayName("Zero-sized tensor has zero byte size")
        void zeroSizedTensorZeroBytes() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 0);
            assertEquals(0, spec.byteSize());
        }

        @Test
        @DisplayName("Multiple zero dimensions still result in zero elements")
        void multipleZeroDimensions() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 0, 0, 0);
            assertEquals(0, spec.elementCount());
        }

        @Test
        @DisplayName("Zero dimension mixed with non-zero dimensions")
        void mixedZeroNonZeroDimensions() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 5, 0, 10);
            assertEquals(0, spec.elementCount());
            assertEquals(0, spec.byteSize());
        }
    }

    @Nested
    @DisplayName("Stride Calculations")
    class StrideCalculations {

        @Test
        @DisplayName("Row-major strides: last dimension has stride 1")
        void rowMajorLastDimensionStride1() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            long[] strides = spec.strides();
            assertEquals(1, strides[2], "Last dimension should have stride 1");
        }

        @Test
        @DisplayName("Row-major strides for 1D tensor")
        void rowMajor1D() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 10);
            assertArrayEquals(new long[]{1}, spec.strides());
        }

        @Test
        @DisplayName("Row-major strides for 2D tensor")
        void rowMajor2D() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 3, 4);
            assertArrayEquals(new long[]{4, 1}, spec.strides());
        }

        @Test
        @DisplayName("Row-major strides for 3D tensor")
        void rowMajor3D() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            assertArrayEquals(new long[]{12, 4, 1}, spec.strides());
        }

        @Test
        @DisplayName("Row-major strides for 4D tensor")
        void rowMajor4D() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3, 4, 5);
            assertArrayEquals(new long[]{60, 20, 5, 1}, spec.strides());
        }

        @Test
        @DisplayName("Column-major strides: first dimension has stride 1")
        void columnMajorFirstDimensionStride1() {
            long[] strides = TensorSpec.computeColumnMajorStrides(new int[]{2, 3, 4});
            assertEquals(1, strides[0], "First dimension should have stride 1");
        }

        @Test
        @DisplayName("Column-major strides for 2D tensor")
        void columnMajor2D() {
            long[] strides = TensorSpec.computeColumnMajorStrides(new int[]{3, 4});
            assertArrayEquals(new long[]{1, 3}, strides);
        }

        @Test
        @DisplayName("Column-major strides for 3D tensor")
        void columnMajor3D() {
            long[] strides = TensorSpec.computeColumnMajorStrides(new int[]{2, 3, 4});
            assertArrayEquals(new long[]{1, 2, 6}, strides);
        }

        @Test
        @DisplayName("Strides with zero dimension")
        void stridesWithZeroDimension() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 0, 4);
            // Strides should still be computed (even if tensor has no elements)
            assertEquals(3, spec.strides().length);
        }

        @Test
        @DisplayName("withStrides validates shape/strides length match")
        void withStridesValidatesLength() {
            assertThrows(IllegalArgumentException.class,
                () -> TensorSpec.withStrides(ScalarType.F32, new int[]{2, 3}, new long[]{1}));
        }

        @Test
        @DisplayName("Custom strides are preserved")
        void customStridesPreserved() {
            long[] customStrides = {100, 10, 1};
            TensorSpec spec = TensorSpec.withStrides(ScalarType.F32, new int[]{2, 3, 4}, customStrides);
            assertArrayEquals(customStrides, spec.strides());
        }
    }

    @Nested
    @DisplayName("Contiguity Detection")
    class ContiguityDetection {

        @Test
        @DisplayName("Row-major tensor is contiguous")
        void rowMajorIsContiguous() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            assertTrue(spec.isContiguous());
        }

        @Test
        @DisplayName("Custom non-contiguous strides detected")
        void nonContiguousStridesDetected() {
            // Strides that skip elements
            TensorSpec spec = TensorSpec.withStrides(ScalarType.F32, new int[]{2, 3}, new long[]{6, 2});
            assertFalse(spec.isContiguous());
        }

        @Test
        @DisplayName("Column-major tensor is not row-contiguous")
        void columnMajorNotRowContiguous() {
            long[] colMajorStrides = TensorSpec.computeColumnMajorStrides(new int[]{2, 3});
            TensorSpec spec = TensorSpec.withStrides(ScalarType.F32, new int[]{2, 3}, colMajorStrides);
            assertFalse(spec.isContiguous());
        }
    }

    @Nested
    @DisplayName("Shape Equality")
    class ShapeEquality {

        @Test
        @DisplayName("Same shapes are equal")
        void sameShapesEqual() {
            TensorSpec a = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            TensorSpec b = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            assertTrue(a.shapeEquals(b));
        }

        @Test
        @DisplayName("Different shapes are not equal")
        void differentShapesNotEqual() {
            TensorSpec a = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            TensorSpec b = TensorSpec.of(ScalarType.F32, 2, 4, 3);
            assertFalse(a.shapeEquals(b));
        }

        @Test
        @DisplayName("Different ranks are not equal")
        void differentRanksNotEqual() {
            TensorSpec a = TensorSpec.of(ScalarType.F32, 2, 3);
            TensorSpec b = TensorSpec.of(ScalarType.F32, 2, 3, 1);
            assertFalse(a.shapeEquals(b));
        }

        @Test
        @DisplayName("Shape equality ignores dtype")
        void shapeEqualityIgnoresDtype() {
            TensorSpec a = TensorSpec.of(ScalarType.F32, 2, 3);
            TensorSpec b = TensorSpec.of(ScalarType.F64, 2, 3);
            assertTrue(a.shapeEquals(b));
        }

        @Test
        @DisplayName("Shape equality ignores strides")
        void shapeEqualityIgnoresStrides() {
            TensorSpec a = TensorSpec.of(ScalarType.F32, 2, 3);
            TensorSpec b = TensorSpec.withStrides(ScalarType.F32, new int[]{2, 3}, new long[]{10, 1});
            assertTrue(a.shapeEquals(b));
        }

        @Test
        @DisplayName("Scalar shapes are equal")
        void scalarShapesEqual() {
            TensorSpec a = TensorSpec.of(ScalarType.F32);
            TensorSpec b = TensorSpec.of(ScalarType.F64);
            assertTrue(a.shapeEquals(b));
        }
    }

    @Nested
    @DisplayName("Element Count")
    class ElementCount {

        @Test
        @DisplayName("1D tensor element count")
        void elementCount1D() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 10);
            assertEquals(10, spec.elementCount());
        }

        @Test
        @DisplayName("2D tensor element count is product of dimensions")
        void elementCount2D() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 3, 4);
            assertEquals(12, spec.elementCount());
        }

        @Test
        @DisplayName("3D tensor element count")
        void elementCount3D() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            assertEquals(24, spec.elementCount());
        }

        @Test
        @DisplayName("Large tensor element count")
        void elementCountLarge() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 100, 100, 100);
            assertEquals(1_000_000, spec.elementCount());
        }
    }

    @Nested
    @DisplayName("Flat Index Calculation")
    class FlatIndexCalculation {

        @Test
        @DisplayName("Row-major flat index: first element")
        void flatIndexFirstElement() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            assertEquals(0, spec.flatIndex(0, 0, 0));
        }

        @Test
        @DisplayName("Row-major flat index: last element")
        void flatIndexLastElement() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3, 4);
            assertEquals(23, spec.flatIndex(1, 2, 3));
        }

        @Test
        @DisplayName("Row-major flat index: sequential in last dimension")
        void flatIndexSequentialLastDim() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            assertEquals(0, spec.flatIndex(0, 0));
            assertEquals(1, spec.flatIndex(0, 1));
            assertEquals(2, spec.flatIndex(0, 2));
            assertEquals(3, spec.flatIndex(1, 0));
        }

        @Test
        @DisplayName("Custom strides affect flat index")
        void flatIndexWithCustomStrides() {
            // Every-other-element strides
            TensorSpec spec = TensorSpec.withStrides(ScalarType.F32, new int[]{2, 3}, new long[]{6, 2});
            assertEquals(0, spec.flatIndex(0, 0));
            assertEquals(2, spec.flatIndex(0, 1));
            assertEquals(4, spec.flatIndex(0, 2));
            assertEquals(6, spec.flatIndex(1, 0));
        }
    }

    @Nested
    @DisplayName("Byte Size Calculation")
    class ByteSizeCalculation {

        @Test
        @DisplayName("F32 tensor byte size")
        void f32ByteSize() {
            TensorSpec spec = TensorSpec.of(ScalarType.F32, 2, 3);
            assertEquals(6 * 4, spec.byteSize());
        }

        @Test
        @DisplayName("F64 tensor byte size")
        void f64ByteSize() {
            TensorSpec spec = TensorSpec.of(ScalarType.F64, 2, 3);
            assertEquals(6 * 8, spec.byteSize());
        }

        @Test
        @DisplayName("I32 tensor byte size")
        void i32ByteSize() {
            TensorSpec spec = TensorSpec.of(ScalarType.I32, 2, 3);
            assertEquals(6 * 4, spec.byteSize());
        }

        @Test
        @DisplayName("F16 tensor byte size")
        void f16ByteSize() {
            TensorSpec spec = TensorSpec.of(ScalarType.F16, 2, 3);
            assertEquals(6 * 2, spec.byteSize());
        }
    }
}
