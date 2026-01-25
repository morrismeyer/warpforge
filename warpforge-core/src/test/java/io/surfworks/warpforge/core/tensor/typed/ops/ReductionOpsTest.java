package io.surfworks.warpforge.core.tensor.typed.ops;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Scalar;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Tests for ReductionOps reduction operations.
 */
@DisplayName("ReductionOps")
class ReductionOpsTest {

    private static final float EPSILON = 1e-5f;
    private static final double EPSILON_D = 1e-10;

    @Nested
    @DisplayName("Full Reductions")
    class FullReductions {

        @Test
        @DisplayName("sum computes total of all elements")
        void sumComputesTotal() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5}, new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                float result = ReductionOps.sum(t);
                assertEquals(15.0f, result, EPSILON);
            }
        }

        @Test
        @DisplayName("sum works with matrices")
        void sumWorksWithMatrices() {
            try (TypedTensor<Matrix, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                float result = ReductionOps.sum(t);
                assertEquals(21.0f, result, EPSILON);
            }
        }

        @Test
        @DisplayName("sumF64 works with F64")
        void sumF64WorksWithF64() {
            try (TypedTensor<Vector, F64, Cpu> t = TypedTensor.fromDoubleArray(
                    new double[]{1.5, 2.5, 3.5}, new Vector(3), F64.INSTANCE, Cpu.INSTANCE)) {

                double result = ReductionOps.sumF64(t);
                assertEquals(7.5, result, EPSILON_D);
            }
        }

        @Test
        @DisplayName("mean computes average")
        void meanComputesAverage() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{2, 4, 6, 8, 10}, new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                float result = ReductionOps.mean(t);
                assertEquals(6.0f, result, EPSILON);
            }
        }

        @Test
        @DisplayName("max finds maximum element")
        void maxFindsMaximum() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{3, 1, 4, 1, 5, 9, 2, 6}, new Vector(8), F32.INSTANCE, Cpu.INSTANCE)) {

                float result = ReductionOps.max(t);
                assertEquals(9.0f, result, EPSILON);
            }
        }

        @Test
        @DisplayName("min finds minimum element")
        void minFindsMinimum() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{3, 1, 4, 1, 5, 9, 2, 6}, new Vector(8), F32.INSTANCE, Cpu.INSTANCE)) {

                float result = ReductionOps.min(t);
                assertEquals(1.0f, result, EPSILON);
            }
        }

        @Test
        @DisplayName("max handles negative numbers")
        void maxHandlesNegativeNumbers() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{-5, -2, -8, -1, -10}, new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                float result = ReductionOps.max(t);
                assertEquals(-1.0f, result, EPSILON);
            }
        }

        @Test
        @DisplayName("max rejects empty tensor")
        void maxRejectsEmptyTensor() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(0), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> ReductionOps.max(t));
            }
        }

        @Test
        @DisplayName("prod computes product")
        void prodComputesProduct() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE)) {

                float result = ReductionOps.prod(t);
                assertEquals(24.0f, result, EPSILON);
            }
        }

        @Test
        @DisplayName("variance computes variance")
        void varianceComputesVariance() {
            // Values: 2, 4, 4, 4, 5, 5, 7, 9
            // Mean = 5
            // Variance = ((2-5)^2 + (4-5)^2 + ... + (9-5)^2) / 8 = 4
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{2, 4, 4, 4, 5, 5, 7, 9}, new Vector(8), F32.INSTANCE, Cpu.INSTANCE)) {

                float result = ReductionOps.variance(t);
                assertEquals(4.0f, result, EPSILON);
            }
        }

        @Test
        @DisplayName("std computes standard deviation")
        void stdComputesStandardDeviation() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{2, 4, 4, 4, 5, 5, 7, 9}, new Vector(8), F32.INSTANCE, Cpu.INSTANCE)) {

                float result = ReductionOps.std(t);
                assertEquals(2.0f, result, EPSILON);
            }
        }

        @Test
        @DisplayName("norm computes L2 norm")
        void normComputesL2Norm() {
            // ||[3, 4]|| = sqrt(9 + 16) = 5
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{3, 4}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE)) {

                float result = ReductionOps.norm(t);
                assertEquals(5.0f, result, EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Index Reductions")
    class IndexReductions {

        @Test
        @DisplayName("argmax finds index of maximum")
        void argmaxFindsIndexOfMaximum() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{3, 1, 4, 1, 5, 9, 2, 6}, new Vector(8), F32.INSTANCE, Cpu.INSTANCE)) {

                long result = ReductionOps.argmax(t);
                assertEquals(5, result);  // Index of 9
            }
        }

        @Test
        @DisplayName("argmin finds index of minimum")
        void argminFindsIndexOfMinimum() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{3, 1, 4, 1, 5, 9, 2, 6}, new Vector(8), F32.INSTANCE, Cpu.INSTANCE)) {

                long result = ReductionOps.argmin(t);
                assertEquals(1, result);  // First index of 1
            }
        }

        @Test
        @DisplayName("argmax returns first occurrence on tie")
        void argmaxReturnsFirstOnTie() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{5, 5, 5}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                long result = ReductionOps.argmax(t);
                assertEquals(0, result);
            }
        }
    }

    @Nested
    @DisplayName("Axis Reductions")
    class AxisReductions {

        @Test
        @DisplayName("sumAxis reduces along rows (axis=0)")
        void sumAxisReducesAlongRows() {
            // Matrix: [[1, 2, 3], [4, 5, 6]]
            // Sum along axis 0 (collapse rows): [5, 7, 9]
            try (TypedTensor<Matrix, F32, Cpu> m = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = ReductionOps.sumAxis(m, 0)) {

                assertEquals(3, result.shapeType().length());
                assertArrayEquals(new float[]{5, 7, 9}, result.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("sumAxis reduces along columns (axis=1)")
        void sumAxisReducesAlongColumns() {
            // Matrix: [[1, 2, 3], [4, 5, 6]]
            // Sum along axis 1 (collapse columns): [6, 15]
            try (TypedTensor<Matrix, F32, Cpu> m = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = ReductionOps.sumAxis(m, 1)) {

                assertEquals(2, result.shapeType().length());
                assertArrayEquals(new float[]{6, 15}, result.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("meanAxis computes mean along axis")
        void meanAxisComputesMean() {
            // Matrix: [[1, 2, 3], [4, 5, 6]]
            // Mean along axis 0: [2.5, 3.5, 4.5]
            try (TypedTensor<Matrix, F32, Cpu> m = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = ReductionOps.meanAxis(m, 0)) {

                assertArrayEquals(new float[]{2.5f, 3.5f, 4.5f}, result.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("maxAxis finds max along axis")
        void maxAxisFindsMax() {
            // Matrix: [[1, 5, 3], [4, 2, 6]]
            // Max along axis 0: [4, 5, 6]
            try (TypedTensor<Matrix, F32, Cpu> m = TypedTensor.fromFloatArray(
                    new float[]{1, 5, 3, 4, 2, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = ReductionOps.maxAxis(m, 0)) {

                assertArrayEquals(new float[]{4, 5, 6}, result.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("minAxis finds min along axis")
        void minAxisFindsMin() {
            // Matrix: [[1, 5, 3], [4, 2, 6]]
            // Min along axis 1: [1, 2]
            try (TypedTensor<Matrix, F32, Cpu> m = TypedTensor.fromFloatArray(
                    new float[]{1, 5, 3, 4, 2, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = ReductionOps.minAxis(m, 1)) {

                assertArrayEquals(new float[]{1, 2}, result.underlying().toFloatArray(), EPSILON);
            }
        }

        @Test
        @DisplayName("sumAxis rejects invalid axis")
        void sumAxisRejectsInvalidAxis() {
            try (TypedTensor<Matrix, F32, Cpu> m = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> ReductionOps.sumAxis(m, 2));
            }
        }
    }

    @Nested
    @DisplayName("Typed Scalar Reductions")
    class TypedScalarReductions {

        @Test
        @DisplayName("sumToScalar returns scalar tensor")
        void sumToScalarReturnsScalarTensor() {
            try (TypedTensor<Vector, F32, Cpu> v = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Scalar, F32, Cpu> result = ReductionOps.sumToScalar(v)) {

                assertEquals(0, result.rank());
                assertEquals(1, result.elementCount());
                assertEquals(6.0f, result.underlying().getFloatFlat(0), EPSILON);
            }
        }

        @Test
        @DisplayName("meanToScalar returns scalar tensor")
        void meanToScalarReturnsScalarTensor() {
            try (TypedTensor<Vector, F32, Cpu> v = TypedTensor.fromFloatArray(
                    new float[]{2, 4, 6}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Scalar, F32, Cpu> result = ReductionOps.meanToScalar(v)) {

                assertEquals(4.0f, result.underlying().getFloatFlat(0), EPSILON);
            }
        }

        @Test
        @DisplayName("maxToScalar returns scalar tensor")
        void maxToScalarReturnsScalarTensor() {
            try (TypedTensor<Vector, F32, Cpu> v = TypedTensor.fromFloatArray(
                    new float[]{3, 7, 2}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Scalar, F32, Cpu> result = ReductionOps.maxToScalar(v)) {

                assertEquals(7.0f, result.underlying().getFloatFlat(0), EPSILON);
            }
        }

        @Test
        @DisplayName("minToScalar returns scalar tensor")
        void minToScalarReturnsScalarTensor() {
            try (TypedTensor<Vector, F32, Cpu> v = TypedTensor.fromFloatArray(
                    new float[]{3, 7, 2}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Scalar, F32, Cpu> result = ReductionOps.minToScalar(v)) {

                assertEquals(2.0f, result.underlying().getFloatFlat(0), EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Type Safety")
    class TypeSafety {

        @Test
        @DisplayName("axis reduction returns correct shape type")
        void axisReductionReturnsCorrectShapeType() {
            try (TypedTensor<Matrix, F32, Cpu> m = TypedTensor.zeros(
                    new Matrix(10, 20), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> v = ReductionOps.sumAxis(m, 1)) {

                // Return type is TypedTensor<Vector, F32, Cpu> - verified by compilation
                assertEquals(10, v.shapeType().length());
            }
        }

        @Test
        @DisplayName("scalar reduction returns correct shape type")
        void scalarReductionReturnsCorrectShapeType() {
            try (TypedTensor<Matrix, F32, Cpu> m = TypedTensor.zeros(
                    new Matrix(5, 5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Scalar, F32, Cpu> s = ReductionOps.sumToScalar(m)) {

                // Return type is TypedTensor<Scalar, F32, Cpu> - verified by compilation
                assertEquals(Scalar.INSTANCE, s.shapeType());
            }
        }
    }
}
