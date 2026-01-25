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
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Tests for TypedOps elementwise operations.
 */
@DisplayName("TypedOps")
class TypedOpsTest {

    private static final float EPSILON = 1e-6f;
    private static final double EPSILON_D = 1e-10;

    @Nested
    @DisplayName("Binary Elementwise Operations")
    class BinaryOps {

        @Test
        @DisplayName("add performs elementwise addition")
        void addPerformsElementwiseAddition() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> b = TypedTensor.fromFloatArray(
                    new float[]{10, 20, 30, 40}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> c = TypedOps.add(a, b)) {

                float[] result = c.underlying().toFloatArray();
                assertArrayEquals(new float[]{11, 22, 33, 44}, result, EPSILON);
            }
        }

        @Test
        @DisplayName("sub performs elementwise subtraction")
        void subPerformsElementwiseSubtraction() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{10, 20, 30, 40}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> b = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> c = TypedOps.sub(a, b)) {

                float[] result = c.underlying().toFloatArray();
                assertArrayEquals(new float[]{9, 18, 27, 36}, result, EPSILON);
            }
        }

        @Test
        @DisplayName("mul performs elementwise multiplication")
        void mulPerformsElementwiseMultiplication() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> b = TypedTensor.fromFloatArray(
                    new float[]{2, 3, 4, 5}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> c = TypedOps.mul(a, b)) {

                float[] result = c.underlying().toFloatArray();
                assertArrayEquals(new float[]{2, 6, 12, 20}, result, EPSILON);
            }
        }

        @Test
        @DisplayName("div performs elementwise division")
        void divPerformsElementwiseDivision() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{10, 20, 30, 40}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> b = TypedTensor.fromFloatArray(
                    new float[]{2, 4, 5, 8}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> c = TypedOps.div(a, b)) {

                float[] result = c.underlying().toFloatArray();
                assertArrayEquals(new float[]{5, 5, 6, 5}, result, EPSILON);
            }
        }

        @Test
        @DisplayName("binary ops reject shape mismatch")
        void binaryOpsRejectShapeMismatch() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> b = TypedTensor.zeros(
                    new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class, () -> TypedOps.add(a, b));
            }
        }

        @Test
        @DisplayName("add works with F64")
        void addWorksWithF64() {
            try (TypedTensor<Vector, F64, Cpu> a = TypedTensor.fromDoubleArray(
                    new double[]{1.0, 2.0, 3.0}, new Vector(3), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F64, Cpu> b = TypedTensor.fromDoubleArray(
                    new double[]{0.1, 0.2, 0.3}, new Vector(3), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F64, Cpu> c = TypedOps.add(a, b)) {

                double[] result = c.underlying().toDoubleArray();
                assertArrayEquals(new double[]{1.1, 2.2, 3.3}, result, EPSILON_D);
            }
        }

        @Test
        @DisplayName("operations work with matrices")
        void opsWorkWithMatrices() {
            try (TypedTensor<Matrix, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> b = TypedTensor.fromFloatArray(
                    new float[]{6, 5, 4, 3, 2, 1}, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> c = TypedOps.add(a, b)) {

                float[] result = c.underlying().toFloatArray();
                assertArrayEquals(new float[]{7, 7, 7, 7, 7, 7}, result, EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Scalar Operations")
    class ScalarOps {

        @Test
        @DisplayName("scale multiplies by scalar")
        void scaleMultipliesByScalar() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> scaled = TypedOps.scale(a, 3.0f)) {

                float[] result = scaled.underlying().toFloatArray();
                assertArrayEquals(new float[]{3, 6, 9, 12}, result, EPSILON);
            }
        }

        @Test
        @DisplayName("scale works with F64")
        void scaleWorksWithF64() {
            try (TypedTensor<Vector, F64, Cpu> a = TypedTensor.fromDoubleArray(
                    new double[]{1.0, 2.0, 3.0}, new Vector(3), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F64, Cpu> scaled = TypedOps.scale(a, 2.5)) {

                double[] result = scaled.underlying().toDoubleArray();
                assertArrayEquals(new double[]{2.5, 5.0, 7.5}, result, EPSILON_D);
            }
        }

        @Test
        @DisplayName("addScalar adds to all elements")
        void addScalarAddsToAllElements() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> shifted = TypedOps.addScalar(a, 10.0f)) {

                float[] result = shifted.underlying().toFloatArray();
                assertArrayEquals(new float[]{11, 12, 13, 14}, result, EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Unary Operations")
    class UnaryOps {

        @Test
        @DisplayName("neg negates all elements")
        void negNegatesElements() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, -2, 3, -4}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> negated = TypedOps.neg(a)) {

                float[] result = negated.underlying().toFloatArray();
                assertArrayEquals(new float[]{-1, 2, -3, 4}, result, EPSILON);
            }
        }

        @Test
        @DisplayName("abs computes absolute values")
        void absComputesAbsoluteValues() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{-1, 2, -3, 4}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = TypedOps.abs(a)) {

                float[] data = result.underlying().toFloatArray();
                assertArrayEquals(new float[]{1, 2, 3, 4}, data, EPSILON);
            }
        }

        @Test
        @DisplayName("sqrt computes square roots")
        void sqrtComputesSquareRoots() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, 4, 9, 16}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = TypedOps.sqrt(a)) {

                float[] data = result.underlying().toFloatArray();
                assertArrayEquals(new float[]{1, 2, 3, 4}, data, EPSILON);
            }
        }

        @Test
        @DisplayName("exp computes exponentials")
        void expComputesExponentials() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{0, 1, 2}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = TypedOps.exp(a)) {

                float[] data = result.underlying().toFloatArray();
                assertEquals(1.0f, data[0], EPSILON);
                assertEquals((float) Math.E, data[1], 0.001f);
                assertEquals((float) (Math.E * Math.E), data[2], 0.01f);
            }
        }

        @Test
        @DisplayName("log computes natural logarithms")
        void logComputesLogarithms() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.fromFloatArray(
                    new float[]{1, (float) Math.E, (float) (Math.E * Math.E)}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> result = TypedOps.log(a)) {

                float[] data = result.underlying().toFloatArray();
                assertEquals(0.0f, data[0], EPSILON);
                assertEquals(1.0f, data[1], 0.001f);
                assertEquals(2.0f, data[2], 0.001f);
            }
        }

        @Test
        @DisplayName("neg works with F64")
        void negWorksWithF64() {
            try (TypedTensor<Vector, F64, Cpu> a = TypedTensor.fromDoubleArray(
                    new double[]{1.0, -2.0, 3.0}, new Vector(3), F64.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F64, Cpu> negated = TypedOps.neg(a)) {

                double[] result = negated.underlying().toDoubleArray();
                assertArrayEquals(new double[]{-1.0, 2.0, -3.0}, result, EPSILON_D);
            }
        }
    }

    @Nested
    @DisplayName("Type Safety")
    class TypeSafety {

        @Test
        @DisplayName("operations preserve shape type")
        void operationsPreserveShapeType() {
            try (TypedTensor<Matrix, F32, Cpu> a = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> b = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> c = TypedOps.add(a, b)) {

                // The return type is TypedTensor<Matrix, F32, Cpu>
                assertEquals(2, c.rank());
                assertEquals(12, c.elementCount());
            }
        }

        @Test
        @DisplayName("operations preserve dtype type")
        void operationsPreserveDtypeType() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> b = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> c = TypedOps.add(a, b)) {

                assertEquals(F32.INSTANCE, c.dtypeType());
            }
        }

        @Test
        @DisplayName("operations preserve device type")
        void operationsPreserveDeviceType() {
            try (TypedTensor<Vector, F32, Cpu> a = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> b = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> c = TypedOps.add(a, b)) {

                assertEquals(Cpu.INSTANCE, c.deviceType());
            }
        }

        // Note: The following would not compile if uncommented, demonstrating type safety:
        //
        // @Test
        // void dtypeMismatchDoesNotCompile() {
        //     TypedTensor<Vector, F32, Cpu> a = TypedTensor.zeros(new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
        //     TypedTensor<Vector, F64, Cpu> b = TypedTensor.zeros(new Vector(5), F64.INSTANCE, Cpu.INSTANCE);
        //     TypedOps.add(a, b);  // ERROR: F32 != F64
        // }
        //
        // @Test
        // void deviceMismatchDoesNotCompile() {
        //     TypedTensor<Vector, F32, Cpu> a = TypedTensor.zeros(new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
        //     TypedTensor<Vector, F32, Nvidia> b = TypedTensor.zeros(new Vector(5), F32.INSTANCE, Nvidia.DEFAULT);
        //     TypedOps.add(a, b);  // ERROR: Cpu != Nvidia
        // }
    }
}
