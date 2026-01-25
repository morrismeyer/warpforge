package io.surfworks.warpforge.core.tensor.typed;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.typed.device.AnyDevice;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.device.Nvidia;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.dtype.I32;
import io.surfworks.warpforge.core.tensor.typed.shape.Dynamic;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Scalar;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Tests for TypedTensor wrapper class.
 */
@DisplayName("TypedTensor")
class TypedTensorTest {

    @Nested
    @DisplayName("Factory Methods")
    class FactoryMethods {

        @Test
        @DisplayName("zeros creates zero-initialized tensor")
        void zerosCreatesZeroTensor() {
            try (TypedTensor<Matrix, F32, Cpu> tensor = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                assertEquals(2, tensor.rank());
                assertEquals(12, tensor.elementCount());
                assertArrayEquals(new int[]{3, 4}, tensor.dimensions());

                // Verify all zeros
                float[] data = tensor.underlying().toFloatArray();
                for (float v : data) {
                    assertEquals(0.0f, v);
                }
            }
        }

        @Test
        @DisplayName("full creates filled tensor")
        void fullCreatesFilledTensor() {
            try (TypedTensor<Vector, F32, Cpu> tensor = TypedTensor.full(
                    3.14f, new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                assertEquals(1, tensor.rank());
                assertEquals(5, tensor.elementCount());

                float[] data = tensor.underlying().toFloatArray();
                for (float v : data) {
                    assertEquals(3.14f, v, 0.0001f);
                }
            }
        }

        @Test
        @DisplayName("fromFloatArray creates tensor from data")
        void fromFloatArrayCreatesTensor() {
            float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            try (TypedTensor<Matrix, F32, Cpu> tensor = TypedTensor.fromFloatArray(
                    data, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                assertArrayEquals(new int[]{2, 3}, tensor.dimensions());

                float[] result = tensor.underlying().toFloatArray();
                assertArrayEquals(data, result);
            }
        }

        @Test
        @DisplayName("fromDoubleArray creates tensor from data")
        void fromDoubleArrayCreatesTensor() {
            double[] data = {1.0, 2.0, 3.0, 4.0};
            try (TypedTensor<Vector, F64, Cpu> tensor = TypedTensor.fromDoubleArray(
                    data, new Vector(4), F64.INSTANCE, Cpu.INSTANCE)) {

                assertEquals(4, tensor.elementCount());

                double[] result = tensor.underlying().toDoubleArray();
                assertArrayEquals(data, result);
            }
        }

        @Test
        @DisplayName("fromIntArray creates tensor from data")
        void fromIntArrayCreatesTensor() {
            int[] data = {1, 2, 3, 4, 5, 6};
            try (TypedTensor<Matrix, I32, Cpu> tensor = TypedTensor.fromIntArray(
                    data, new Matrix(2, 3), I32.INSTANCE, Cpu.INSTANCE)) {

                int[] result = tensor.underlying().toIntArray();
                assertArrayEquals(data, result);
            }
        }

        @Test
        @DisplayName("fromFloatArray rejects wrong dtype")
        void fromFloatArrayRejectsWrongDtype() {
            float[] data = {1.0f, 2.0f};

            assertThrows(IllegalArgumentException.class, () ->
                    TypedTensor.fromFloatArray(data, new Vector(2), F64.INSTANCE, Cpu.INSTANCE));
        }

        @Test
        @DisplayName("zeros rejects GPU device (not yet implemented)")
        void zerosRejectsGpu() {
            assertThrows(UnsupportedOperationException.class, () ->
                    TypedTensor.zeros(new Matrix(10, 10), F32.INSTANCE, Nvidia.DEFAULT));
        }
    }

    @Nested
    @DisplayName("Interop with Untyped Tensor")
    class InteropTests {

        @Test
        @DisplayName("from wraps matching tensor")
        void fromWrapsMatchingTensor() {
            try (Tensor untyped = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3)) {
                TypedTensor<Matrix, F32, Cpu> typed = TypedTensor.from(
                        untyped, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);

                assertSame(untyped, typed.underlying());
                assertEquals(F32.INSTANCE, typed.dtypeType());
                assertEquals(Cpu.INSTANCE, typed.deviceType());
            }
        }

        @Test
        @DisplayName("from rejects dtype mismatch")
        void fromRejectsDtypeMismatch() {
            try (Tensor untyped = Tensor.zeros(ScalarType.F32, 3, 3)) {
                assertThrows(IllegalArgumentException.class, () ->
                        TypedTensor.from(untyped, new Matrix(3, 3), F64.INSTANCE, Cpu.INSTANCE));
            }
        }

        @Test
        @DisplayName("from rejects shape mismatch")
        void fromRejectsShapeMismatch() {
            try (Tensor untyped = Tensor.zeros(ScalarType.F32, 2, 3)) {
                assertThrows(IllegalArgumentException.class, () ->
                        TypedTensor.from(untyped, new Matrix(3, 2), F32.INSTANCE, Cpu.INSTANCE));
            }
        }

        @Test
        @DisplayName("fromDynamic accepts any shape")
        void fromDynamicAcceptsAnyShape() {
            try (Tensor untyped = Tensor.zeros(ScalarType.F32, 2, 3, 4)) {
                TypedTensor<Dynamic, F32, Cpu> typed = TypedTensor.fromDynamic(
                        untyped, F32.INSTANCE, Cpu.INSTANCE);

                assertArrayEquals(new int[]{2, 3, 4}, typed.dimensions());
                assertEquals(3, typed.rank());
            }
        }

        @Test
        @DisplayName("underlying returns wrapped tensor")
        void underlyingReturnsWrappedTensor() {
            try (TypedTensor<Matrix, F32, Cpu> typed = TypedTensor.zeros(
                    new Matrix(5, 5), F32.INSTANCE, Cpu.INSTANCE)) {

                Tensor underlying = typed.underlying();

                assertNotNull(underlying);
                assertEquals(ScalarType.F32, underlying.dtype());
                assertArrayEquals(new int[]{5, 5}, underlying.shape());
            }
        }
    }

    @Nested
    @DisplayName("Type Accessors")
    class TypeAccessors {

        @Test
        @DisplayName("shapeType returns correct phantom type")
        void shapeTypeReturnsCorrect() {
            try (TypedTensor<Matrix, F32, Cpu> tensor = TypedTensor.zeros(
                    new Matrix(10, 20), F32.INSTANCE, Cpu.INSTANCE)) {

                Matrix shape = tensor.shapeType();

                assertEquals(10, shape.rows());
                assertEquals(20, shape.cols());
            }
        }

        @Test
        @DisplayName("dtypeType returns correct phantom type")
        void dtypeTypeReturnsCorrect() {
            try (TypedTensor<Vector, F32, Cpu> tensor = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                assertEquals(F32.INSTANCE, tensor.dtypeType());
                assertEquals(ScalarType.F32, tensor.dtypeType().scalarType());
            }
        }

        @Test
        @DisplayName("deviceType returns correct phantom type")
        void deviceTypeReturnsCorrect() {
            try (TypedTensor<Vector, F32, Cpu> tensor = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                assertEquals(Cpu.INSTANCE, tensor.deviceType());
                assertEquals("cpu", tensor.deviceType().deviceName());
            }
        }
    }

    @Nested
    @DisplayName("Type Conversions")
    class TypeConversions {

        @Test
        @DisplayName("reshape changes shape type")
        void reshapeChangesShape() {
            try (TypedTensor<Matrix, F32, Cpu> matrix = TypedTensor.zeros(
                    new Matrix(4, 6), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<Vector, F32, Cpu> vector = matrix.reshape(new Vector(24));

                assertEquals(1, vector.rank());
                assertEquals(24, vector.elementCount());
            }
        }

        @Test
        @DisplayName("reshape rejects element count mismatch")
        void reshapeRejectsCountMismatch() {
            try (TypedTensor<Matrix, F32, Cpu> tensor = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class, () ->
                        tensor.reshape(new Vector(10)));
            }
        }

        @Test
        @DisplayName("asAnyDevice converts device type")
        void asAnyDeviceConverts() {
            try (TypedTensor<Matrix, F32, Cpu> cpuTensor = TypedTensor.zeros(
                    new Matrix(2, 2), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<Matrix, F32, AnyDevice> anyTensor = cpuTensor.asAnyDevice();

                assertEquals(AnyDevice.INSTANCE, anyTensor.deviceType());
            }
        }

        @Test
        @DisplayName("asDynamic converts shape type")
        void asDynamicConverts() {
            try (TypedTensor<Matrix, F32, Cpu> matrix = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                TypedTensor<Dynamic, F32, Cpu> dynamic = matrix.asDynamic();

                assertTrue(dynamic.shapeType() instanceof Dynamic);
                assertArrayEquals(new int[]{3, 4}, dynamic.dimensions());
            }
        }

        @Test
        @DisplayName("copy creates independent tensor")
        void copyCreatesIndependent() {
            try (TypedTensor<Matrix, F32, Cpu> original = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4}, new Matrix(2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> copied = original.copy()) {

                assertNotSame(original.underlying(), copied.underlying());
                assertArrayEquals(original.underlying().toFloatArray(),
                                  copied.underlying().toFloatArray());

                // Modify original, verify copy unchanged
                original.underlying().setFloat(99.0f, 0, 0);
                assertEquals(1.0f, copied.underlying().getFloat(0, 0));
            }
        }
    }

    @Nested
    @DisplayName("Shape Specialization")
    class ShapeSpecialization {

        @Test
        @DisplayName("toVector specializes dynamic to vector")
        void toVectorSpecializes() {
            try (Tensor untyped = Tensor.zeros(ScalarType.F32, 10)) {
                TypedTensor<Dynamic, F32, Cpu> dynamic = TypedTensor.fromDynamic(
                        untyped, F32.INSTANCE, Cpu.INSTANCE);

                TypedTensor<Vector, F32, Cpu> vector = TypedTensor.toVector(dynamic);

                assertEquals(10, vector.shapeType().length());
            }
        }

        @Test
        @DisplayName("toMatrix specializes dynamic to matrix")
        void toMatrixSpecializes() {
            try (Tensor untyped = Tensor.zeros(ScalarType.F32, 3, 4)) {
                TypedTensor<Dynamic, F32, Cpu> dynamic = TypedTensor.fromDynamic(
                        untyped, F32.INSTANCE, Cpu.INSTANCE);

                TypedTensor<Matrix, F32, Cpu> matrix = TypedTensor.toMatrix(dynamic);

                assertEquals(3, matrix.shapeType().rows());
                assertEquals(4, matrix.shapeType().cols());
            }
        }

        @Test
        @DisplayName("toMatrix rejects wrong rank")
        void toMatrixRejectsWrongRank() {
            try (Tensor untyped = Tensor.zeros(ScalarType.F32, 2, 3, 4)) {
                TypedTensor<Dynamic, F32, Cpu> dynamic = TypedTensor.fromDynamic(
                        untyped, F32.INSTANCE, Cpu.INSTANCE);

                assertThrows(IllegalArgumentException.class, () ->
                        TypedTensor.toMatrix(dynamic));
            }
        }

        @Test
        @DisplayName("toScalar specializes rank-0 tensor")
        void toScalarSpecializes() {
            try (Tensor untyped = Tensor.fromFloatArray(new float[]{42.0f})) {
                TypedTensor<Dynamic, F32, Cpu> dynamic = TypedTensor.fromDynamic(
                        untyped, F32.INSTANCE, Cpu.INSTANCE);

                TypedTensor<Scalar, F32, Cpu> scalar = TypedTensor.toScalar(dynamic);

                assertEquals(0, scalar.rank());
                assertEquals(1, scalar.elementCount());
            }
        }
    }

    @Nested
    @DisplayName("Device Transfer")
    class DeviceTransfer {

        @Test
        @DisplayName("to CPU creates copy")
        void toCpuCreatesCopy() {
            try (TypedTensor<Matrix, F32, Cpu> original = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4}, new Matrix(2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> copy = original.to(Cpu.INSTANCE)) {

                assertNotSame(original.underlying(), copy.underlying());
                assertArrayEquals(original.underlying().toFloatArray(),
                                  copy.underlying().toFloatArray());
            }
        }

        @Test
        @DisplayName("to GPU throws UnsupportedOperationException")
        void toGpuThrowsUnsupported() {
            try (TypedTensor<Matrix, F32, Cpu> tensor = TypedTensor.zeros(
                    new Matrix(2, 2), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(UnsupportedOperationException.class, () ->
                        tensor.to(Nvidia.DEFAULT));
            }
        }
    }

    @Nested
    @DisplayName("Lifecycle")
    class Lifecycle {

        @Test
        @DisplayName("toString includes type information")
        void toStringIncludesTypeInfo() {
            try (TypedTensor<Matrix, F32, Cpu> tensor = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                String str = tensor.toString();

                assertTrue(str.contains("Matrix"));
                assertTrue(str.contains("F32"));
                assertTrue(str.contains("Cpu"));
                assertTrue(str.contains("12")); // elements
            }
        }
    }
}
