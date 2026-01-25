package io.surfworks.warpforge.core.tensor;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorBridgeTest {

    // =========================================================================
    // DType ↔ ScalarType Conversion Tests
    // =========================================================================

    @Nested
    class DTypeConversionTests {

        @Test
        void testStandardFloatTypes() {
            assertEquals(ScalarType.F32, TensorBridge.toScalarType(DType.F32));
            assertEquals(ScalarType.F64, TensorBridge.toScalarType(DType.F64));
            assertEquals(ScalarType.F16, TensorBridge.toScalarType(DType.F16));
            assertEquals(ScalarType.BF16, TensorBridge.toScalarType(DType.BF16));
        }

        @Test
        void testP3109FP8Types() {
            assertEquals(ScalarType.F8_E5M2, TensorBridge.toScalarType(DType.F8_E5M2));
            assertEquals(ScalarType.F8_E4M3, TensorBridge.toScalarType(DType.F8_E4M3));
            assertEquals(ScalarType.F8_E4M3FN, TensorBridge.toScalarType(DType.F8_E4M3FN));
            assertEquals(ScalarType.F8_E8M0, TensorBridge.toScalarType(DType.F8_E8M0));
        }

        @Test
        void testP3109FP6Types() {
            assertEquals(ScalarType.F6_E3M2, TensorBridge.toScalarType(DType.F6_E3M2));
            assertEquals(ScalarType.F6_E2M3, TensorBridge.toScalarType(DType.F6_E2M3));
        }

        @Test
        void testP3109FP4Types() {
            assertEquals(ScalarType.F4_E2M1, TensorBridge.toScalarType(DType.F4_E2M1));
            assertEquals(ScalarType.F4_E1M2, TensorBridge.toScalarType(DType.F4_E1M2));
        }

        @Test
        void testIntegerTypes() {
            assertEquals(ScalarType.I8, TensorBridge.toScalarType(DType.I8));
            assertEquals(ScalarType.I16, TensorBridge.toScalarType(DType.I16));
            assertEquals(ScalarType.I32, TensorBridge.toScalarType(DType.I32));
            assertEquals(ScalarType.I64, TensorBridge.toScalarType(DType.I64));
        }

        @Test
        void testUnsignedMapsToSigned() {
            // Unsigned types map to signed equivalents
            assertEquals(ScalarType.I8, TensorBridge.toScalarType(DType.U8));
            assertEquals(ScalarType.I16, TensorBridge.toScalarType(DType.U16));
            assertEquals(ScalarType.I32, TensorBridge.toScalarType(DType.U32));
            assertEquals(ScalarType.I64, TensorBridge.toScalarType(DType.U64));
        }

        @Test
        void testBoolType() {
            assertEquals(ScalarType.BOOL, TensorBridge.toScalarType(DType.BOOL));
        }

        @Test
        void testQuantizedTypesThrow() {
            assertThrows(IllegalArgumentException.class,
                () -> TensorBridge.toScalarType(DType.Q4_0));
            assertThrows(IllegalArgumentException.class,
                () -> TensorBridge.toScalarType(DType.Q4_K_M));
            assertThrows(IllegalArgumentException.class,
                () -> TensorBridge.toScalarType(DType.Q8_0));
        }
    }

    @Nested
    class ScalarTypeConversionTests {

        @Test
        void testRoundTrip() {
            // Test that non-quantized types round-trip correctly
            DType[] testTypes = {
                DType.F32, DType.F64, DType.F16, DType.BF16,
                DType.F8_E5M2, DType.F8_E4M3, DType.F8_E4M3FN,
                DType.F4_E2M1, DType.F4_E1M2,
                DType.I8, DType.I16, DType.I32, DType.I64,
                DType.BOOL
            };

            for (DType dtype : testTypes) {
                ScalarType scalarType = TensorBridge.toScalarType(dtype);
                DType roundTrip = TensorBridge.toDType(scalarType);
                // Note: BOOL maps to I1 in ScalarType, which maps back to BOOL
                if (dtype == DType.BOOL) {
                    assertEquals(DType.BOOL, roundTrip);
                } else {
                    assertEquals(dtype, roundTrip, "Round-trip failed for " + dtype);
                }
            }
        }

        @ParameterizedTest
        @EnumSource(ScalarType.class)
        void testAllScalarTypesConvert(ScalarType scalarType) {
            // All ScalarTypes should convert to DType
            DType dtype = TensorBridge.toDType(scalarType);
            assertNotNull(dtype);
        }
    }

    // =========================================================================
    // TensorInfo → TensorSpec Conversion Tests
    // =========================================================================

    @Nested
    class TensorSpecConversionTests {

        @Test
        void testBasicConversion() {
            TensorInfo info = new TensorInfo(
                "test_tensor",
                DType.F32,
                new long[]{2, 3, 4},
                0,
                96
            );

            TensorSpec spec = TensorBridge.toTensorSpec(info);

            assertEquals(ScalarType.F32, spec.dtype());
            assertArrayEquals(new int[]{2, 3, 4}, spec.shape());
            assertEquals(24, spec.elementCount());
            assertEquals(96, spec.byteSize());
        }

        @Test
        void testScalarTensor() {
            TensorInfo info = new TensorInfo(
                "scalar",
                DType.F32,
                new long[]{},
                0,
                4
            );

            TensorSpec spec = TensorBridge.toTensorSpec(info);

            assertEquals(0, spec.rank());
            assertEquals(1, spec.elementCount());
        }

        @Test
        void test1DTensor() {
            TensorInfo info = new TensorInfo(
                "vector",
                DType.F64,
                new long[]{100},
                0,
                800
            );

            TensorSpec spec = TensorBridge.toTensorSpec(info);

            assertEquals(1, spec.rank());
            assertArrayEquals(new int[]{100}, spec.shape());
            assertEquals(ScalarType.F64, spec.dtype());
        }

        @Test
        void testRowMajorStrides() {
            TensorInfo info = new TensorInfo(
                "matrix",
                DType.F32,
                new long[]{3, 4},
                0,
                48
            );

            TensorSpec spec = TensorBridge.toTensorSpec(info);

            // Row-major strides for [3, 4]: [4, 1]
            assertArrayEquals(new long[]{4, 1}, spec.strides());
            assertTrue(spec.isContiguous());
        }
    }

    // =========================================================================
    // TensorView → Tensor Conversion Tests
    // =========================================================================

    @Nested
    class TensorViewConversionTests {

        @Test
        void testZeroCopyConversion() {
            try (Arena arena = Arena.ofConfined()) {
                // Create a TensorView
                TensorView view = createTestTensorView(arena, new float[]{1, 2, 3, 4, 5, 6}, 2, 3);

                // Convert to Tensor (zero-copy)
                Tensor tensor = TensorBridge.toTensorZeroCopy(view);

                // Verify data is shared (same memory)
                assertSame(view.data(), tensor.data());

                // Verify values
                assertEquals(1.0f, tensor.getFloat(0, 0), 0.001f);
                assertEquals(6.0f, tensor.getFloat(1, 2), 0.001f);

                // Verify spec
                assertEquals(ScalarType.F32, tensor.dtype());
                assertArrayEquals(new int[]{2, 3}, tensor.shape());
            }
        }

        @Test
        void testCopyConversion() {
            try (Arena sourceArena = Arena.ofConfined()) {
                TensorView view = createTestTensorView(sourceArena, new float[]{1, 2, 3, 4}, 2, 2);

                // Convert to Tensor (copy)
                try (Tensor tensor = TensorBridge.toTensorCopy(view)) {
                    // Verify data is copied (different memory)
                    assertNotSame(view.data(), tensor.data());

                    // Verify values
                    assertEquals(1.0f, tensor.getFloat(0, 0), 0.001f);
                    assertEquals(4.0f, tensor.getFloat(1, 1), 0.001f);

                    // Verify spec
                    assertEquals(ScalarType.F32, tensor.dtype());
                    assertArrayEquals(new int[]{2, 2}, tensor.shape());
                }
            }
        }

        @Test
        void testCopyWithSharedArena() {
            Tensor tensor;
            try (Arena targetArena = Arena.ofShared()) {
                // Create source view and copy to shared arena
                try (Arena sourceArena = Arena.ofConfined()) {
                    TensorView view = createTestTensorView(sourceArena, new float[]{1, 2, 3}, 3);

                    // Convert using shared arena
                    tensor = TensorBridge.toTensor(view, targetArena);

                    // Verify values while source is still valid
                    assertEquals(1.0f, tensor.getFloatFlat(0), 0.001f);
                    assertEquals(3.0f, tensor.getFloatFlat(2), 0.001f);
                }

                // Source arena is now closed, tensor should still be valid
                // because data was copied to targetArena
                assertEquals(2.0f, tensor.getFloatFlat(1), 0.001f);
                assertEquals(1.0f, tensor.getFloatFlat(0), 0.001f);
            }
        }

        @Test
        void testToFloatArray() {
            try (Arena arena = Arena.ofConfined()) {
                float[] values = {1.5f, 2.5f, 3.5f, 4.5f};
                TensorView view = createTestTensorView(arena, values, 2, 2);

                try (Tensor tensor = TensorBridge.toTensorCopy(view)) {
                    float[] result = tensor.toFloatArray();
                    assertArrayEquals(values, result, 0.001f);
                }
            }
        }
    }

    // =========================================================================
    // F16/BF16 Conversion Tests
    // =========================================================================

    @Nested
    class HalfPrecisionConversionTests {

        @Test
        void testF16ToFloat32Conversion() {
            try (Arena arena = Arena.ofConfined()) {
                // Create F16 tensor view
                TensorView view = createF16TensorView(arena, new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 4);

                // Convert to F32
                try (Tensor tensor = TensorBridge.toTensorAsFloat32(view)) {
                    assertEquals(ScalarType.F32, tensor.dtype());
                    assertEquals(1.0f, tensor.getFloatFlat(0), 0.01f);
                    assertEquals(2.0f, tensor.getFloatFlat(1), 0.01f);
                    assertEquals(3.0f, tensor.getFloatFlat(2), 0.01f);
                    assertEquals(4.0f, tensor.getFloatFlat(3), 0.01f);
                }
            }
        }

        @Test
        void testBF16ToFloat32Conversion() {
            try (Arena arena = Arena.ofConfined()) {
                // Create BF16 tensor view
                TensorView view = createBF16TensorView(arena, new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 4);

                // Convert to F32
                try (Tensor tensor = TensorBridge.toTensorAsFloat32(view)) {
                    assertEquals(ScalarType.F32, tensor.dtype());
                    assertEquals(1.0f, tensor.getFloatFlat(0), 0.01f);
                    assertEquals(2.0f, tensor.getFloatFlat(1), 0.01f);
                    assertEquals(3.0f, tensor.getFloatFlat(2), 0.01f);
                    assertEquals(4.0f, tensor.getFloatFlat(3), 0.01f);
                }
            }
        }

        @Test
        void testF32ToFloat32IsNoop() {
            try (Arena arena = Arena.ofConfined()) {
                float[] values = {1.0f, 2.0f, 3.0f};
                TensorView view = createTestTensorView(arena, values, 3);

                try (Tensor tensor = TensorBridge.toTensorAsFloat32(view)) {
                    assertEquals(ScalarType.F32, tensor.dtype());
                    assertArrayEquals(values, tensor.toFloatArray(), 0.001f);
                }
            }
        }
    }

    // =========================================================================
    // Utility Method Tests
    // =========================================================================

    @Nested
    class UtilityMethodTests {

        @Test
        void testIsDirectlyConvertible() {
            assertTrue(TensorBridge.isDirectlyConvertible(DType.F32));
            assertTrue(TensorBridge.isDirectlyConvertible(DType.F16));
            assertTrue(TensorBridge.isDirectlyConvertible(DType.BF16));
            assertTrue(TensorBridge.isDirectlyConvertible(DType.F8_E5M2));
            assertTrue(TensorBridge.isDirectlyConvertible(DType.I32));

            assertFalse(TensorBridge.isDirectlyConvertible(DType.Q4_0));
            assertFalse(TensorBridge.isDirectlyConvertible(DType.Q4_K_M));
            assertFalse(TensorBridge.isDirectlyConvertible(DType.Q8_0));
        }
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    private TensorView createTestTensorView(Arena arena, float[] values, int... shape) {
        MemorySegment data = arena.allocate((long) values.length * 4);
        for (int i = 0; i < values.length; i++) {
            data.setAtIndex(ValueLayout.JAVA_FLOAT, i, values[i]);
        }

        long[] longShape = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            longShape[i] = shape[i];
        }

        TensorInfo info = new TensorInfo(
            "test",
            DType.F32,
            longShape,
            0,
            (long) values.length * 4
        );

        return new TensorView(data, info);
    }

    private TensorView createF16TensorView(Arena arena, float[] values, int... shape) {
        MemorySegment data = arena.allocate((long) values.length * 2);
        for (int i = 0; i < values.length; i++) {
            short f16Bits = floatToF16(values[i]);
            data.setAtIndex(ValueLayout.JAVA_SHORT, i, f16Bits);
        }

        long[] longShape = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            longShape[i] = shape[i];
        }

        TensorInfo info = new TensorInfo(
            "f16_test",
            DType.F16,
            longShape,
            0,
            (long) values.length * 2
        );

        return new TensorView(data, info);
    }

    private TensorView createBF16TensorView(Arena arena, float[] values, int... shape) {
        MemorySegment data = arena.allocate((long) values.length * 2);
        for (int i = 0; i < values.length; i++) {
            short bf16Bits = DType.floatToBf16(values[i]);
            data.setAtIndex(ValueLayout.JAVA_SHORT, i, bf16Bits);
        }

        long[] longShape = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            longShape[i] = shape[i];
        }

        TensorInfo info = new TensorInfo(
            "bf16_test",
            DType.BF16,
            longShape,
            0,
            (long) values.length * 2
        );

        return new TensorView(data, info);
    }

    /**
     * Convert float to F16 bits.
     */
    private static short floatToF16(float f) {
        int bits = Float.floatToRawIntBits(f);
        int sign = (bits >>> 16) & 0x8000;
        int exponent = ((bits >>> 23) & 0xFF) - 127 + 15;
        int mantissa = (bits >>> 13) & 0x3FF;

        if (exponent <= 0) {
            return (short) sign; // Underflow to zero
        }
        if (exponent >= 31) {
            return (short) (sign | 0x7C00); // Overflow to infinity
        }

        return (short) (sign | (exponent << 10) | mantissa);
    }
}
