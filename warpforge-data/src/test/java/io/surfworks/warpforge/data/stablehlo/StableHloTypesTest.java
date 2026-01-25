package io.surfworks.warpforge.data.stablehlo;

import io.surfworks.warpforge.data.stablehlo.StableHloTypes.FunctionType;
import io.surfworks.warpforge.data.stablehlo.StableHloTypes.ScalarType;
import io.surfworks.warpforge.data.stablehlo.StableHloTypes.TensorType;
import io.surfworks.warpforge.data.stablehlo.StableHloTypes.Type;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StableHloTypesTest {

    @Nested
    class ScalarTypeTests {

        @Test
        void testToMlir() {
            assertEquals("f32", ScalarType.F32.toMlir());
            assertEquals("f64", ScalarType.F64.toMlir());
            assertEquals("i32", ScalarType.I32.toMlir());
            assertEquals("i64", ScalarType.I64.toMlir());
            assertEquals("bf16", ScalarType.BF16.toMlir());
        }

        @Test
        void testFromMlir() {
            assertEquals(ScalarType.F32, ScalarType.fromMlir("f32"));
            assertEquals(ScalarType.F64, ScalarType.fromMlir("f64"));
            assertEquals(ScalarType.I32, ScalarType.fromMlir("i32"));
            assertEquals(ScalarType.BF16, ScalarType.fromMlir("bf16"));
        }

        @Test
        void testFromMlirInvalid() {
            assertThrows(IllegalArgumentException.class, () -> ScalarType.fromMlir("unknown"));
        }

        @Test
        void testIsFloatingPoint() {
            assertTrue(ScalarType.F32.isFloatingPoint());
            assertTrue(ScalarType.F64.isFloatingPoint());
            assertTrue(ScalarType.F16.isFloatingPoint());
            assertTrue(ScalarType.BF16.isFloatingPoint());
            assertFalse(ScalarType.I32.isFloatingPoint());
            assertFalse(ScalarType.I64.isFloatingPoint());
        }

        @Test
        void testIsInteger() {
            assertTrue(ScalarType.I32.isInteger());
            assertTrue(ScalarType.I64.isInteger());
            assertTrue(ScalarType.I8.isInteger());
            assertTrue(ScalarType.I1.isInteger());
            assertFalse(ScalarType.F32.isInteger());
            assertFalse(ScalarType.BF16.isInteger());
        }

        @Test
        void testByteWidth() {
            assertEquals(4, ScalarType.F32.byteWidth());
            assertEquals(8, ScalarType.F64.byteWidth());
            assertEquals(2, ScalarType.F16.byteWidth());
            assertEquals(2, ScalarType.BF16.byteWidth());
            assertEquals(4, ScalarType.I32.byteWidth());
            assertEquals(8, ScalarType.I64.byteWidth());
            assertEquals(1, ScalarType.I8.byteWidth());
        }
    }

    @Nested
    class TensorTypeTests {

        @Test
        void testOf() {
            TensorType type = TensorType.of(ScalarType.F32, 4, 8);
            assertArrayEquals(new long[]{4, 8}, type.shape());
            assertEquals(ScalarType.F32, type.elementType());
        }

        @Test
        void testRank() {
            assertEquals(0, TensorType.of(ScalarType.F32).rank());
            assertEquals(1, TensorType.of(ScalarType.F32, 4).rank());
            assertEquals(2, TensorType.of(ScalarType.F32, 4, 8).rank());
            assertEquals(3, TensorType.of(ScalarType.F32, 2, 4, 8).rank());
        }

        @Test
        void testDim() {
            TensorType type = TensorType.of(ScalarType.F32, 2, 4, 8);
            assertEquals(2, type.dim(0));
            assertEquals(4, type.dim(1));
            assertEquals(8, type.dim(2));
        }

        @Test
        void testElementCount() {
            assertEquals(1, TensorType.of(ScalarType.F32).elementCount());
            assertEquals(4, TensorType.of(ScalarType.F32, 4).elementCount());
            assertEquals(32, TensorType.of(ScalarType.F32, 4, 8).elementCount());
            assertEquals(64, TensorType.of(ScalarType.F32, 2, 4, 8).elementCount());
        }

        @Test
        void testByteSize() {
            assertEquals(4, TensorType.of(ScalarType.F32).byteSize());
            assertEquals(16, TensorType.of(ScalarType.F32, 4).byteSize());
            assertEquals(256, TensorType.of(ScalarType.F64, 4, 8).byteSize());
        }

        @Test
        void testToMlir() {
            assertEquals("tensor<f32>", TensorType.of(ScalarType.F32).toMlir());
            assertEquals("tensor<4xf32>", TensorType.of(ScalarType.F32, 4).toMlir());
            assertEquals("tensor<4x8xf32>", TensorType.of(ScalarType.F32, 4, 8).toMlir());
            assertEquals("tensor<2x4x8xi64>", TensorType.of(ScalarType.I64, 2, 4, 8).toMlir());
        }

        @Test
        void testFromMlir() {
            TensorType scalar = TensorType.fromMlir("tensor<f32>");
            assertEquals(0, scalar.rank());
            assertEquals(ScalarType.F32, scalar.elementType());

            TensorType vec = TensorType.fromMlir("tensor<4xf32>");
            assertEquals(1, vec.rank());
            assertEquals(4, vec.dim(0));

            TensorType mat = TensorType.fromMlir("tensor<4x8xf32>");
            assertEquals(2, mat.rank());
            assertEquals(4, mat.dim(0));
            assertEquals(8, mat.dim(1));

            TensorType tensor3d = TensorType.fromMlir("tensor<2x4x8xi64>");
            assertEquals(3, tensor3d.rank());
            assertEquals(ScalarType.I64, tensor3d.elementType());
        }

        @Test
        void testFromMlirInvalid() {
            assertThrows(IllegalArgumentException.class, () -> TensorType.fromMlir("invalid"));
            assertThrows(IllegalArgumentException.class, () -> TensorType.fromMlir("tensor<>"));
        }
    }

    @Nested
    class FunctionTypeTests {

        @Test
        void testToMlir() {
            FunctionType func = new FunctionType(
                    List.of(TensorType.of(ScalarType.F32, 1, 8), TensorType.of(ScalarType.F32, 8, 8)),
                    List.of(TensorType.of(ScalarType.F32, 1, 8))
            );

            assertEquals("(tensor<1x8xf32>, tensor<8x8xf32>) -> (tensor<1x8xf32>)", func.toMlir());
        }

        @Test
        void testMultipleReturns() {
            FunctionType func = new FunctionType(
                    List.of(TensorType.of(ScalarType.F32, 4)),
                    List.of(TensorType.of(ScalarType.F32, 4), TensorType.of(ScalarType.I64, 4))
            );

            assertEquals("(tensor<4xf32>) -> (tensor<4xf32>, tensor<4xi64>)", func.toMlir());
        }

        @Test
        void testEmptyInputs() {
            FunctionType func = new FunctionType(
                    List.of(),
                    List.of(TensorType.of(ScalarType.F32, 4))
            );

            assertEquals("() -> (tensor<4xf32>)", func.toMlir());
        }
    }
}
