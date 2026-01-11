package io.surfworks.warpforge.core.testing;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorAssertTest {

    @Nested
    @DisplayName("assertEquals")
    class AssertEqualsTests {

        @Test
        void identicalTensorsPass() {
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 2, 2);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 2, 2)) {
                assertDoesNotThrow(() -> TensorAssert.assertEquals(a, b));
            }
        }

        @Test
        void withinTolerancePass() {
            try (Tensor a = Tensor.fromFloatArray(new float[]{1.0f, 2.0f, 3.0f}, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{1.00001f, 2.00001f, 3.00001f}, 3)) {
                ToleranceConfig tol = new ToleranceConfig(1e-4, 0);
                assertDoesNotThrow(() -> TensorAssert.assertEquals(a, b, tol));
            }
        }

        @Test
        void outsideToleranceFails() {
            try (Tensor a = Tensor.fromFloatArray(new float[]{1.0f, 2.0f, 3.0f}, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{1.1f, 2.0f, 3.0f}, 3)) {
                ToleranceConfig tol = new ToleranceConfig(1e-4, 1e-4);
                AssertionError error = assertThrows(AssertionError.class, () ->
                    TensorAssert.assertEquals(a, b, tol));
                assertTrue(error.getMessage().contains("tensors differ"));
                assertTrue(error.getMessage().contains("1/")); // 1 of 3 elements
            }
        }

        @Test
        void shapeMismatchFails() {
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 2, 2);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                AssertionError error = assertThrows(AssertionError.class, () ->
                    TensorAssert.assertEquals(a, b));
                assertTrue(error.getMessage().contains("shape mismatch"));
            }
        }

        @Test
        void dtypeMismatchFails() {
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4);
                 Tensor b = Tensor.fromDoubleArray(new double[]{1, 2, 3, 4}, 4)) {
                AssertionError error = assertThrows(AssertionError.class, () ->
                    TensorAssert.assertEquals(a, b));
                assertTrue(error.getMessage().contains("dtype mismatch"));
            }
        }

        @Test
        void customMessageIncluded() {
            try (Tensor a = Tensor.fromFloatArray(new float[]{1}, 1);
                 Tensor b = Tensor.fromFloatArray(new float[]{2}, 1)) {
                AssertionError error = assertThrows(AssertionError.class, () ->
                    TensorAssert.assertEquals("Custom message", a, b, ToleranceConfig.STRICT));
                assertTrue(error.getMessage().startsWith("Custom message:"));
            }
        }

        @Test
        void nullHandling() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1}, 1)) {
                // Both null passes
                assertDoesNotThrow(() -> TensorAssert.assertEquals(null, null));

                // One null fails
                assertThrows(AssertionError.class, () -> TensorAssert.assertEquals(null, t));
                assertThrows(AssertionError.class, () -> TensorAssert.assertEquals(t, null));
            }
        }

        @Test
        void detailedErrorMessage() {
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 2, 999, 4, 5, 6}, 2, 3)) {
                AssertionError error = assertThrows(AssertionError.class, () ->
                    TensorAssert.assertEquals(a, b, ToleranceConfig.STRICT));
                String msg = error.getMessage();
                assertTrue(msg.contains("max diff"));
                assertTrue(msg.contains("[0, 2]")); // Index of mismatch
            }
        }
    }

    @Nested
    @DisplayName("assertShapeEquals")
    class AssertShapeEqualsTests {

        @Test
        void matchingShapesPass() {
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{9, 8, 7, 6, 5, 4}, 2, 3)) {
                assertDoesNotThrow(() -> TensorAssert.assertShapeEquals(a, b));
            }
        }

        @Test
        void mismatchingShapesFail() {
            try (Tensor a = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
                 Tensor b = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 3, 2)) {
                assertThrows(AssertionError.class, () -> TensorAssert.assertShapeEquals(a, b));
            }
        }

        @Test
        void expectedShapeArray() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3)) {
                assertDoesNotThrow(() -> TensorAssert.assertShapeEquals(new int[]{2, 3}, t));
                assertThrows(AssertionError.class, () ->
                    TensorAssert.assertShapeEquals(new int[]{3, 2}, t));
            }
        }
    }

    @Nested
    @DisplayName("assertDtypeEquals")
    class AssertDtypeEqualsTests {

        @Test
        void matchingDtypePass() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1, 2, 3}, 3)) {
                assertDoesNotThrow(() -> TensorAssert.assertDtypeEquals(ScalarType.F32, t));
            }
        }

        @Test
        void mismatchingDtypeFails() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1, 2, 3}, 3)) {
                assertThrows(AssertionError.class, () ->
                    TensorAssert.assertDtypeEquals(ScalarType.F64, t));
            }
        }
    }

    @Nested
    @DisplayName("assertAllClose")
    class AssertAllCloseTests {

        @Test
        void allElementsClosePass() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1.0f, 1.00001f, 0.99999f}, 3)) {
                ToleranceConfig tol = new ToleranceConfig(1e-4, 0);
                assertDoesNotThrow(() -> TensorAssert.assertAllClose(1.0, t, tol));
            }
        }

        @Test
        void elementNotCloseFails() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1.0f, 2.0f, 1.0f}, 3)) {
                ToleranceConfig tol = new ToleranceConfig(1e-4, 0);
                AssertionError error = assertThrows(AssertionError.class, () ->
                    TensorAssert.assertAllClose(1.0, t, tol));
                assertTrue(error.getMessage().contains("[1]"));
            }
        }
    }

    @Nested
    @DisplayName("assertFinite")
    class AssertFiniteTests {

        @Test
        void finiteValuesPass() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1.0f, -2.0f, 0.0f, 1e30f}, 4)) {
                assertDoesNotThrow(() -> TensorAssert.assertFinite(t));
            }
        }

        @Test
        void nanFails() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1.0f, Float.NaN, 3.0f}, 3)) {
                AssertionError error = assertThrows(AssertionError.class, () ->
                    TensorAssert.assertFinite(t));
                assertTrue(error.getMessage().contains("non-finite"));
            }
        }

        @Test
        void infinityFails() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1.0f, Float.POSITIVE_INFINITY, 3.0f}, 3)) {
                AssertionError error = assertThrows(AssertionError.class, () ->
                    TensorAssert.assertFinite(t));
                assertTrue(error.getMessage().contains("non-finite"));
            }
        }
    }

    @Nested
    @DisplayName("assertNoNaN")
    class AssertNoNaNTests {

        @Test
        void noNaNPass() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1.0f, Float.POSITIVE_INFINITY, -1.0f}, 3)) {
                assertDoesNotThrow(() -> TensorAssert.assertNoNaN(t));
            }
        }

        @Test
        void containsNaNFails() {
            try (Tensor t = Tensor.fromFloatArray(new float[]{1.0f, Float.NaN, 3.0f}, 3)) {
                AssertionError error = assertThrows(AssertionError.class, () ->
                    TensorAssert.assertNoNaN(t));
                assertTrue(error.getMessage().contains("NaN"));
            }
        }
    }
}
