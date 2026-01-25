package io.surfworks.warpforge.core.tensor.typed.numerical;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.dtype.I32;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Scalar;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Comprehensive tests for NumericalContext NaN/Inf checking.
 */
@DisplayName("NumericalContext")
class NumericalContextTest {

    @AfterEach
    void cleanup() {
        // Ensure we're back to default policy after each test
        assertEquals(NaNPolicy.IGNORE, NumericalContext.currentPolicy());
    }

    // ==================== Default State Tests ====================

    @Nested
    @DisplayName("Default State")
    class DefaultState {

        @Test
        @DisplayName("default policy is IGNORE")
        void defaultPolicyIsIgnore() {
            assertEquals(NaNPolicy.IGNORE, NumericalContext.currentPolicy());
        }

        @Test
        @DisplayName("check does nothing with IGNORE policy")
        void checkDoesNothingWithIgnore() {
            try (TypedTensor<Vector, F32, Cpu> t = createTensorWithNaN()) {
                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }
    }

    // ==================== Policy Switching ====================

    @Nested
    @DisplayName("Policy Switching")
    class PolicySwitching {

        @ParameterizedTest
        @EnumSource(NaNPolicy.class)
        @DisplayName("can switch to all policies")
        void canSwitchToAllPolicies(NaNPolicy policy) {
            try (var ctx = new NumericalContext(policy)) {
                assertEquals(policy, NumericalContext.currentPolicy());
            }
            assertEquals(NaNPolicy.IGNORE, NumericalContext.currentPolicy());
        }

        @Test
        @DisplayName("convenience method errorOnInvalid")
        void convenienceMethodErrorOnInvalid() {
            try (var ctx = NumericalContext.errorOnInvalid()) {
                assertEquals(NaNPolicy.ERROR, NumericalContext.currentPolicy());
            }
        }

        @Test
        @DisplayName("convenience method warnOnInvalid")
        void convenienceMethodWarnOnInvalid() {
            try (var ctx = NumericalContext.warnOnInvalid()) {
                assertEquals(NaNPolicy.WARN, NumericalContext.currentPolicy());
            }
        }

        @Test
        @DisplayName("rejects null policy")
        void rejectsNullPolicy() {
            assertThrows(NullPointerException.class,
                    () -> new NumericalContext(null));
        }
    }

    // ==================== NaN Detection Tests ====================

    @Nested
    @DisplayName("NaN Detection")
    class NaNDetection {

        @Test
        @DisplayName("detects single NaN with ERROR policy")
        void detectsSingleNaNWithError() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = createTensorWithNaN()) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test_op"));

                assertEquals(NumericalException.InvalidValueType.NAN, ex.valueType());
                assertEquals("test_op", ex.operation());
            }
        }

        @Test
        @DisplayName("NaN at position 0")
        void nanAtPositionZero() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{Float.NaN, 1.0f, 2.0f},
                         new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(0, ex.position());
            }
        }

        @Test
        @DisplayName("NaN at last position")
        void nanAtLastPosition() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{1.0f, 2.0f, Float.NaN},
                         new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(2, ex.position());
            }
        }

        @Test
        @DisplayName("NaN in middle")
        void nanInMiddle() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{1.0f, 2.0f, Float.NaN, 4.0f, 5.0f},
                         new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(2, ex.position());
            }
        }

        @Test
        @DisplayName("tensor with all NaN")
        void allNaN() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{Float.NaN, Float.NaN, Float.NaN},
                         new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(0, ex.position());  // First NaN
            }
        }
    }

    // ==================== Infinity Detection Tests ====================

    @Nested
    @DisplayName("Infinity Detection")
    class InfDetection {

        @Test
        @DisplayName("detects +Inf")
        void detectsPositiveInf() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{1.0f, Float.POSITIVE_INFINITY, 2.0f},
                         new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(NumericalException.InvalidValueType.POSITIVE_INF, ex.valueType());
            }
        }

        @Test
        @DisplayName("detects -Inf")
        void detectsNegativeInf() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{1.0f, Float.NEGATIVE_INFINITY, 2.0f},
                         new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(NumericalException.InvalidValueType.NEGATIVE_INF, ex.valueType());
            }
        }

        @Test
        @DisplayName("mixed NaN and Inf detects first occurrence")
        void mixedNaNAndInf() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{1.0f, Float.POSITIVE_INFINITY, Float.NaN},
                         new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(1, ex.position());  // First invalid (Inf at position 1)
            }
        }
    }

    // ==================== Valid Values Tests ====================

    @Nested
    @DisplayName("Valid Values Pass")
    class ValidValuePass {

        @Test
        @DisplayName("normal values pass")
        void normalValuesPass() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                         new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }

        @Test
        @DisplayName("Float.MAX_VALUE passes")
        void maxValuePasses() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{Float.MAX_VALUE, -Float.MAX_VALUE},
                         new Vector(2), F32.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }

        @Test
        @DisplayName("Float.MIN_VALUE passes")
        void minValuePasses() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{Float.MIN_VALUE, -Float.MIN_VALUE},
                         new Vector(2), F32.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }

        @Test
        @DisplayName("subnormal values pass")
        void subnormalValuesPasses() {
            float subnormal = Float.MIN_VALUE / 2;  // Subnormal
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{subnormal, -subnormal},
                         new Vector(2), F32.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }

        @Test
        @DisplayName("zero values pass")
        void zeroValuesPasses() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{0.0f, -0.0f},
                         new Vector(2), F32.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }

        @Test
        @DisplayName("negative values pass")
        void negativeValuesPasses() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{-1.0f, -2.0f, -100.0f},
                         new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }
    }

    // ==================== Empty Tensor Tests ====================

    @Nested
    @DisplayName("Empty Tensor")
    class EmptyTensor {

        @Test
        @DisplayName("empty vector passes")
        void emptyVectorPasses() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                         new Vector(0), F32.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }
    }

    // ==================== Scalar Tensor Tests ====================

    @Nested
    @DisplayName("Scalar Tensor")
    class ScalarTensor {

        @Test
        @DisplayName("scalar with valid value passes")
        void scalarValidPasses() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Scalar, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{3.14f}, Scalar.INSTANCE, F32.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }

        @Test
        @DisplayName("scalar with NaN throws")
        void scalarNaNThrows() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Scalar, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{Float.NaN}, Scalar.INSTANCE, F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));
            }
        }
    }

    // ==================== F64 Tests ====================

    @Nested
    @DisplayName("F64 (Double) Tensor")
    class F64Tensor {

        @Test
        @DisplayName("detects double NaN")
        void detectsDoubleNaN() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F64, Cpu> t = TypedTensor.fromDoubleArray(
                         new double[]{1.0, Double.NaN, 3.0},
                         new Vector(3), F64.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(NumericalException.InvalidValueType.NAN, ex.valueType());
            }
        }

        @Test
        @DisplayName("detects double +Inf")
        void detectsDoublePositiveInf() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F64, Cpu> t = TypedTensor.fromDoubleArray(
                         new double[]{1.0, Double.POSITIVE_INFINITY, 3.0},
                         new Vector(3), F64.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(NumericalException.InvalidValueType.POSITIVE_INF, ex.valueType());
            }
        }

        @Test
        @DisplayName("detects double -Inf")
        void detectsDoubleNegativeInf() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F64, Cpu> t = TypedTensor.fromDoubleArray(
                         new double[]{1.0, Double.NEGATIVE_INFINITY, 3.0},
                         new Vector(3), F64.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(NumericalException.InvalidValueType.NEGATIVE_INF, ex.valueType());
            }
        }

        @Test
        @DisplayName("valid double passes")
        void validDoublePasses() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F64, Cpu> t = TypedTensor.fromDoubleArray(
                         new double[]{1.0, 2.0, 3.0},
                         new Vector(3), F64.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }
    }

    // ==================== Integer Tests ====================

    @Nested
    @DisplayName("Integer Tensor")
    class IntegerTensor {

        @Test
        @DisplayName("integer tensors never have NaN/Inf")
        void integerNeverHasNaN() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, I32, Cpu> t = TypedTensor.zeros(
                         new Vector(10), I32.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }
    }

    // ==================== Matrix Tests ====================

    @Nested
    @DisplayName("Matrix Tensor")
    class MatrixTensor {

        @Test
        @DisplayName("detects NaN in matrix")
        void detectsNaNInMatrix() {
            float[] data = new float[12];
            data[7] = Float.NaN;  // Position 7 in flat array

            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Matrix, F32, Cpu> t = TypedTensor.fromFloatArray(
                         data, new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertEquals(7, ex.position());
            }
        }
    }

    // ==================== WARN Policy Tests ====================

    @Nested
    @DisplayName("WARN Policy")
    class WarnPolicy {

        @Test
        @DisplayName("WARN policy doesn't throw")
        void warnDoesntThrow() {
            try (var ctx = NumericalContext.warnOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = createTensorWithNaN()) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }

        @Test
        @DisplayName("WARN policy logs but continues")
        void warnLogsContinues() {
            // We can't easily test logging, but we can verify it doesn't throw
            try (var ctx = NumericalContext.warnOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{Float.NaN, Float.POSITIVE_INFINITY},
                         new Vector(2), F32.INSTANCE, Cpu.INSTANCE)) {

                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
            }
        }
    }

    // ==================== Nested Context Tests ====================

    @Nested
    @DisplayName("Nested Contexts")
    class NestedContexts {

        @Test
        @DisplayName("nested contexts override correctly")
        void nestedContextsOverride() {
            assertEquals(NaNPolicy.IGNORE, NumericalContext.currentPolicy());

            try (var outer = new NumericalContext(NaNPolicy.ERROR)) {
                assertEquals(NaNPolicy.ERROR, NumericalContext.currentPolicy());

                try (var inner = new NumericalContext(NaNPolicy.WARN)) {
                    assertEquals(NaNPolicy.WARN, NumericalContext.currentPolicy());
                }

                assertEquals(NaNPolicy.ERROR, NumericalContext.currentPolicy());
            }

            assertEquals(NaNPolicy.IGNORE, NumericalContext.currentPolicy());
        }

        @Test
        @DisplayName("deeply nested contexts restore correctly")
        void deeplyNestedRestore() {
            try (var c1 = new NumericalContext(NaNPolicy.ERROR)) {
                try (var c2 = new NumericalContext(NaNPolicy.WARN)) {
                    try (var c3 = new NumericalContext(NaNPolicy.IGNORE)) {
                        try (var c4 = new NumericalContext(NaNPolicy.ERROR)) {
                            assertEquals(NaNPolicy.ERROR, NumericalContext.currentPolicy());
                        }
                        assertEquals(NaNPolicy.IGNORE, NumericalContext.currentPolicy());
                    }
                    assertEquals(NaNPolicy.WARN, NumericalContext.currentPolicy());
                }
                assertEquals(NaNPolicy.ERROR, NumericalContext.currentPolicy());
            }
            assertEquals(NaNPolicy.IGNORE, NumericalContext.currentPolicy());
        }
    }

    // ==================== containsNaNOrInf Tests ====================

    @Nested
    @DisplayName("containsNaNOrInf")
    class ContainsNaNOrInf {

        @Test
        @DisplayName("returns true for NaN")
        void returnsTrueForNaN() {
            try (TypedTensor<Vector, F32, Cpu> t = createTensorWithNaN()) {
                assertTrue(NumericalContext.containsNaNOrInf(t));
            }
        }

        @Test
        @DisplayName("returns true for Inf")
        void returnsTrueForInf() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{1.0f, Float.POSITIVE_INFINITY},
                    new Vector(2), F32.INSTANCE, Cpu.INSTANCE)) {

                assertTrue(NumericalContext.containsNaNOrInf(t));
            }
        }

        @Test
        @DisplayName("returns false for valid values")
        void returnsFalseForValid() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                    new float[]{1.0f, 2.0f, 3.0f},
                    new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                assertFalse(NumericalContext.containsNaNOrInf(t));
            }
        }

        @Test
        @DisplayName("works regardless of policy")
        void worksRegardlessOfPolicy() {
            try (TypedTensor<Vector, F32, Cpu> t = createTensorWithNaN()) {
                // Default is IGNORE, but containsNaNOrInf always checks
                assertTrue(NumericalContext.containsNaNOrInf(t));
            }
        }
    }

    // ==================== containsNaN Tests ====================

    @Nested
    @DisplayName("containsNaN")
    class ContainsNaN {

        @Test
        @DisplayName("returns true only for NaN, not Inf")
        void returnsTrueOnlyForNaN() {
            try (TypedTensor<Vector, F32, Cpu> nan = createTensorWithNaN();
                 TypedTensor<Vector, F32, Cpu> inf = TypedTensor.fromFloatArray(
                         new float[]{1.0f, Float.POSITIVE_INFINITY},
                         new Vector(2), F32.INSTANCE, Cpu.INSTANCE)) {

                assertTrue(NumericalContext.containsNaN(nan));
                assertFalse(NumericalContext.containsNaN(inf));
            }
        }
    }

    // ==================== containsInf Tests ====================

    @Nested
    @DisplayName("containsInf")
    class ContainsInf {

        @Test
        @DisplayName("returns true only for Inf, not NaN")
        void returnsTrueOnlyForInf() {
            try (TypedTensor<Vector, F32, Cpu> nan = createTensorWithNaN();
                 TypedTensor<Vector, F32, Cpu> inf = TypedTensor.fromFloatArray(
                         new float[]{1.0f, Float.POSITIVE_INFINITY},
                         new Vector(2), F32.INSTANCE, Cpu.INSTANCE)) {

                assertFalse(NumericalContext.containsInf(nan));
                assertTrue(NumericalContext.containsInf(inf));
            }
        }
    }

    // ==================== Thread Safety Tests ====================

    @Nested
    @DisplayName("Thread Safety")
    class ThreadSafety {

        @Test
        @DisplayName("different threads have different policies")
        void differentThreadsDifferentPolicies() throws InterruptedException {
            CountDownLatch latch = new CountDownLatch(2);
            AtomicReference<NaNPolicy> t1Policy = new AtomicReference<>();
            AtomicReference<NaNPolicy> t2Policy = new AtomicReference<>();

            Thread thread1 = new Thread(() -> {
                try (var ctx = new NumericalContext(NaNPolicy.ERROR)) {
                    t1Policy.set(NumericalContext.currentPolicy());
                    latch.countDown();
                    try {
                        latch.await();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    // Verify our policy wasn't changed
                    assertEquals(NaNPolicy.ERROR, NumericalContext.currentPolicy());
                }
            });

            Thread thread2 = new Thread(() -> {
                try (var ctx = new NumericalContext(NaNPolicy.WARN)) {
                    t2Policy.set(NumericalContext.currentPolicy());
                    latch.countDown();
                    try {
                        latch.await();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    // Verify our policy wasn't changed
                    assertEquals(NaNPolicy.WARN, NumericalContext.currentPolicy());
                }
            });

            thread1.start();
            thread2.start();
            thread1.join(5000);
            thread2.join(5000);

            assertEquals(NaNPolicy.ERROR, t1Policy.get());
            assertEquals(NaNPolicy.WARN, t2Policy.get());
        }
    }

    // ==================== Large Tensor Performance ====================

    @Nested
    @DisplayName("Large Tensor Performance")
    class LargeTensorPerformance {

        @Test
        @DisplayName("checking large tensor is performant")
        void largeTensorPerformant() {
            // 1 million elements
            float[] data = new float[1_000_000];
            for (int i = 0; i < data.length; i++) {
                data[i] = (float) Math.random();
            }

            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         data, new Vector(1_000_000), F32.INSTANCE, Cpu.INSTANCE)) {

                long start = System.nanoTime();
                assertDoesNotThrow(() -> NumericalContext.check(t, "test"));
                long elapsed = System.nanoTime() - start;

                // Should complete in reasonable time (under 1 second)
                assertTrue(elapsed < 1_000_000_000L,
                        "Took too long: " + elapsed / 1_000_000 + "ms");
            }
        }
    }

    // ==================== Exception Message Tests ====================

    @Nested
    @DisplayName("Exception Messages")
    class ExceptionMessages {

        @Test
        @DisplayName("exception message includes operation name")
        void exceptionIncludesOperation() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = createTensorWithNaN()) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "matmul_backward"));

                assertTrue(ex.getMessage().contains("matmul_backward"));
            }
        }

        @Test
        @DisplayName("exception message includes position")
        void exceptionIncludesPosition() {
            try (var ctx = NumericalContext.errorOnInvalid();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.fromFloatArray(
                         new float[]{1.0f, 2.0f, Float.NaN},
                         new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                NumericalException ex = assertThrows(NumericalException.class,
                        () -> NumericalContext.check(t, "test"));

                assertTrue(ex.getMessage().contains("2"));  // Position 2
            }
        }
    }

    // ==================== Close Behavior ====================

    @Nested
    @DisplayName("Close Behavior")
    class CloseBehavior {

        @Test
        @DisplayName("close is idempotent")
        void closeIsIdempotent() {
            NumericalContext ctx = new NumericalContext(NaNPolicy.ERROR);
            ctx.close();
            assertDoesNotThrow(() -> ctx.close());
        }

        @Test
        @DisplayName("isClosed reflects state")
        void isClosedReflectsState() {
            NumericalContext ctx = new NumericalContext(NaNPolicy.ERROR);
            assertFalse(ctx.isClosed());
            ctx.close();
            assertTrue(ctx.isClosed());
        }
    }

    // ==================== Helper Methods ====================

    private TypedTensor<Vector, F32, Cpu> createTensorWithNaN() {
        return TypedTensor.fromFloatArray(
                new float[]{1.0f, Float.NaN, 3.0f},
                new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
    }
}
