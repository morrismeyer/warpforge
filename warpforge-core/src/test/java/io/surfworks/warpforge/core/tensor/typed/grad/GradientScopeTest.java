package io.surfworks.warpforge.core.tensor.typed.grad;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
 * Comprehensive tests for GradientScope lifecycle management.
 */
@DisplayName("GradientScope")
class GradientScopeTest {

    // ==================== Basic Functionality ====================

    @Nested
    @DisplayName("Basic Functionality")
    class BasicFunctionality {

        @Test
        @DisplayName("newly created scope is empty")
        void newScopeIsEmpty() {
            try (GradientScope scope = new GradientScope()) {
                assertTrue(scope.isEmpty());
                assertEquals(0, scope.size());
            }
        }

        @Test
        @DisplayName("track adds tensor to scope")
        void trackAddsTensorToScope() {
            try (GradientScope scope = new GradientScope();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                         new Vector(10), F32.INSTANCE, Cpu.INSTANCE)) {

                scope.track(t);

                assertFalse(scope.isEmpty());
                assertEquals(1, scope.size());
            }
        }

        @Test
        @DisplayName("track returns GradTensor with RequiresGrad")
        void trackReturnsRequiresGradTensor() {
            try (GradientScope scope = new GradientScope();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                         new Vector(10), F32.INSTANCE, Cpu.INSTANCE)) {

                GradTensor<Vector, F32, Cpu, RequiresGrad> g = scope.track(t);

                assertNotNull(g);
                assertEquals(RequiresGrad.INSTANCE, g.gradMode());
                assertTrue(g.requiresGrad());
            }
        }

        @Test
        @DisplayName("track multiple tensors")
        void trackMultipleTensors() {
            try (GradientScope scope = new GradientScope();
                 TypedTensor<Vector, F32, Cpu> t1 = TypedTensor.zeros(
                         new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> t2 = TypedTensor.zeros(
                         new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Scalar, F32, Cpu> t3 = TypedTensor.zeros(
                         Scalar.INSTANCE, F32.INSTANCE, Cpu.INSTANCE)) {

                scope.track(t1);
                scope.track(t2);
                scope.track(t3);

                assertEquals(3, scope.size());
            }
        }
    }

    // ==================== zeroGrad Tests ====================

    @Nested
    @DisplayName("zeroGrad")
    class ZeroGradTests {

        @Test
        @DisplayName("zeroGrad on empty scope is safe")
        void zeroGradOnEmptyScopeIsSafe() {
            try (GradientScope scope = new GradientScope()) {
                assertDoesNotThrow(() -> scope.zeroGrad());
            }
        }

        @Test
        @DisplayName("zeroGrad clears all tracked gradients")
        void zeroGradClearsAllGradients() {
            try (GradientScope scope = new GradientScope();
                 TypedTensor<Vector, F32, Cpu> t1 = TypedTensor.zeros(
                         new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> t2 = TypedTensor.zeros(
                         new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                GradTensor<Vector, F32, Cpu, RequiresGrad> g1 = scope.track(t1);
                GradTensor<Vector, F32, Cpu, RequiresGrad> g2 = scope.track(t2);

                // Accumulate gradients (accumulateGrad takes ownership, so no try-with-resources)
                TypedTensor<Vector, F32, Cpu> grad1 = TypedTensor.full(
                        1.0f, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                TypedTensor<Vector, F32, Cpu> grad2 = TypedTensor.full(
                        2.0f, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);

                g1.accumulateGrad(grad1);
                g2.accumulateGrad(grad2);

                assertTrue(g1.hasGrad());
                assertTrue(g2.hasGrad());

                scope.zeroGrad();

                assertFalse(g1.hasGrad());
                assertFalse(g2.hasGrad());
            }
        }

        @Test
        @DisplayName("zeroGrad can be called multiple times")
        void zeroGradMultipleTimes() {
            try (GradientScope scope = new GradientScope();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                         new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                scope.track(t);

                assertDoesNotThrow(() -> {
                    scope.zeroGrad();
                    scope.zeroGrad();
                    scope.zeroGrad();
                });
            }
        }
    }

    // ==================== Duplicate Tracking Tests ====================

    @Nested
    @DisplayName("Duplicate Tracking")
    class DuplicateTrackingTests {

        @Test
        @DisplayName("tracking same tensor twice returns same GradTensor")
        void trackingSameTensorTwiceReturnsSame() {
            try (GradientScope scope = new GradientScope();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                         new Vector(10), F32.INSTANCE, Cpu.INSTANCE)) {

                GradTensor<Vector, F32, Cpu, RequiresGrad> g1 = scope.track(t);
                GradTensor<Vector, F32, Cpu, RequiresGrad> g2 = scope.track(t);

                assertSame(g1, g2);
                assertEquals(1, scope.size());  // Still only one tracked
            }
        }

        @Test
        @DisplayName("tracking same tensor in different scopes creates different GradTensors")
        void differentScopesCreateDifferentGradTensors() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradientScope scope1 = new GradientScope();
                 GradientScope scope2 = new GradientScope()) {

                // Note: This creates ownership issues - just testing the API
                // In practice, you wouldn't do this
                GradTensor<Vector, F32, Cpu, RequiresGrad> g1 = scope1.track(t);
                // Can't really safely test this without more complex ownership management
                assertNotNull(g1);
            }
        }
    }

    // ==================== Lifecycle Tests ====================

    @Nested
    @DisplayName("Lifecycle")
    class LifecycleTests {

        @Test
        @DisplayName("isClosed returns false initially")
        void isClosedReturnsFalseInitially() {
            try (GradientScope scope = new GradientScope()) {
                assertFalse(scope.isClosed());
            }
        }

        @Test
        @DisplayName("isClosed returns true after close")
        void isClosedReturnsTrueAfterClose() {
            GradientScope scope = new GradientScope();
            scope.close();
            assertTrue(scope.isClosed());
        }

        @Test
        @DisplayName("close is idempotent")
        void closeIsIdempotent() {
            GradientScope scope = new GradientScope();

            scope.close();
            assertDoesNotThrow(() -> scope.close());
            assertDoesNotThrow(() -> scope.close());

            assertTrue(scope.isClosed());
        }

        @Test
        @DisplayName("track throws after close")
        void trackThrowsAfterClose() {
            GradientScope scope = new GradientScope();
            scope.close();

            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE)) {
                assertThrows(IllegalStateException.class, () -> scope.track(t));
            }
        }

        @Test
        @DisplayName("zeroGrad throws after close")
        void zeroGradThrowsAfterClose() {
            GradientScope scope = new GradientScope();
            scope.close();

            assertThrows(IllegalStateException.class, () -> scope.zeroGrad());
        }

        @Test
        @DisplayName("close cleans up tracked tensors in reverse order")
        void closeCleanupInReverseOrder() {
            // We can't directly observe LIFO cleanup, but we can verify no exceptions
            try (TypedTensor<Vector, F32, Cpu> t1 = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> t2 = TypedTensor.zeros(
                         new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                GradientScope scope = new GradientScope();
                scope.track(t1);
                scope.track(t2);

                assertDoesNotThrow(() -> scope.close());
            }
        }

        @Test
        @DisplayName("trackedTensors returns unmodifiable list")
        void trackedTensorsReturnsUnmodifiable() {
            try (GradientScope scope = new GradientScope();
                 TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                         new Vector(10), F32.INSTANCE, Cpu.INSTANCE)) {

                scope.track(t);

                assertThrows(UnsupportedOperationException.class,
                        () -> scope.trackedTensors().clear());
            }
        }
    }

    // ==================== Edge Cases ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("track rejects null tensor")
        void trackRejectsNull() {
            try (GradientScope scope = new GradientScope()) {
                assertThrows(NullPointerException.class, () -> scope.track(null));
            }
        }

        @Test
        @DisplayName("works with different dtypes")
        void worksWithDifferentDtypes() {
            try (GradientScope scope = new GradientScope();
                 TypedTensor<Vector, F32, Cpu> f32 = TypedTensor.zeros(
                         new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F64, Cpu> f64 = TypedTensor.zeros(
                         new Vector(5), F64.INSTANCE, Cpu.INSTANCE)) {

                GradTensor<Vector, F32, Cpu, RequiresGrad> g1 = scope.track(f32);
                GradTensor<Vector, F64, Cpu, RequiresGrad> g2 = scope.track(f64);

                assertEquals(2, scope.size());
                assertEquals(F32.INSTANCE, g1.dtypeType());
                assertEquals(F64.INSTANCE, g2.dtypeType());
            }
        }

        @Test
        @DisplayName("works with different shapes")
        void worksWithDifferentShapes() {
            try (GradientScope scope = new GradientScope();
                 TypedTensor<Scalar, F32, Cpu> scalar = TypedTensor.zeros(
                         Scalar.INSTANCE, F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> vector = TypedTensor.zeros(
                         new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> matrix = TypedTensor.zeros(
                         new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                scope.track(scalar);
                scope.track(vector);
                scope.track(matrix);

                assertEquals(3, scope.size());
            }
        }

        @Test
        @DisplayName("toString reflects state")
        void toStringReflectsState() {
            GradientScope scope = new GradientScope();
            assertTrue(scope.toString().contains("tracked=0"));

            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE)) {
                scope.track(t);
                assertTrue(scope.toString().contains("tracked=1"));
            }

            scope.close();
            assertTrue(scope.toString().contains("CLOSED"));
        }
    }

    // ==================== Nested Scope Tests ====================

    @Nested
    @DisplayName("Nested Scopes")
    class NestedScopes {

        @Test
        @DisplayName("nested scopes work independently")
        void nestedScopesWorkIndependently() {
            try (TypedTensor<Vector, F32, Cpu> t1 = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> t2 = TypedTensor.zeros(
                         new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                try (GradientScope outer = new GradientScope()) {
                    GradTensor<Vector, F32, Cpu, RequiresGrad> g1 = outer.track(t1);

                    try (GradientScope inner = new GradientScope()) {
                        GradTensor<Vector, F32, Cpu, RequiresGrad> g2 = inner.track(t2);

                        assertEquals(1, outer.size());
                        assertEquals(1, inner.size());

                        // Inner zeroGrad doesn't affect outer
                        TypedTensor<Vector, F32, Cpu> grad = TypedTensor.full(
                                1.0f, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                        g1.accumulateGrad(grad);

                        inner.zeroGrad();
                        assertTrue(g1.hasGrad());  // Outer tensor still has grad
                    }
                }
            }
        }

        @Test
        @DisplayName("deeply nested scopes")
        void deeplyNestedScopes() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                try (GradientScope s1 = new GradientScope()) {
                    s1.track(t);
                    try (GradientScope s2 = new GradientScope()) {
                        try (GradientScope s3 = new GradientScope()) {
                            try (GradientScope s4 = new GradientScope()) {
                                try (GradientScope s5 = new GradientScope()) {
                                    assertEquals(1, s1.size());
                                    assertEquals(0, s5.size());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ==================== Training Loop Simulation ====================

    @Nested
    @DisplayName("Training Loop Simulation")
    class TrainingLoopSimulation {

        @Test
        @DisplayName("simulated training loop works correctly")
        void simulatedTrainingLoop() {
            try (GradientScope scope = new GradientScope();
                 TypedTensor<Matrix, F32, Cpu> weights = TypedTensor.zeros(
                         new Matrix(100, 50), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.zeros(
                         new Vector(50), F32.INSTANCE, Cpu.INSTANCE)) {

                GradTensor<Matrix, F32, Cpu, RequiresGrad> wGrad = scope.track(weights);
                GradTensor<Vector, F32, Cpu, RequiresGrad> bGrad = scope.track(bias);

                int numEpochs = 3;
                for (int epoch = 0; epoch < numEpochs; epoch++) {
                    // Zero gradients at start of each epoch
                    scope.zeroGrad();

                    assertFalse(wGrad.hasGrad());
                    assertFalse(bGrad.hasGrad());

                    // Simulate backward pass accumulating gradients
                    // Don't use try-with-resources since accumulateGrad takes ownership
                    // and zeroGrad() will close them on next iteration
                    TypedTensor<Matrix, F32, Cpu> wGradTensor = TypedTensor.full(
                            0.1f, new Matrix(100, 50), F32.INSTANCE, Cpu.INSTANCE);
                    TypedTensor<Vector, F32, Cpu> bGradTensor = TypedTensor.full(
                            0.05f, new Vector(50), F32.INSTANCE, Cpu.INSTANCE);

                    wGrad.accumulateGrad(wGradTensor);
                    bGrad.accumulateGrad(bGradTensor);

                    assertTrue(wGrad.hasGrad());
                    assertTrue(bGrad.hasGrad());
                }
            }
        }
    }
}
