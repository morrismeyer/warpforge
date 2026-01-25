package io.surfworks.warpforge.core.tensor.typed.grad;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.stream.Stream;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.dtype.I32;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Scalar;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Comprehensive tests for GradTensor wrapper class.
 */
@DisplayName("GradTensor")
class GradTensorTest {

    // ==================== Factory Method Tests ====================

    @Nested
    @DisplayName("Factory Methods")
    class FactoryMethods {

        @Test
        @DisplayName("requiresGrad creates tensor with RequiresGrad mode")
        void requiresGradCreatesCorrectMode() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertEquals(RequiresGrad.INSTANCE, g.gradMode());
                assertTrue(g.requiresGrad());
            }
        }

        @Test
        @DisplayName("noGrad creates tensor with NoGrad mode")
        void noGradCreatesCorrectMode() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, NoGrad> g = GradTensor.noGrad(t)) {

                assertEquals(NoGrad.INSTANCE, g.gradMode());
                assertFalse(g.requiresGrad());
            }
        }

        @Test
        @DisplayName("requiresGrad rejects null tensor")
        void requiresGradRejectsNull() {
            assertThrows(NullPointerException.class,
                    () -> GradTensor.requiresGrad(null));
        }

        @Test
        @DisplayName("noGrad rejects null tensor")
        void noGradRejectsNull() {
            assertThrows(NullPointerException.class,
                    () -> GradTensor.noGrad(null));
        }

        @Test
        @DisplayName("requiresGradOwning takes ownership")
        void requiresGradOwningTakesOwnership() {
            TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);

            GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGradOwning(t);
            g.close();

            // After closing GradTensor, underlying should be closed too
            // We can't directly test this without internal access, but at least verify no exception
            assertDoesNotThrow(() -> g.toString());
        }
    }

    // ==================== Accessor Tests ====================

    @Nested
    @DisplayName("Accessors")
    class Accessors {

        @Test
        @DisplayName("data() returns underlying tensor")
        void dataReturnsUnderlying() {
            try (TypedTensor<Matrix, F32, Cpu> t = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Matrix, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertSame(t, g.data());
            }
        }

        @Test
        @DisplayName("shapeType() returns correct shape")
        void shapeTypeReturnsCorrectShape() {
            Matrix shape = new Matrix(5, 6);
            try (TypedTensor<Matrix, F32, Cpu> t = TypedTensor.zeros(
                    shape, F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Matrix, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertEquals(shape, g.shapeType());
            }
        }

        @Test
        @DisplayName("dtypeType() returns correct dtype")
        void dtypeTypeReturnsCorrectDtype() {
            try (TypedTensor<Vector, F64, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F64.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F64, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertEquals(F64.INSTANCE, g.dtypeType());
            }
        }

        @Test
        @DisplayName("deviceType() returns correct device")
        void deviceTypeReturnsCorrectDevice() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertEquals(Cpu.INSTANCE, g.deviceType());
            }
        }

        @Test
        @DisplayName("data() throws on closed tensor")
        void dataThrowsOnClosed() {
            TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
            GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
            g.close();

            assertThrows(IllegalStateException.class, () -> g.data());
            t.close();
        }
    }

    // ==================== Gradient Operations Tests ====================

    @Nested
    @DisplayName("Gradient Operations")
    class GradientOperations {

        @Test
        @DisplayName("grad() returns null before accumulation")
        void gradReturnsNullInitially() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertNull(g.grad());
                assertFalse(g.hasGrad());
            }
        }

        @Test
        @DisplayName("accumulateGrad stores gradient")
        void accumulateGradStoresGradient() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                // Don't put grad in try-with-resources since accumulateGrad takes ownership
                TypedTensor<Vector, F32, Cpu> grad = TypedTensor.full(
                    1.0f, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                g.accumulateGrad(grad);

                assertTrue(g.hasGrad());
                assertNotNull(g.grad());
            }
        }

        @Test
        @DisplayName("zeroGrad clears gradient")
        void zeroGradClearsGradient() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                // Don't put grad in try-with-resources since accumulateGrad takes ownership
                TypedTensor<Vector, F32, Cpu> grad = TypedTensor.full(
                    1.0f, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                g.accumulateGrad(grad);
                assertTrue(g.hasGrad());

                g.zeroGrad();

                assertFalse(g.hasGrad());
                assertNull(g.grad());
            }
        }

        @Test
        @DisplayName("zeroGrad is safe when no gradient exists")
        void zeroGradSafeWithNoGradient() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertDoesNotThrow(() -> g.zeroGrad());
            }
        }

        @Test
        @DisplayName("grad() throws on NoGrad tensor")
        void gradThrowsOnNoGrad() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, NoGrad> g = GradTensor.noGrad(t)) {

                assertThrows(IllegalStateException.class, () -> g.grad());
            }
        }

        @Test
        @DisplayName("zeroGrad() throws on NoGrad tensor")
        void zeroGradThrowsOnNoGrad() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, NoGrad> g = GradTensor.noGrad(t)) {

                assertThrows(IllegalStateException.class, () -> g.zeroGrad());
            }
        }

        @Test
        @DisplayName("hasGrad() throws on NoGrad tensor")
        void hasGradThrowsOnNoGrad() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, NoGrad> g = GradTensor.noGrad(t)) {

                assertThrows(IllegalStateException.class, () -> g.hasGrad());
            }
        }

        @Test
        @DisplayName("accumulateGrad() throws on NoGrad tensor")
        void accumulateGradThrowsOnNoGrad() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> grad = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, NoGrad> g = GradTensor.noGrad(t)) {

                assertThrows(IllegalStateException.class, () -> g.accumulateGrad(grad));
            }
        }

        @Test
        @DisplayName("accumulateGrad rejects null")
        void accumulateGradRejectsNull() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertThrows(NullPointerException.class, () -> g.accumulateGrad(null));
            }
        }

        @Test
        @DisplayName("gradient operations throw on closed tensor")
        void gradientOpsThrowOnClosed() {
            TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
            GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
            g.close();

            assertThrows(IllegalStateException.class, () -> g.grad());
            assertThrows(IllegalStateException.class, () -> g.zeroGrad());
            assertThrows(IllegalStateException.class, () -> g.hasGrad());

            t.close();
        }
    }

    // ==================== Mode Transition Tests ====================

    @Nested
    @DisplayName("Mode Transitions")
    class ModeTransitions {

        @Test
        @DisplayName("detach() creates Detached tensor")
        void detachCreatesDetachedTensor() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
                 GradTensor<Vector, F32, Cpu, Detached> d = g.detach()) {

                assertEquals(Detached.INSTANCE, d.gradMode());
                assertFalse(d.requiresGrad());
            }
        }

        @Test
        @DisplayName("detached tensor shares data with original")
        void detachedSharesData() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
                 GradTensor<Vector, F32, Cpu, Detached> d = g.detach()) {

                assertSame(g.data(), d.data());
            }
        }

        @Test
        @DisplayName("grad() throws on Detached tensor")
        void gradThrowsOnDetached() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
                 GradTensor<Vector, F32, Cpu, Detached> d = g.detach()) {

                assertThrows(IllegalStateException.class, () -> d.grad());
            }
        }

        @Test
        @DisplayName("zeroGrad() throws on Detached tensor")
        void zeroGradThrowsOnDetached() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
                 GradTensor<Vector, F32, Cpu, Detached> d = g.detach()) {

                assertThrows(IllegalStateException.class, () -> d.zeroGrad());
            }
        }

        @Test
        @DisplayName("double detach is idempotent")
        void doubleDetachIsIdempotent() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
                 GradTensor<Vector, F32, Cpu, Detached> d1 = g.detach();
                 GradTensor<Vector, F32, Cpu, Detached> d2 = d1.detach()) {

                assertEquals(Detached.INSTANCE, d2.gradMode());
                assertSame(d1.data(), d2.data());
            }
        }

        @Test
        @DisplayName("asNoGrad creates NoGrad view")
        void asNoGradCreatesNoGradView() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
                 GradTensor<Vector, F32, Cpu, NoGrad> n = g.asNoGrad()) {

                assertEquals(NoGrad.INSTANCE, n.gradMode());
                assertSame(g.data(), n.data());
            }
        }

        @Test
        @DisplayName("detach() throws on closed tensor")
        void detachThrowsOnClosed() {
            TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
            GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
            g.close();

            assertThrows(IllegalStateException.class, () -> g.detach());
            t.close();
        }
    }

    // ==================== Lifecycle Tests ====================

    @Nested
    @DisplayName("Lifecycle")
    class Lifecycle {

        @Test
        @DisplayName("isClosed returns false initially")
        void isClosedReturnsFalseInitially() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertFalse(g.isClosed());
            }
        }

        @Test
        @DisplayName("isClosed returns true after close")
        void isClosedReturnsTrueAfterClose() {
            TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
            GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);

            g.close();

            assertTrue(g.isClosed());
            t.close();
        }

        @Test
        @DisplayName("close is idempotent")
        void closeIsIdempotent() {
            TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
            GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);

            g.close();
            assertDoesNotThrow(() -> g.close());
            assertDoesNotThrow(() -> g.close());

            assertTrue(g.isClosed());
            t.close();
        }

        @Test
        @DisplayName("close releases gradient memory")
        void closeReleasesGradientMemory() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);

                TypedTensor<Vector, F32, Cpu> grad = TypedTensor.full(
                        1.0f, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                g.accumulateGrad(grad);

                g.close();

                // Gradient should be released (we can't directly verify but no exception)
                assertTrue(g.isClosed());
            }
        }

        @Test
        @DisplayName("toString works on closed tensor")
        void toStringWorksOnClosed() {
            TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
            GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
            g.close();

            String str = g.toString();
            assertTrue(str.contains("CLOSED"));

            t.close();
        }

        @Test
        @DisplayName("toString includes mode and shape info")
        void toStringIncludesInfo() {
            try (TypedTensor<Matrix, F32, Cpu> t = TypedTensor.zeros(
                    new Matrix(3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Matrix, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                String str = g.toString();
                assertTrue(str.contains("requires_grad"));
                assertTrue(str.contains("Matrix"));
            }
        }
    }

    // ==================== Combinatoric Tests ====================

    @Nested
    @DisplayName("Combinatoric Tests - All Shape/DType Combinations")
    class CombinatorialTests {

        static Stream<Arguments> shapeProvider() {
            return Stream.of(
                    Arguments.of(Scalar.INSTANCE, "Scalar"),
                    Arguments.of(new Vector(10), "Vector"),
                    Arguments.of(new Matrix(3, 4), "Matrix")
            );
        }

        static Stream<Arguments> dtypeProvider() {
            return Stream.of(
                    Arguments.of(F32.INSTANCE, "F32"),
                    Arguments.of(F64.INSTANCE, "F64"),
                    Arguments.of(I32.INSTANCE, "I32")
            );
        }

        static Stream<Arguments> shapeAndDtypeCombinations() {
            return Stream.of(
                    // Scalar combinations
                    Arguments.of(Scalar.INSTANCE, F32.INSTANCE),
                    Arguments.of(Scalar.INSTANCE, F64.INSTANCE),
                    Arguments.of(Scalar.INSTANCE, I32.INSTANCE),
                    // Vector combinations
                    Arguments.of(new Vector(5), F32.INSTANCE),
                    Arguments.of(new Vector(5), F64.INSTANCE),
                    Arguments.of(new Vector(5), I32.INSTANCE),
                    Arguments.of(new Vector(0), F32.INSTANCE),  // Empty vector
                    Arguments.of(new Vector(1), F32.INSTANCE),  // Single element
                    // Matrix combinations
                    Arguments.of(new Matrix(2, 3), F32.INSTANCE),
                    Arguments.of(new Matrix(2, 3), F64.INSTANCE),
                    Arguments.of(new Matrix(2, 3), I32.INSTANCE),
                    Arguments.of(new Matrix(1, 1), F32.INSTANCE),  // Single element
                    Arguments.of(new Matrix(10, 10), F32.INSTANCE)  // Square
            );
        }

        @ParameterizedTest(name = "{0} with {1}")
        @MethodSource("shapeAndDtypeCombinations")
        @DisplayName("requiresGrad works with all shape/dtype combinations")
        void requiresGradWorksWithAllCombinations(Shape shape, DTypeTag dtype) {
            try (TypedTensor<Shape, DTypeTag, Cpu> t = createTensor(shape, dtype);
                 GradTensor<Shape, DTypeTag, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertEquals(RequiresGrad.INSTANCE, g.gradMode());
                assertTrue(g.requiresGrad());
                assertFalse(g.hasGrad());
            }
        }

        @ParameterizedTest(name = "{0} with {1}")
        @MethodSource("shapeAndDtypeCombinations")
        @DisplayName("noGrad works with all shape/dtype combinations")
        void noGradWorksWithAllCombinations(Shape shape, DTypeTag dtype) {
            try (TypedTensor<Shape, DTypeTag, Cpu> t = createTensor(shape, dtype);
                 GradTensor<Shape, DTypeTag, Cpu, NoGrad> g = GradTensor.noGrad(t)) {

                assertEquals(NoGrad.INSTANCE, g.gradMode());
                assertFalse(g.requiresGrad());
            }
        }

        @ParameterizedTest(name = "{0} with {1}")
        @MethodSource("shapeAndDtypeCombinations")
        @DisplayName("detach works with all shape/dtype combinations")
        void detachWorksWithAllCombinations(Shape shape, DTypeTag dtype) {
            try (TypedTensor<Shape, DTypeTag, Cpu> t = createTensor(shape, dtype);
                 GradTensor<Shape, DTypeTag, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t);
                 GradTensor<Shape, DTypeTag, Cpu, Detached> d = g.detach()) {

                assertEquals(Detached.INSTANCE, d.gradMode());
                assertSame(g.data(), d.data());
            }
        }

        @SuppressWarnings("unchecked")
        private <S extends Shape, D extends DTypeTag>
        TypedTensor<S, D, Cpu> createTensor(S shape, D dtype) {
            // Create appropriate tensor based on dtype
            if (dtype instanceof F32) {
                return (TypedTensor<S, D, Cpu>) TypedTensor.zeros(shape, F32.INSTANCE, Cpu.INSTANCE);
            } else if (dtype instanceof F64) {
                return (TypedTensor<S, D, Cpu>) TypedTensor.zeros(shape, F64.INSTANCE, Cpu.INSTANCE);
            } else if (dtype instanceof I32) {
                return (TypedTensor<S, D, Cpu>) TypedTensor.zeros(shape, I32.INSTANCE, Cpu.INSTANCE);
            }
            throw new IllegalArgumentException("Unsupported dtype: " + dtype);
        }
    }

    // ==================== Edge Case Tests ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("works with scalar tensor")
        void worksWithScalar() {
            try (TypedTensor<Scalar, F32, Cpu> t = TypedTensor.zeros(
                    Scalar.INSTANCE, F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Scalar, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertEquals(0, g.shapeType().rank());
                assertNull(g.grad());
            }
        }

        @Test
        @DisplayName("works with empty vector")
        void worksWithEmptyVector() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(0), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertEquals(0, g.shapeType().length());
                assertNull(g.grad());
            }
        }

        @Test
        @DisplayName("works with large tensor")
        void worksWithLargeTensor() {
            try (TypedTensor<Matrix, F32, Cpu> t = TypedTensor.zeros(
                    new Matrix(1000, 1000), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Matrix, F32, Cpu, RequiresGrad> g = GradTensor.requiresGrad(t)) {

                assertEquals(1000 * 1000, g.data().elementCount());
            }
        }

        @Test
        @DisplayName("multiple GradTensors can wrap same underlying")
        void multipleGradTensorsCanWrapSameUnderlying() {
            try (TypedTensor<Vector, F32, Cpu> t = TypedTensor.zeros(
                    new Vector(10), F32.INSTANCE, Cpu.INSTANCE);
                 GradTensor<Vector, F32, Cpu, RequiresGrad> g1 = GradTensor.requiresGrad(t);
                 GradTensor<Vector, F32, Cpu, NoGrad> g2 = GradTensor.noGrad(t)) {

                assertSame(g1.data(), g2.data());
            }
        }
    }
}
