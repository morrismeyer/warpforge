package io.surfworks.warpforge.core.tensor.typed.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Tests for GradMode phantom types.
 */
@DisplayName("GradMode")
class GradModeTest {

    @Nested
    @DisplayName("RequiresGrad")
    class RequiresGradTests {

        @Test
        @DisplayName("tracksGradient returns true")
        void tracksGradientReturnsTrue() {
            assertTrue(RequiresGrad.INSTANCE.tracksGradient());
        }

        @Test
        @DisplayName("modeName returns requires_grad")
        void modeNameReturnsCorrectValue() {
            assertEquals("requires_grad", RequiresGrad.INSTANCE.modeName());
        }

        @Test
        @DisplayName("INSTANCE singleton is consistent")
        void instanceIsSingleton() {
            assertSame(RequiresGrad.INSTANCE, RequiresGrad.INSTANCE);
            assertNotNull(RequiresGrad.INSTANCE);
        }

        @Test
        @DisplayName("record equality works")
        void recordEqualityWorks() {
            RequiresGrad a = new RequiresGrad();
            RequiresGrad b = new RequiresGrad();
            assertEquals(a, b);
            assertEquals(a.hashCode(), b.hashCode());
        }
    }

    @Nested
    @DisplayName("NoGrad")
    class NoGradTests {

        @Test
        @DisplayName("tracksGradient returns false")
        void tracksGradientReturnsFalse() {
            assertFalse(NoGrad.INSTANCE.tracksGradient());
        }

        @Test
        @DisplayName("modeName returns no_grad")
        void modeNameReturnsCorrectValue() {
            assertEquals("no_grad", NoGrad.INSTANCE.modeName());
        }

        @Test
        @DisplayName("INSTANCE singleton is consistent")
        void instanceIsSingleton() {
            assertSame(NoGrad.INSTANCE, NoGrad.INSTANCE);
            assertNotNull(NoGrad.INSTANCE);
        }

        @Test
        @DisplayName("record equality works")
        void recordEqualityWorks() {
            NoGrad a = new NoGrad();
            NoGrad b = new NoGrad();
            assertEquals(a, b);
            assertEquals(a.hashCode(), b.hashCode());
        }
    }

    @Nested
    @DisplayName("Detached")
    class DetachedTests {

        @Test
        @DisplayName("tracksGradient returns false")
        void tracksGradientReturnsFalse() {
            assertFalse(Detached.INSTANCE.tracksGradient());
        }

        @Test
        @DisplayName("modeName returns detached")
        void modeNameReturnsCorrectValue() {
            assertEquals("detached", Detached.INSTANCE.modeName());
        }

        @Test
        @DisplayName("INSTANCE singleton is consistent")
        void instanceIsSingleton() {
            assertSame(Detached.INSTANCE, Detached.INSTANCE);
            assertNotNull(Detached.INSTANCE);
        }

        @Test
        @DisplayName("record equality works")
        void recordEqualityWorks() {
            Detached a = new Detached();
            Detached b = new Detached();
            assertEquals(a, b);
            assertEquals(a.hashCode(), b.hashCode());
        }
    }

    @Nested
    @DisplayName("Sealed Interface Behavior")
    class SealedInterfaceTests {

        @Test
        @DisplayName("all modes implement GradMode")
        void allModesImplementGradMode() {
            GradMode requiresGrad = RequiresGrad.INSTANCE;
            GradMode noGrad = NoGrad.INSTANCE;
            GradMode detached = Detached.INSTANCE;

            assertNotNull(requiresGrad);
            assertNotNull(noGrad);
            assertNotNull(detached);
        }

        @Test
        @DisplayName("can use pattern matching on GradMode")
        void patternMatchingWorks() {
            GradMode mode = RequiresGrad.INSTANCE;
            String result = switch (mode) {
                case RequiresGrad r -> "requires";
                case NoGrad n -> "no";
                case Detached d -> "detached";
            };
            assertEquals("requires", result);
        }

        @Test
        @DisplayName("can distinguish modes via tracksGradient")
        void canDistinguishModes() {
            assertTrue(RequiresGrad.INSTANCE.tracksGradient());
            assertFalse(NoGrad.INSTANCE.tracksGradient());
            assertFalse(Detached.INSTANCE.tracksGradient());
        }
    }
}
