package io.surfworks.warpforge.core.tensor.typed.training;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Comprehensive tests for ModeContext thread-local mode switching.
 */
@DisplayName("ModeContext")
class ModeContextTest {

    @BeforeEach
    void resetToDefault() {
        // Ensure clean state before each test
        // Default should be Training
    }

    @AfterEach
    void cleanup() {
        // Ensure we're back to default state
        // In case a test leaks a context
        while (!ModeContext.isTraining()) {
            try (var ctx = new ModeContext(Training.INSTANCE)) {
                // This will reset if we're in a weird state
            }
            break;
        }
    }

    // ==================== Mode Interface Tests ====================

    @Nested
    @DisplayName("Mode Interface")
    class ModeInterfaceTests {

        @Test
        @DisplayName("Training.INSTANCE is training mode")
        void trainingIsTrainingMode() {
            assertTrue(Training.INSTANCE.isTraining());
            assertEquals("training", Training.INSTANCE.modeName());
        }

        @Test
        @DisplayName("Inference.INSTANCE is inference mode")
        void inferenceIsInferenceMode() {
            assertFalse(Inference.INSTANCE.isTraining());
            assertEquals("inference", Inference.INSTANCE.modeName());
        }

        @Test
        @DisplayName("singletons are consistent")
        void singletonsAreConsistent() {
            assertSame(Training.INSTANCE, Training.INSTANCE);
            assertSame(Inference.INSTANCE, Inference.INSTANCE);
            assertNotNull(Training.INSTANCE);
            assertNotNull(Inference.INSTANCE);
        }

        @Test
        @DisplayName("pattern matching works on Mode")
        void patternMatchingWorks() {
            Mode mode = Training.INSTANCE;
            String result = switch (mode) {
                case Training t -> "train";
                case Inference i -> "eval";
            };
            assertEquals("train", result);
        }
    }

    // ==================== Default State Tests ====================

    @Nested
    @DisplayName("Default State")
    class DefaultStateTests {

        @Test
        @DisplayName("default mode is Training")
        void defaultModeIsTraining() {
            assertEquals(Training.INSTANCE, ModeContext.current());
            assertTrue(ModeContext.isTraining());
            assertFalse(ModeContext.isInference());
        }

        @Test
        @DisplayName("isTraining returns true by default")
        void isTrainingDefaultsTrue() {
            assertTrue(ModeContext.isTraining());
        }

        @Test
        @DisplayName("isInference returns false by default")
        void isInferenceDefaultsFalse() {
            assertFalse(ModeContext.isInference());
        }
    }

    // ==================== Basic Context Switching ====================

    @Nested
    @DisplayName("Basic Context Switching")
    class BasicContextSwitching {

        @Test
        @DisplayName("switching to Inference mode")
        void switchingToInference() {
            try (var ctx = new ModeContext(Inference.INSTANCE)) {
                assertFalse(ModeContext.isTraining());
                assertTrue(ModeContext.isInference());
                assertEquals(Inference.INSTANCE, ModeContext.current());
            }
        }

        @Test
        @DisplayName("mode restores after context closes")
        void modeRestoresAfterClose() {
            assertTrue(ModeContext.isTraining());

            try (var ctx = new ModeContext(Inference.INSTANCE)) {
                assertFalse(ModeContext.isTraining());
            }

            assertTrue(ModeContext.isTraining());
        }

        @Test
        @DisplayName("convenience method inference()")
        void convenienceMethodInference() {
            try (var ctx = ModeContext.inference()) {
                assertTrue(ModeContext.isInference());
            }
        }

        @Test
        @DisplayName("convenience method training()")
        void convenienceMethodTraining() {
            try (var ctx = ModeContext.training()) {
                assertTrue(ModeContext.isTraining());
            }
        }

        @Test
        @DisplayName("rejects null mode")
        void rejectsNullMode() {
            assertThrows(NullPointerException.class,
                    () -> new ModeContext(null));
        }
    }

    // ==================== Nested Context Tests ====================

    @Nested
    @DisplayName("Nested Contexts")
    class NestedContexts {

        @Test
        @DisplayName("Training -> Inference -> Training nesting")
        void trainingInferenceTrainingNesting() {
            assertTrue(ModeContext.isTraining());

            try (var outer = new ModeContext(Inference.INSTANCE)) {
                assertFalse(ModeContext.isTraining());

                try (var inner = new ModeContext(Training.INSTANCE)) {
                    assertTrue(ModeContext.isTraining());
                }

                assertFalse(ModeContext.isTraining());
            }

            assertTrue(ModeContext.isTraining());
        }

        @Test
        @DisplayName("deep nesting restores correctly")
        void deepNestingRestoresCorrectly() {
            assertTrue(ModeContext.isTraining());

            try (var c1 = new ModeContext(Inference.INSTANCE)) {
                assertFalse(ModeContext.isTraining());

                try (var c2 = new ModeContext(Training.INSTANCE)) {
                    assertTrue(ModeContext.isTraining());

                    try (var c3 = new ModeContext(Inference.INSTANCE)) {
                        assertFalse(ModeContext.isTraining());

                        try (var c4 = new ModeContext(Training.INSTANCE)) {
                            assertTrue(ModeContext.isTraining());

                            try (var c5 = new ModeContext(Inference.INSTANCE)) {
                                assertFalse(ModeContext.isTraining());
                            }

                            assertTrue(ModeContext.isTraining());
                        }

                        assertFalse(ModeContext.isTraining());
                    }

                    assertTrue(ModeContext.isTraining());
                }

                assertFalse(ModeContext.isTraining());
            }

            assertTrue(ModeContext.isTraining());
        }

        @Test
        @DisplayName("10 levels of nesting")
        void tenLevelsOfNesting() {
            Mode[] expected = new Mode[10];
            for (int i = 0; i < 10; i++) {
                expected[i] = i % 2 == 0 ? Inference.INSTANCE : Training.INSTANCE;
            }

            try (var c0 = new ModeContext(expected[0])) {
                try (var c1 = new ModeContext(expected[1])) {
                    try (var c2 = new ModeContext(expected[2])) {
                        try (var c3 = new ModeContext(expected[3])) {
                            try (var c4 = new ModeContext(expected[4])) {
                                try (var c5 = new ModeContext(expected[5])) {
                                    try (var c6 = new ModeContext(expected[6])) {
                                        try (var c7 = new ModeContext(expected[7])) {
                                            try (var c8 = new ModeContext(expected[8])) {
                                                try (var c9 = new ModeContext(expected[9])) {
                                                    assertEquals(expected[9], ModeContext.current());
                                                }
                                                assertEquals(expected[8], ModeContext.current());
                                            }
                                            assertEquals(expected[7], ModeContext.current());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            assertTrue(ModeContext.isTraining());  // Back to default
        }
    }

    // ==================== Exception Safety Tests ====================

    @Nested
    @DisplayName("Exception Safety")
    class ExceptionSafety {

        @Test
        @DisplayName("mode restored when exception thrown in context")
        void modeRestoredOnException() {
            assertTrue(ModeContext.isTraining());

            try {
                try (var ctx = new ModeContext(Inference.INSTANCE)) {
                    assertFalse(ModeContext.isTraining());
                    throw new RuntimeException("test exception");
                }
            } catch (RuntimeException e) {
                // Expected
            }

            assertTrue(ModeContext.isTraining());
        }

        @Test
        @DisplayName("nested exception restores all levels")
        void nestedExceptionRestoresAllLevels() {
            assertTrue(ModeContext.isTraining());

            try {
                try (var c1 = new ModeContext(Inference.INSTANCE)) {
                    try (var c2 = new ModeContext(Training.INSTANCE)) {
                        try (var c3 = new ModeContext(Inference.INSTANCE)) {
                            throw new RuntimeException("test");
                        }
                    }
                }
            } catch (RuntimeException e) {
                // Expected
            }

            assertTrue(ModeContext.isTraining());
        }
    }

    // ==================== Thread Safety Tests ====================

    @Nested
    @DisplayName("Thread Safety")
    class ThreadSafety {

        @Test
        @DisplayName("each thread has its own mode")
        void eachThreadHasOwnMode() throws InterruptedException {
            AtomicBoolean thread1Training = new AtomicBoolean(false);
            AtomicBoolean thread2Training = new AtomicBoolean(false);
            CountDownLatch latch = new CountDownLatch(2);

            Thread t1 = new Thread(() -> {
                try (var ctx = new ModeContext(Training.INSTANCE)) {
                    thread1Training.set(ModeContext.isTraining());
                    latch.countDown();
                    try {
                        latch.await();  // Wait for other thread to set its mode
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    // Verify our mode wasn't affected
                    assertTrue(ModeContext.isTraining());
                }
            });

            Thread t2 = new Thread(() -> {
                try (var ctx = new ModeContext(Inference.INSTANCE)) {
                    thread2Training.set(ModeContext.isTraining());
                    latch.countDown();
                    try {
                        latch.await();  // Wait for other thread to set its mode
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    // Verify our mode wasn't affected
                    assertFalse(ModeContext.isTraining());
                }
            });

            t1.start();
            t2.start();
            t1.join(5000);
            t2.join(5000);

            assertTrue(thread1Training.get());
            assertFalse(thread2Training.get());
        }

        @Test
        @DisplayName("concurrent mode switches don't interfere")
        void concurrentModeSwitchesDontInterfere() throws InterruptedException {
            int numThreads = 10;
            int numIterations = 100;
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch doneLatch = new CountDownLatch(numThreads);
            AtomicReference<Throwable> error = new AtomicReference<>();

            for (int i = 0; i < numThreads; i++) {
                final int threadId = i;
                executor.submit(() -> {
                    try {
                        startLatch.await();

                        for (int j = 0; j < numIterations; j++) {
                            Mode mode = (threadId + j) % 2 == 0 ?
                                    Training.INSTANCE : Inference.INSTANCE;

                            try (var ctx = new ModeContext(mode)) {
                                // Verify the mode is what we set
                                if (ModeContext.current() != mode) {
                                    error.set(new AssertionError(
                                            "Mode mismatch in thread " + threadId));
                                }

                                // Do some nested switches
                                Mode innerMode = mode == Training.INSTANCE ?
                                        Inference.INSTANCE : Training.INSTANCE;
                                try (var inner = new ModeContext(innerMode)) {
                                    if (ModeContext.current() != innerMode) {
                                        error.set(new AssertionError(
                                                "Inner mode mismatch in thread " + threadId));
                                    }
                                }

                                // After inner closes, should be back to outer mode
                                if (ModeContext.current() != mode) {
                                    error.set(new AssertionError(
                                            "Mode not restored in thread " + threadId));
                                }
                            }
                        }
                    } catch (Exception e) {
                        error.set(e);
                    } finally {
                        doneLatch.countDown();
                    }
                });
            }

            startLatch.countDown();  // Start all threads
            assertTrue(doneLatch.await(30, TimeUnit.SECONDS));
            executor.shutdown();

            if (error.get() != null) {
                throw new AssertionError("Thread safety violation", error.get());
            }
        }
    }

    // ==================== Close Behavior Tests ====================

    @Nested
    @DisplayName("Close Behavior")
    class CloseBehavior {

        @Test
        @DisplayName("close is idempotent")
        void closeIsIdempotent() {
            ModeContext ctx = new ModeContext(Inference.INSTANCE);

            ctx.close();
            assertDoesNotThrow(() -> ctx.close());
            assertDoesNotThrow(() -> ctx.close());
        }

        @Test
        @DisplayName("isClosed returns correct state")
        void isClosedReturnsCorrectState() {
            ModeContext ctx = new ModeContext(Inference.INSTANCE);

            assertFalse(ctx.isClosed());
            ctx.close();
            assertTrue(ctx.isClosed());
        }

        @Test
        @DisplayName("toString reflects state")
        void toStringReflectsState() {
            ModeContext ctx = new ModeContext(Inference.INSTANCE);

            String open = ctx.toString();
            assertTrue(open.contains("inference"));
            assertTrue(open.contains("training"));  // Previous was training

            ctx.close();

            String closed = ctx.toString();
            assertTrue(closed.contains("CLOSED"));
        }
    }

    // ==================== Usage Pattern Tests ====================

    @Nested
    @DisplayName("Usage Patterns")
    class UsagePatterns {

        @Test
        @DisplayName("evaluation pattern: training -> inference -> training")
        void evaluationPattern() {
            // Simulate training loop with periodic evaluation
            for (int epoch = 0; epoch < 3; epoch++) {
                // Training phase
                assertTrue(ModeContext.isTraining());
                // ... do training ...

                // Evaluation phase
                try (var evalCtx = ModeContext.inference()) {
                    assertTrue(ModeContext.isInference());
                    // ... do evaluation ...
                }

                // Back to training
                assertTrue(ModeContext.isTraining());
            }
        }

        @Test
        @DisplayName("mixed-precision training pattern")
        void mixedPrecisionPattern() {
            // Outer training context
            assertTrue(ModeContext.isTraining());

            // Periodically switch for checkpointing (hypothetically needs inference)
            try (var ctx = ModeContext.inference()) {
                // Save model state
                assertFalse(ModeContext.isTraining());
            }

            // Continue training
            assertTrue(ModeContext.isTraining());
        }
    }
}
