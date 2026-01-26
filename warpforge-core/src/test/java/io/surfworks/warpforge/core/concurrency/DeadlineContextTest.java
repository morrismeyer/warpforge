package io.surfworks.warpforge.core.concurrency;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.time.Instant;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link DeadlineContext}.
 *
 * <p>These tests use {@link MockGpuBackend} to verify deadline-aware
 * execution context behavior.
 */
@DisplayName("DeadlineContext Unit Tests")
class DeadlineContextTest {

    private MockGpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = new MockGpuBackend();
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    // ==================== Factory Tests ====================

    @Nested
    @DisplayName("Factory")
    class FactoryTests {

        @Test
        @DisplayName("withDeadline() creates context")
        void withDeadlineCreatesContext() {
            Instant deadline = Instant.now().plusSeconds(10);
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, deadline);

            assertNotNull(ctx);
            assertEquals(deadline, ctx.deadline());
        }

        @Test
        @DisplayName("withTimeout() creates context")
        void withTimeoutCreatesContext() {
            Duration timeout = Duration.ofMillis(500);
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, timeout);

            assertNotNull(ctx);
            assertEquals(timeout, ctx.timeout());
        }

        @Test
        @DisplayName("withDeadline() calculates timeout")
        void withDeadlineCalculatesTimeout() {
            Instant deadline = Instant.now().plusMillis(1000);
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, deadline);

            // Timeout should be approximately 1 second (allow some tolerance)
            Duration timeout = ctx.timeout();
            assertTrue(timeout.toMillis() <= 1000);
            assertTrue(timeout.toMillis() >= 900);
        }

        @Test
        @DisplayName("withTimeout() calculates deadline")
        void withTimeoutCalculatesDeadline() {
            Instant before = Instant.now();
            Duration timeout = Duration.ofMillis(500);
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, timeout);
            Instant after = Instant.now();

            // Deadline should be between before+timeout and after+timeout
            assertTrue(ctx.deadline().isAfter(before.plus(timeout).minusMillis(50)));
            assertTrue(ctx.deadline().isBefore(after.plus(timeout).plusMillis(50)));
        }
    }

    // ==================== Execute Tests ====================

    @Nested
    @DisplayName("Execute")
    class ExecuteTests {

        @Test
        @DisplayName("execute() runs operation")
        void executeRunsOperation() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            Integer result = ctx.execute(scope -> 42);

            assertEquals(42, result);
        }

        @Test
        @DisplayName("execute() passes scope to operation")
        void executePassesScope() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            String result = ctx.execute(scope -> {
                assertNotNull(scope);
                return scope.scopeName();
            });

            assertEquals("deadline-context", result);
        }

        @Test
        @DisplayName("execute() closes scope after completion")
        void executeScopesClosed() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            ctx.execute(scope -> {
                scope.fork(() -> 1);
                try {
                    scope.joinAll();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                return null;
            });

            // All streams should be cleaned up
            assertEquals(0, backend.activeStreamCount());
        }

        @Test
        @DisplayName("execute() returns result")
        void executeReturnResult() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            String result = ctx.execute(scope -> "hello-world");

            assertEquals("hello-world", result);
        }

        @Test
        @DisplayName("execute() with scope operations")
        void executeWithScopeOperations() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            Integer result = ctx.execute(scope -> {
                GpuTask<Integer> task1 = scope.fork(() -> 10);
                GpuTask<Integer> task2 = scope.fork(() -> 20);
                try {
                    scope.joinAll();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }
                return task1.get() + task2.get();
            });

            assertEquals(30, result);
        }
    }

    // ==================== Deadline Checking Tests ====================

    @Nested
    @DisplayName("Deadline Checking")
    class DeadlineCheckingTests {

        @Test
        @DisplayName("checkDeadline() passes before expiry")
        void checkDeadlinePassesBeforeExpiry() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertDoesNotThrow(ctx::checkDeadline);
        }

        @Test
        @DisplayName("checkDeadline() throws after expiry")
        void checkDeadlineThrowsAfterExpiry() {
            // Create context that's already expired
            Instant pastDeadline = Instant.now().minusMillis(100);
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, pastDeadline);

            assertThrows(DeadlineExceededException.class, ctx::checkDeadline);
        }

        @Test
        @DisplayName("checkDeadline() throws when cancelled")
        void checkDeadlineThrowsWhenCancelled() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
            ctx.cancel();

            assertThrows(DeadlineExceededException.class, ctx::checkDeadline);
        }
    }

    // ==================== Time Query Tests ====================

    @Nested
    @DisplayName("Time Queries")
    class TimeQueryTests {

        @Test
        @DisplayName("remainingTime() positive before deadline")
        void remainingTimePositiveBeforeDeadline() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            Duration remaining = ctx.remainingTime();

            assertTrue(remaining.isPositive());
            assertTrue(remaining.toSeconds() <= 10);
        }

        @Test
        @DisplayName("remainingTime() near zero at deadline")
        void remainingTimeNearZeroAtDeadline() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofMillis(50));

            // Wait for deadline to approach
            try {
                Thread.sleep(60);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            Duration remaining = ctx.remainingTime();
            assertTrue(remaining.isNegative() || remaining.toMillis() <= 10);
        }

        @Test
        @DisplayName("remainingTime() negative after deadline")
        void remainingTimeNegativeAfterExpiry() {
            Instant pastDeadline = Instant.now().minusMillis(100);
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, pastDeadline);

            Duration remaining = ctx.remainingTime();

            assertTrue(remaining.isNegative());
        }
    }

    // ==================== Expiry Tests ====================

    @Nested
    @DisplayName("Expiry")
    class ExpiryTests {

        @Test
        @DisplayName("isExpired() false before deadline")
        void isExpiredFalseBeforeDeadline() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertFalse(ctx.isExpired());
        }

        @Test
        @DisplayName("isExpired() true after deadline")
        void isExpiredTrueAfterDeadline() {
            Instant pastDeadline = Instant.now().minusMillis(100);
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, pastDeadline);

            assertTrue(ctx.isExpired());
        }

        @Test
        @DisplayName("isExpired() changes over time")
        void isExpiredChangesOverTime() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofMillis(50));

            assertFalse(ctx.isExpired());

            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            assertTrue(ctx.isExpired());
        }
    }

    // ==================== Cancellation Tests ====================

    @Nested
    @DisplayName("Cancellation")
    class CancellationTests {

        @Test
        @DisplayName("isCancelled() false initially")
        void isCancelledFalseInitially() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertFalse(ctx.isCancelled());
        }

        @Test
        @DisplayName("cancel() sets cancelled flag")
        void cancelSetsCancelled() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            ctx.cancel();

            assertTrue(ctx.isCancelled());
        }

        @Test
        @DisplayName("cancel() is idempotent")
        void cancelIdempotent() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            ctx.cancel();
            ctx.cancel();
            ctx.cancel();

            assertTrue(ctx.isCancelled());
        }

        @Test
        @DisplayName("execute() after cancel throws")
        void executeAfterCancelThrows() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
            ctx.cancel();

            assertThrows(DeadlineExceededException.class, () -> ctx.execute(scope -> 42));
        }
    }

    // ==================== Edge Case Tests ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("Zero timeout expires immediately")
        void zeroTimeoutExpiresImmediately() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ZERO);

            assertTrue(ctx.isExpired());
        }

        @Test
        @DisplayName("Very short timeout")
        void veryShortTimeout() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofNanos(1));

            // Should expire almost immediately
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            assertTrue(ctx.isExpired());
        }

        @Test
        @DisplayName("Very long timeout doesn't overflow")
        void veryLongTimeout() {
            Duration longTimeout = Duration.ofDays(365);
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, longTimeout);

            assertFalse(ctx.isExpired());
            assertTrue(ctx.remainingTime().toDays() >= 364);
        }

        @Test
        @DisplayName("Negative remaining time after expired deadline")
        void negativeRemainingTime() {
            Instant veryPastDeadline = Instant.now().minusSeconds(60);
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, veryPastDeadline);

            assertTrue(ctx.remainingTime().isNegative());
            assertTrue(ctx.remainingTime().toSeconds() < -50);
        }
    }

    // ==================== Exception Handling Tests ====================

    @Nested
    @DisplayName("Exception Handling")
    class ExceptionHandlingTests {

        @Test
        @DisplayName("Operation exception propagates")
        void operationExceptionPropagates() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertThrows(IllegalStateException.class, () ->
                ctx.execute(scope -> {
                    throw new IllegalStateException("Operation failed");
                })
            );
        }

        @Test
        @DisplayName("DeadlineExceededException has diagnostics")
        void deadlineExceptionHasDiagnostics() {
            Instant pastDeadline = Instant.now().minusMillis(100);
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, pastDeadline);

            DeadlineExceededException ex = assertThrows(
                DeadlineExceededException.class,
                ctx::checkDeadline
            );

            assertNotNull(ex.getElapsedTime());
            assertNotNull(ex.getAllowedTime());
            assertNotNull(ex.getLastOperation());
        }

        @Test
        @DisplayName("RuntimeException in execute propagates")
        void runtimeExceptionPropagates() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertThrows(RuntimeException.class, () ->
                ctx.execute(scope -> {
                    throw new RuntimeException("Test exception");
                })
            );
        }
    }

    // ==================== Accessor Tests ====================

    @Nested
    @DisplayName("Accessors")
    class AccessorTests {

        @Test
        @DisplayName("deadline() returns instant")
        void deadlineReturnsInstant() {
            Instant deadline = Instant.now().plusSeconds(60);
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, deadline);

            assertEquals(deadline, ctx.deadline());
        }

        @Test
        @DisplayName("timeout() returns duration")
        void timeoutReturnsDuration() {
            Duration timeout = Duration.ofMillis(500);
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, timeout);

            assertEquals(timeout, ctx.timeout());
        }

        @Test
        @DisplayName("backend() returns configured backend")
        void backendReturnsConfigured() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertEquals(backend, ctx.backend());
        }
    }

    // ==================== Integration Tests ====================

    @Nested
    @DisplayName("Integration")
    class IntegrationTests {

        @Test
        @DisplayName("Multiple checkDeadline() calls work")
        void multipleCheckDeadlineCalls() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertDoesNotThrow(() -> {
                for (int i = 0; i < 100; i++) {
                    ctx.checkDeadline();
                }
            });
        }

        @Test
        @DisplayName("Nested execute with different contexts")
        void nestedExecuteWithDifferentContexts() throws DeadlineExceededException {
            DeadlineContext outer = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
            DeadlineContext inner = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

            Integer result = outer.execute(outerScope -> {
                try {
                    return inner.execute(innerScope -> 42);
                } catch (DeadlineExceededException e) {
                    return -1;
                }
            });

            assertEquals(42, result);
        }

        @Test
        @DisplayName("Context can be checked from multiple threads")
        void contextCheckedFromMultipleThreads() throws Exception {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            Thread t1 = new Thread(() -> {
                for (int i = 0; i < 50; i++) {
                    try {
                        ctx.checkDeadline();
                    } catch (DeadlineExceededException e) {
                        break;
                    }
                }
            });

            Thread t2 = new Thread(() -> {
                for (int i = 0; i < 50; i++) {
                    try {
                        ctx.checkDeadline();
                    } catch (DeadlineExceededException e) {
                        break;
                    }
                }
            });

            t1.start();
            t2.start();
            t1.join();
            t2.join();

            // Should complete without errors
            assertFalse(ctx.isCancelled());
        }
    }
}
