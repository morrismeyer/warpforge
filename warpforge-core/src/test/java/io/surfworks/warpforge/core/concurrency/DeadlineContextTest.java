package io.surfworks.warpforge.core.concurrency;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.time.Instant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link DeadlineContext}.
 */
@DisplayName("DeadlineContext")
class DeadlineContextTest {

    private MockGpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = new MockGpuBackend();
    }

    @Nested
    @DisplayName("Creation")
    class Creation {

        @Test
        @DisplayName("withTimeout creates context with correct duration")
        void withTimeoutCreatesCorrectly() {
            Duration timeout = Duration.ofMillis(100);
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, timeout);

            assertNotNull(ctx);
            assertEquals(timeout, ctx.timeout());
            assertEquals(backend, ctx.backend());
            assertFalse(ctx.isExpired());
            assertFalse(ctx.isCancelled());
        }

        @Test
        @DisplayName("withDeadline creates context with correct deadline")
        void withDeadlineCreatesCorrectly() {
            Instant deadline = Instant.now().plus(Duration.ofSeconds(1));
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, deadline);

            assertNotNull(ctx);
            assertEquals(deadline, ctx.deadline());
            assertEquals(backend, ctx.backend());
            assertFalse(ctx.isExpired());
        }

        @Test
        @DisplayName("expired deadline is detected")
        void expiredDeadlineDetected() {
            Instant pastDeadline = Instant.now().minus(Duration.ofSeconds(1));
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, pastDeadline);

            assertTrue(ctx.isExpired());
        }
    }

    @Nested
    @DisplayName("Deadline Checking")
    class DeadlineChecking {

        @Test
        @DisplayName("checkDeadline passes before expiry")
        void checkDeadlinePassesBeforeExpiry() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
            ctx.checkDeadline();
        }

        @Test
        @DisplayName("checkDeadline throws after expiry")
        void checkDeadlineThrowsAfterExpiry() {
            Instant pastDeadline = Instant.now().minus(Duration.ofMillis(100));
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, pastDeadline);

            assertThrows(DeadlineExceededException.class, ctx::checkDeadline);
        }

        @Test
        @DisplayName("checkDeadline throws when cancelled")
        void checkDeadlineThrowsWhenCancelled() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            ctx.cancel();
            assertTrue(ctx.isCancelled());

            assertThrows(DeadlineExceededException.class, ctx::checkDeadline);
        }

        @Test
        @DisplayName("remainingTime is positive before expiry")
        void remainingTimePositiveBeforeExpiry() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            Duration remaining = ctx.remainingTime();
            assertTrue(remaining.toMillis() > 0);
            assertTrue(remaining.toSeconds() <= 10);
        }

        @Test
        @DisplayName("remainingTime is negative after expiry")
        void remainingTimeNegativeAfterExpiry() {
            Instant pastDeadline = Instant.now().minus(Duration.ofMillis(100));
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, pastDeadline);

            Duration remaining = ctx.remainingTime();
            assertTrue(remaining.isNegative());
        }
    }

    @Nested
    @DisplayName("Execution")
    class Execution {

        @Test
        @DisplayName("execute runs operation within scope")
        void executeRunsOperation() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            Integer result = ctx.execute(scope -> {
                assertNotNull(scope);
                return 42;
            });

            assertEquals(42, result);
        }

        @Test
        @DisplayName("execute throws when deadline already passed")
        void executeThrowsWhenExpired() {
            Instant pastDeadline = Instant.now().minus(Duration.ofMillis(100));
            DeadlineContext ctx = DeadlineContext.withDeadline(backend, pastDeadline);

            assertThrows(DeadlineExceededException.class, () ->
                ctx.execute(scope -> 42));
        }

        @Test
        @DisplayName("execute provides functional scope")
        void executeProvidesScope() throws DeadlineExceededException {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            String result = ctx.execute(scope -> {
                GpuTask<String> task = scope.fork(() -> "hello");
                try {
                    scope.joinAll();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return "interrupted";
                }
                return task.get();
            });

            assertEquals("hello", result);
        }
    }

    @Nested
    @DisplayName("DeadlineExceededException")
    class ExceptionTests {

        @Test
        @DisplayName("contains elapsed and allowed times")
        void containsTimingInfo() {
            Duration elapsed = Duration.ofMillis(150);
            Duration allowed = Duration.ofMillis(100);

            DeadlineExceededException ex = new DeadlineExceededException(
                elapsed, allowed, "test-op");

            assertEquals(elapsed, ex.getElapsedTime());
            assertEquals(allowed, ex.getAllowedTime());
            assertEquals("test-op", ex.getLastOperation());
        }

        @Test
        @DisplayName("calculates overrun ratio correctly")
        void calculatesOverrunRatio() {
            Duration elapsed = Duration.ofMillis(200);
            Duration allowed = Duration.ofMillis(100);

            DeadlineExceededException ex = new DeadlineExceededException(
                elapsed, allowed, "test");

            assertEquals(2.0, ex.getOverrunRatio(), 0.01);
        }

        @Test
        @DisplayName("calculates overrun duration correctly")
        void calculatesOverrunDuration() {
            Duration elapsed = Duration.ofMillis(150);
            Duration allowed = Duration.ofMillis(100);

            DeadlineExceededException ex = new DeadlineExceededException(
                elapsed, allowed, "test");

            assertEquals(Duration.ofMillis(50), ex.getOverrunDuration());
        }

        @Test
        @DisplayName("overrun duration is zero when not exceeded")
        void overrunDurationZeroWhenNotExceeded() {
            Duration elapsed = Duration.ofMillis(50);
            Duration allowed = Duration.ofMillis(100);

            DeadlineExceededException ex = new DeadlineExceededException(
                elapsed, allowed, "test");

            assertEquals(Duration.ZERO, ex.getOverrunDuration());
        }

        @Test
        @DisplayName("message contains timing information")
        void messageContainsTiming() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(200), Duration.ofMillis(100), "inference");

            String message = ex.getMessage();
            assertTrue(message.contains("200"));
            assertTrue(message.contains("100"));
            assertTrue(message.contains("2.0x"));
        }
    }

    @Nested
    @DisplayName("Cancellation")
    class Cancellation {

        @Test
        @DisplayName("cancel marks context as cancelled")
        void cancelMarksAsCancelled() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            assertFalse(ctx.isCancelled());
            ctx.cancel();
            assertTrue(ctx.isCancelled());
        }

        @Test
        @DisplayName("cancelled context fails deadline check")
        void cancelledContextFailsCheck() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            ctx.cancel();

            assertThrows(DeadlineExceededException.class, ctx::checkDeadline);
        }

        @Test
        @DisplayName("cancelled context cannot execute")
        void cancelledContextCannotExecute() {
            DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

            ctx.cancel();

            assertThrows(DeadlineExceededException.class, () ->
                ctx.execute(scope -> 42));
        }
    }
}
