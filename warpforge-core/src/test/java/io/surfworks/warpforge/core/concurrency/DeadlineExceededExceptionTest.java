package io.surfworks.warpforge.core.concurrency;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link DeadlineExceededException}.
 */
@DisplayName("DeadlineExceededException Unit Tests")
class DeadlineExceededExceptionTest {

    // ==================== Construction Tests ====================

    @Nested
    @DisplayName("Construction")
    class ConstructionTests {

        @Test
        @DisplayName("Constructor stores all fields")
        void constructorStoresAllFields() {
            Duration elapsed = Duration.ofMillis(150);
            Duration allowed = Duration.ofMillis(100);
            String lastOp = "test-operation";

            DeadlineExceededException ex = new DeadlineExceededException(elapsed, allowed, lastOp);

            assertEquals(elapsed, ex.getElapsedTime());
            assertEquals(allowed, ex.getAllowedTime());
            assertEquals(lastOp, ex.getLastOperation());
        }

        @Test
        @DisplayName("Elapsed time is accessible")
        void elapsedTimeAccessible() {
            Duration elapsed = Duration.ofMillis(200);
            DeadlineExceededException ex = new DeadlineExceededException(
                elapsed, Duration.ofMillis(100), "op");

            assertEquals(elapsed, ex.getElapsedTime());
        }

        @Test
        @DisplayName("Allowed time is accessible")
        void allowedTimeAccessible() {
            Duration allowed = Duration.ofMillis(100);
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(200), allowed, "op");

            assertEquals(allowed, ex.getAllowedTime());
        }

        @Test
        @DisplayName("Last operation is accessible")
        void lastOperationAccessible() {
            String lastOp = "compute-attention";
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(200), Duration.ofMillis(100), lastOp);

            assertEquals(lastOp, ex.getLastOperation());
        }
    }

    // ==================== Overrun Ratio Tests ====================

    @Nested
    @DisplayName("Overrun Ratio")
    class OverrunRatioTests {

        @Test
        @DisplayName("Overrun ratio calculated correctly")
        void overrunRatioCalculatedCorrectly() {
            // 200ms elapsed, 100ms allowed = 2.0x overrun
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(200), Duration.ofMillis(100), "op");

            assertEquals(2.0, ex.getOverrunRatio(), 0.001);
        }

        @Test
        @DisplayName("Overrun ratio is 1.0 at exact deadline")
        void overrunRatioOneAtExact() {
            // 100ms elapsed, 100ms allowed = 1.0x
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(100), Duration.ofMillis(100), "op");

            assertEquals(1.0, ex.getOverrunRatio(), 0.001);
        }

        @Test
        @DisplayName("Overrun ratio > 1 when over")
        void overrunRatioGreaterThanOneWhenOver() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(150), Duration.ofMillis(100), "op");

            assertTrue(ex.getOverrunRatio() > 1.0);
            assertEquals(1.5, ex.getOverrunRatio(), 0.001);
        }

        @Test
        @DisplayName("Overrun ratio < 1 when under (edge case)")
        void overrunRatioLessThanOneWhenUnder() {
            // Edge case: elapsed < allowed (shouldn't normally happen)
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(50), Duration.ofMillis(100), "op");

            assertTrue(ex.getOverrunRatio() < 1.0);
            assertEquals(0.5, ex.getOverrunRatio(), 0.001);
        }

        @Test
        @DisplayName("Large overrun ratio")
        void largeOverrunRatio() {
            // 10 seconds elapsed, 100ms allowed = 100x overrun
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofSeconds(10), Duration.ofMillis(100), "op");

            assertEquals(100.0, ex.getOverrunRatio(), 0.001);
        }
    }

    // ==================== Overrun Duration Tests ====================

    @Nested
    @DisplayName("Overrun Duration")
    class OverrunDurationTests {

        @Test
        @DisplayName("Overrun duration positive when over")
        void overrunDurationPositiveWhenOver() {
            // 200ms - 100ms = 100ms overrun
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(200), Duration.ofMillis(100), "op");

            assertEquals(Duration.ofMillis(100), ex.getOverrunDuration());
        }

        @Test
        @DisplayName("Overrun duration zero at exact")
        void overrunDurationZeroWhenNotOver() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(100), Duration.ofMillis(100), "op");

            assertEquals(Duration.ZERO, ex.getOverrunDuration());
        }

        @Test
        @DisplayName("Overrun duration zero when under (clamped)")
        void overrunDurationZeroWhenUnder() {
            // Edge case: 50ms elapsed, 100ms allowed - should return ZERO not negative
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(50), Duration.ofMillis(100), "op");

            assertEquals(Duration.ZERO, ex.getOverrunDuration());
        }

        @Test
        @DisplayName("Large overrun duration")
        void largeOverrunDuration() {
            // 10 seconds - 100ms = 9900ms overrun
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofSeconds(10), Duration.ofMillis(100), "op");

            assertEquals(Duration.ofMillis(9900), ex.getOverrunDuration());
        }
    }

    // ==================== Message Tests ====================

    @Nested
    @DisplayName("Message")
    class MessageTests {

        @Test
        @DisplayName("Message contains elapsed time")
        void messageContainsElapsed() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(200), Duration.ofMillis(100), "op");

            assertTrue(ex.getMessage().contains("PT0.2S") ||
                       ex.getMessage().contains("200") ||
                       ex.getMessage().toLowerCase().contains("elapsed"));
        }

        @Test
        @DisplayName("Message contains allowed time")
        void messageContainsAllowed() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(200), Duration.ofMillis(100), "op");

            assertTrue(ex.getMessage().contains("PT0.1S") ||
                       ex.getMessage().contains("100") ||
                       ex.getMessage().toLowerCase().contains("allowed"));
        }

        @Test
        @DisplayName("Message contains overrun ratio")
        void messageContainsRatio() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(200), Duration.ofMillis(100), "op");

            assertTrue(ex.getMessage().contains("2.0") ||
                       ex.getMessage().contains("overrun"));
        }

        @Test
        @DisplayName("Message is formatted properly")
        void messageFormatted() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(200), Duration.ofMillis(100), "test-op");

            assertNotNull(ex.getMessage());
            assertTrue(ex.getMessage().length() > 0);
            assertTrue(ex.getMessage().toLowerCase().contains("deadline"));
        }
    }

    // ==================== Hierarchy Tests ====================

    @Nested
    @DisplayName("Hierarchy")
    class HierarchyTests {

        @Test
        @DisplayName("Extends Exception")
        void extendsException() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(100), Duration.ofMillis(100), "op");

            assertTrue(ex instanceof Exception);
        }

        @Test
        @DisplayName("Is a checked exception")
        void isCheckedException() {
            // DeadlineExceededException extends Exception (not RuntimeException)
            // Verify it's a checked exception by checking its parent class
            Class<?> parent = DeadlineExceededException.class.getSuperclass();
            assertEquals(Exception.class, parent);
        }

        @Test
        @DisplayName("Can be caught as Exception")
        void canBeCaughtAsException() {
            boolean caught = false;
            try {
                throw new DeadlineExceededException(
                    Duration.ofMillis(100), Duration.ofMillis(100), "op");
            } catch (Exception e) {
                caught = true;
                assertTrue(e instanceof DeadlineExceededException);
            }
            assertTrue(caught);
        }
    }

    // ==================== Edge Cases ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("Zero elapsed time")
        void zeroElapsedTime() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ZERO, Duration.ofMillis(100), "op");

            assertEquals(Duration.ZERO, ex.getElapsedTime());
            assertEquals(0.0, ex.getOverrunRatio(), 0.001);
        }

        @Test
        @DisplayName("Empty last operation")
        void emptyLastOperation() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(100), Duration.ofMillis(100), "");

            assertEquals("", ex.getLastOperation());
        }

        @Test
        @DisplayName("Very small durations (nanoseconds)")
        void verySmallDurations() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofNanos(200), Duration.ofNanos(100), "op");

            assertEquals(2.0, ex.getOverrunRatio(), 0.001);
            assertEquals(Duration.ofNanos(100), ex.getOverrunDuration());
        }

        @Test
        @DisplayName("Very large durations")
        void veryLargeDurations() {
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofDays(2), Duration.ofDays(1), "op");

            assertEquals(2.0, ex.getOverrunRatio(), 0.001);
            assertEquals(Duration.ofDays(1), ex.getOverrunDuration());
        }

        @Test
        @DisplayName("Null last operation handled")
        void nullLastOperationHandled() {
            // This tests that null is stored (no NPE in constructor)
            DeadlineExceededException ex = new DeadlineExceededException(
                Duration.ofMillis(100), Duration.ofMillis(100), null);

            assertEquals(null, ex.getLastOperation());
        }
    }
}
