package io.surfworks.warpforge.core.kernel;

import io.surfworks.warpforge.core.jfr.GpuKernelEvent;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for LaunchTimer.
 */
@DisplayName("LaunchTimer Unit Tests")
class LaunchTimerTest {

    @Nested
    @DisplayName("Basic Timing")
    class BasicTiming {

        @Test
        @DisplayName("start() creates timer with current timestamp")
        void startCreatesTimer() {
            long before = System.nanoTime();
            LaunchTimer timer = LaunchTimer.start();
            long after = System.nanoTime();

            assertTrue(timer.startNanos() >= before);
            assertTrue(timer.startNanos() <= after);
            assertFalse(timer.isLaunched());
        }

        @Test
        @DisplayName("startAt() creates timer with specific timestamp")
        void startAtCreatesTimer() {
            long timestamp = 1_000_000_000L;
            LaunchTimer timer = LaunchTimer.startAt(timestamp);

            assertEquals(timestamp, timer.startNanos());
            assertFalse(timer.isLaunched());
        }

        @Test
        @DisplayName("markLaunched() records launch time")
        void markLaunchedRecordsTime() throws InterruptedException {
            LaunchTimer timer = LaunchTimer.start();
            Thread.sleep(1); // Ensure some time passes
            timer.markLaunched();

            assertTrue(timer.isLaunched());
            assertTrue(timer.launchNanos() > timer.startNanos());
            assertTrue(timer.launchLatencyNanos() > 0);
        }

        @Test
        @DisplayName("markLaunchedAt() records specific timestamp")
        void markLaunchedAtRecordsTime() {
            long start = 1_000_000_000L;
            long launch = 1_000_100_000L; // 100μs later
            LaunchTimer timer = LaunchTimer.startAt(start);
            timer.markLaunchedAt(launch);

            assertTrue(timer.isLaunched());
            assertEquals(launch, timer.launchNanos());
            assertEquals(100_000, timer.launchLatencyNanos());
        }
    }

    @Nested
    @DisplayName("Queue Delay")
    class QueueDelay {

        @Test
        @DisplayName("setQueueDelayNanos() sets queue delay")
        void setQueueDelay() {
            LaunchTimer timer = LaunchTimer.startAt(0);
            timer.markLaunchedAt(1000);
            timer.setQueueDelayNanos(5000);

            assertEquals(5000, timer.queueDelayNanos());
        }

        @Test
        @DisplayName("queue delay is 0 by default")
        void queueDelayDefaultZero() {
            LaunchTimer timer = LaunchTimer.start();
            timer.markLaunched();

            assertEquals(0, timer.queueDelayNanos());
        }

        @Test
        @DisplayName("totalOverheadNanos() sums launch and queue delay")
        void totalOverhead() {
            LaunchTimer timer = LaunchTimer.startAt(0);
            timer.markLaunchedAt(1000); // 1μs launch latency
            timer.setQueueDelayNanos(500); // 0.5μs queue delay

            assertEquals(1000, timer.launchLatencyNanos());
            assertEquals(500, timer.queueDelayNanos());
            assertEquals(1500, timer.totalOverheadNanos());
        }
    }

    @Nested
    @DisplayName("Event Population")
    class EventPopulation {

        @Test
        @DisplayName("populateEvent() sets timing fields")
        void populateEventSetsFields() {
            LaunchTimer timer = LaunchTimer.startAt(0);
            timer.markLaunchedAt(10_000); // 10μs launch latency
            timer.setQueueDelayNanos(2_000); // 2μs queue delay

            GpuKernelEvent event = new GpuKernelEvent();
            timer.populateEvent(event);

            assertEquals(10_000, event.launchLatencyNanos);
            assertEquals(2_000, event.queueDelayNanos);
        }

        @Test
        @DisplayName("populateEvent() works before markLaunched")
        void populateEventBeforeLaunch() {
            LaunchTimer timer = LaunchTimer.start();
            GpuKernelEvent event = new GpuKernelEvent();
            timer.populateEvent(event);

            assertEquals(0, event.launchLatencyNanos);
            assertEquals(0, event.queueDelayNanos);
        }
    }

    @Nested
    @DisplayName("Convenience Methods")
    class ConvenienceMethods {

        @Test
        @DisplayName("time() measures operation and returns result")
        void timeReturnsResult() throws Exception {
            LaunchTimer.TimedResult<String> result = LaunchTimer.time(() -> {
                Thread.sleep(1);
                return "done";
            });

            assertEquals("done", result.result());
            assertTrue(result.launchLatencyNanos() > 0);
            assertNotNull(result.timer());
        }

        @Test
        @DisplayName("timeVoid() measures void operation")
        void timeVoidMeasures() throws Exception {
            LaunchTimer timer = LaunchTimer.timeVoid(() -> {
                Thread.sleep(1);
            });

            assertTrue(timer.isLaunched());
            assertTrue(timer.launchLatencyNanos() > 0);
        }

        @Test
        @DisplayName("TimedResult.populateEvent() delegates to timer")
        void timedResultPopulatesEvent() throws Exception {
            LaunchTimer.TimedResult<Integer> result = LaunchTimer.time(() -> 42);

            GpuKernelEvent event = new GpuKernelEvent();
            result.populateEvent(event);

            assertTrue(event.launchLatencyNanos >= 0);
        }
    }

    @Nested
    @DisplayName("toString()")
    class ToStringTests {

        @Test
        @DisplayName("toString() before launch shows not launched")
        void toStringBeforeLaunch() {
            LaunchTimer timer = LaunchTimer.start();
            String str = timer.toString();

            assertTrue(str.contains("not launched"));
        }

        @Test
        @DisplayName("toString() after launch shows latencies")
        void toStringAfterLaunch() {
            LaunchTimer timer = LaunchTimer.startAt(0);
            timer.markLaunchedAt(1000);
            timer.setQueueDelayNanos(500);
            String str = timer.toString();

            assertTrue(str.contains("1000ns"));
            assertTrue(str.contains("500ns"));
            assertTrue(str.contains("1500ns")); // total
        }
    }

    @Nested
    @DisplayName("Typical Latency Scenarios")
    class TypicalScenarios {

        @Test
        @DisplayName("typical cuLaunchKernel latency (5-50μs)")
        void typicalCudaLaunchLatency() {
            // Simulate typical CUDA launch latency range
            LaunchTimer timer = LaunchTimer.startAt(0);
            timer.markLaunchedAt(15_000); // 15μs - typical

            long latencyMicros = timer.launchLatencyNanos() / 1000;
            assertTrue(latencyMicros >= 1 && latencyMicros <= 100,
                "Launch latency should be in typical range: " + latencyMicros + "μs");
        }

        @Test
        @DisplayName("empty stream queue delay (near zero)")
        void emptyStreamQueueDelay() {
            LaunchTimer timer = LaunchTimer.startAt(0);
            timer.markLaunchedAt(10_000);
            timer.setQueueDelayNanos(100); // ~100ns for empty stream

            assertTrue(timer.queueDelayNanos() < 1000,
                "Empty stream queue delay should be < 1μs");
        }

        @Test
        @DisplayName("busy stream queue delay (1-100μs)")
        void busyStreamQueueDelay() {
            LaunchTimer timer = LaunchTimer.startAt(0);
            timer.markLaunchedAt(10_000);
            timer.setQueueDelayNanos(50_000); // 50μs for busy stream

            long queueMicros = timer.queueDelayNanos() / 1000;
            assertTrue(queueMicros >= 1,
                "Busy stream queue delay should be noticeable: " + queueMicros + "μs");
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("zero latency is valid")
        void zeroLatency() {
            LaunchTimer timer = LaunchTimer.startAt(1000);
            timer.markLaunchedAt(1000);

            assertEquals(0, timer.launchLatencyNanos());
        }

        @Test
        @DisplayName("very large latency is valid")
        void veryLargeLatency() {
            LaunchTimer timer = LaunchTimer.startAt(0);
            timer.markLaunchedAt(Long.MAX_VALUE / 2);

            assertTrue(timer.launchLatencyNanos() > 0);
        }

        @Test
        @DisplayName("multiple markLaunched calls overwrites")
        void multipleMarkLaunched() {
            LaunchTimer timer = LaunchTimer.startAt(0);
            timer.markLaunchedAt(1000);
            timer.markLaunchedAt(2000);

            assertEquals(2000, timer.launchLatencyNanos());
        }

        @Test
        @DisplayName("isLaunched() returns false initially")
        void isLaunchedFalseInitially() {
            LaunchTimer timer = LaunchTimer.start();
            assertFalse(timer.isLaunched());
        }

        @Test
        @DisplayName("launchNanos() is 0 before launch")
        void launchNanosZeroBeforeLaunch() {
            LaunchTimer timer = LaunchTimer.start();
            assertEquals(0, timer.launchNanos());
        }

        @Test
        @DisplayName("launchLatencyNanos() is 0 before launch")
        void launchLatencyZeroBeforeLaunch() {
            LaunchTimer timer = LaunchTimer.start();
            assertEquals(0, timer.launchLatencyNanos());
        }
    }

    @Nested
    @DisplayName("Exception Handling")
    class ExceptionHandling {

        @Test
        @DisplayName("time() propagates exceptions")
        void timePropagatesExceptions() {
            Exception thrown = null;
            try {
                LaunchTimer.time(() -> {
                    throw new RuntimeException("test exception");
                });
            } catch (Exception e) {
                thrown = e;
            }

            assertNotNull(thrown);
            assertEquals("test exception", thrown.getMessage());
        }

        @Test
        @DisplayName("timeVoid() propagates exceptions")
        void timeVoidPropagatesExceptions() {
            Exception thrown = null;
            try {
                LaunchTimer.timeVoid(() -> {
                    throw new RuntimeException("test exception");
                });
            } catch (Exception e) {
                thrown = e;
            }

            assertNotNull(thrown);
            assertEquals("test exception", thrown.getMessage());
        }
    }
}
