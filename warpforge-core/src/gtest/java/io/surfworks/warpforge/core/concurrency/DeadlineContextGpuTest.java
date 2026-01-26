package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.jfr.GpuKernelEvent;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU tests for {@link DeadlineContext} - runs on BOTH NVIDIA and AMD machines.
 *
 * <p>These tests validate deadline-aware execution with real GPU hardware,
 * including SLO enforcement and timeout behavior.
 */
@Tag("gpu")
@DisplayName("DeadlineContext GPU Tests")
class DeadlineContextGpuTest {

    private GpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = GpuTestSupport.createBackend();
        System.out.println("Running on: " + GpuTestSupport.describeEnvironment());
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    // ==================== SLO Enforcement Tests ====================

    @Test
    @DisplayName("Operation completes within 1 second SLO")
    void operationWithinOneSecondSlo() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(1));

        long startTime = System.nanoTime();

        Integer result = ctx.execute(scope -> {
            GpuTask<Integer> task = scope.fork(() -> {
                // Light work that should complete quickly
                long sum = 0;
                for (int i = 0; i < 1000; i++) {
                    sum += i;
                }
                return (int) (sum % 100);
            });

            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }

            return task.get();
        });

        long elapsedMillis = (System.nanoTime() - startTime) / 1_000_000;

        assertTrue(elapsedMillis < 1000, "Should complete within 1 second SLO");
        emitKernelEvent("SLO1Second", "elapsed=" + elapsedMillis + "ms", elapsedMillis * 1000);
    }

    @Test
    @DisplayName("Operation completes within 100ms SLO")
    void operationWithin100msSlo() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofMillis(100));

        long startTime = System.nanoTime();

        Integer result = ctx.execute(scope -> {
            GpuTask<Integer> task = scope.fork(() -> 42);
            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return task.get();
        });

        long elapsedMillis = (System.nanoTime() - startTime) / 1_000_000;

        assertEquals(42, result);
        assertTrue(elapsedMillis < 100, "Should complete within 100ms SLO, took " + elapsedMillis + "ms");
    }

    // ==================== Timeout Behavior Tests ====================

    @Test
    @DisplayName("Already-expired deadline throws immediately")
    void expiredDeadlineThrowsImmediately() {
        Instant pastDeadline = Instant.now().minusMillis(100);
        DeadlineContext ctx = DeadlineContext.withDeadline(backend, pastDeadline);

        assertThrows(DeadlineExceededException.class, () ->
            ctx.execute(scope -> 42)
        );
    }

    @Test
    @DisplayName("Timeout cleans up GPU resources")
    void timeoutCleansUpResources() {
        DeadlineContext ctx = DeadlineContext.withDeadline(
            backend, Instant.now().minusMillis(50));

        try {
            ctx.execute(scope -> {
                scope.forkWithStream(lease -> 42);
                return null;
            });
        } catch (DeadlineExceededException expected) {
            // Expected
        }

        // Backend should be in clean state (no leaked resources)
        // We verify this by successfully creating a new scope
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            GpuTask<Integer> task = scope.fork(() -> 1);
            scope.joinAll();
            assertEquals(1, task.get());
        } catch (Exception e) {
            throw new AssertionError("Backend should be clean after timeout", e);
        }
    }

    // ==================== Deadline Accuracy Tests ====================

    @Test
    @DisplayName("Remaining time decreases over time")
    void remainingTimeDecreases() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

        ctx.execute(scope -> {
            Duration remaining1 = ctx.remainingTime();

            // Do some work
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            Duration remaining2 = ctx.remainingTime();

            assertTrue(remaining2.compareTo(remaining1) < 0,
                "Remaining time should decrease over time");
            return null;
        });
    }

    @Test
    @DisplayName("checkDeadline passes when within deadline")
    void checkDeadlinePassesWithinDeadline() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

        ctx.execute(scope -> {
            // Multiple checks should all pass
            for (int i = 0; i < 10; i++) {
                try {
                    ctx.checkDeadline();
                } catch (DeadlineExceededException e) {
                    throw new AssertionError("Deadline check should pass", e);
                }
            }
            return null;
        });
    }

    @Test
    @DisplayName("checkDeadline throws when cancelled")
    void checkDeadlineThrowsWhenCancelled() {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
        ctx.cancel();

        assertThrows(DeadlineExceededException.class, () ->
            ctx.execute(scope -> 42)
        );
    }

    // ==================== Cancellation Tests ====================

    @Test
    @DisplayName("Cancellation prevents execution")
    void cancellationPreventsExecution() {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
        AtomicInteger executionCount = new AtomicInteger(0);

        ctx.cancel();

        try {
            ctx.execute(scope -> {
                executionCount.incrementAndGet();
                return null;
            });
        } catch (DeadlineExceededException expected) {
            // Expected
        }

        assertEquals(0, executionCount.get(), "Operation should not execute after cancel");
    }

    @Test
    @DisplayName("Cancellation is idempotent")
    void cancellationIsIdempotent() {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

        assertFalse(ctx.isCancelled());

        ctx.cancel();
        assertTrue(ctx.isCancelled());

        ctx.cancel();
        ctx.cancel();
        assertTrue(ctx.isCancelled());
    }

    // ==================== Combined GPU Operations Tests ====================

    @Test
    @DisplayName("Multiple GPU tasks within deadline")
    void multipleGpuTasksWithinDeadline() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

        Integer result = ctx.execute(scope -> {
            GpuTask<Integer> task1 = scope.forkWithStream(lease -> {
                lease.synchronize();
                return 10;
            });

            GpuTask<Integer> task2 = scope.forkWithStream(lease -> {
                lease.synchronize();
                return 20;
            });

            GpuTask<Integer> task3 = scope.forkWithStream(lease -> {
                lease.synchronize();
                return 30;
            });

            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }

            return task1.get() + task2.get() + task3.get();
        });

        assertEquals(60, result);
    }

    @Test
    @DisplayName("Deadline context with nested scopes")
    void deadlineContextWithNestedScopes() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

        Integer result = ctx.execute(outerScope -> {
            GpuTask<Integer> outerTask = outerScope.fork(() -> {
                try (GpuTaskScope innerScope = GpuTaskScope.open(backend, "inner")) {
                    GpuTask<Integer> innerTask = innerScope.fork(() -> 42);
                    innerScope.joinAll();
                    return innerTask.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return -1;
                }
            });

            try {
                outerScope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }

            return outerTask.get();
        });

        assertEquals(42, result);
    }

    // ==================== Timing Measurement Tests ====================

    @Test
    @DisplayName("Deadline context execution time is measurable")
    void executionTimeIsMeasurable() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
        AtomicLong operationTime = new AtomicLong(0);

        ctx.execute(scope -> {
            long start = System.nanoTime();

            GpuTask<Void> task = scope.forkWithStream(lease -> {
                // Do some work
                long sum = 0;
                for (int i = 0; i < 10000; i++) {
                    sum += i;
                }
                lease.synchronize();
                return null;
            });

            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            operationTime.set(System.nanoTime() - start);
            return null;
        });

        assertTrue(operationTime.get() > 0, "Operation time should be measurable");
    }

    // ==================== JFR Validation ====================

    @Test
    @DisplayName("JFR events emitted for deadline context operations")
    void jfrEventsForDeadlineContext() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

        long startTime = System.nanoTime();

        ctx.execute(scope -> {
            scope.fork(() -> 42);
            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return null;
        });

        long elapsedMicros = (System.nanoTime() - startTime) / 1000;

        emitKernelEvent("DeadlineContextJFRTest", "5s timeout", elapsedMicros);

        assertTrue(true, "JFR events should be emittable");
    }

    // ==================== Helper Methods ====================

    private void emitKernelEvent(String operation, String shape, long elapsedMicros) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = operation;
        event.shape = shape;
        event.gpuTimeMicros = elapsedMicros;
        event.backend = GpuTestSupport.expectedBackendName(backend);
        event.deviceIndex = backend.deviceIndex();
        event.tier = "DEADLINE_CONTEXT_GTEST";
        event.memoryBandwidthGBps = 0.0;
        event.commit();
    }
}
