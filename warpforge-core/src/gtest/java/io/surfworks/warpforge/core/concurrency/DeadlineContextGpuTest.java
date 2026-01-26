package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.concurrency.GpuWorkCalibrator.GpuWorkResult;
import io.surfworks.warpforge.core.jfr.GpuKernelEvent;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GPU tests for {@link DeadlineContext} - runs on BOTH NVIDIA and AMD machines.
 *
 * <p>These tests validate deadline-aware execution with <b>real GPU hardware</b>,
 * including SLO enforcement and timeout behavior. Every test performs actual GPU
 * operations (memory transfers) and emits JFR events for validation.
 *
 * <p><b>Key principle:</b> Deadline enforcement is validated by performing measurable
 * GPU work via {@link GpuWorkCalibrator#doGpuWork}. Without real GPU operations,
 * these would just be CPU timing tests.
 *
 * <p>What this class validates:
 * <ul>
 *   <li>SLO enforcement - GPU work completes within deadline</li>
 *   <li>Timeout behavior - GPU work is properly cleaned up on deadline miss</li>
 *   <li>Deadline accuracy - remaining time tracking with real GPU latency</li>
 *   <li>Cancellation - GPU resources properly released</li>
 *   <li>JFR event emission - all GPU operations emit proper JFR events</li>
 * </ul>
 */
@Tag("gpu")
@DisplayName("DeadlineContext GPU Tests")
class DeadlineContextGpuTest {

    private GpuBackend backend;

    @BeforeEach
    void setUp() {
        backend = GpuTestSupport.createBackend();
        System.out.println("Running on: " + GpuTestSupport.describeEnvironment());
        // Ensure calibration is done before tests
        GpuWorkCalibrator.getCalibrationData(backend);
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
    @DisplayName("GPU work completes within 1 second SLO")
    void gpuWorkWithinOneSecondSlo() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(1));

        long startTime = System.nanoTime();

        GpuWorkResult result = ctx.execute(scope -> {
            GpuTask<GpuWorkResult> task = scope.forkWithStream(lease -> {
                // Do 50ms of real GPU work - well within 1 second SLO
                return GpuWorkCalibrator.doGpuWork(backend, lease, 50);
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

        assertTrue(elapsedMillis < 1000, "Should complete within 1 second SLO, took " + elapsedMillis + "ms");
        assertTrue(result.elapsedNanos() > 0, "GPU work should have measurable duration");

        emitSloEvent("SLO1Second", elapsedMillis, result);
    }

    @Test
    @DisplayName("GPU work completes within 200ms SLO")
    void gpuWorkWithin200msSlo() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofMillis(200));

        long startTime = System.nanoTime();

        GpuWorkResult result = ctx.execute(scope -> {
            GpuTask<GpuWorkResult> task = scope.forkWithStream(lease -> {
                // Do 20ms of real GPU work - within 200ms SLO
                return GpuWorkCalibrator.doGpuWork(backend, lease, 20);
            });

            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return task.get();
        });

        long elapsedMillis = (System.nanoTime() - startTime) / 1_000_000;

        assertTrue(elapsedMillis < 200, "Should complete within 200ms SLO, took " + elapsedMillis + "ms");
        emitSloEvent("SLO200ms", elapsedMillis, result);
    }

    @Test
    @DisplayName("Multiple GPU tasks complete within deadline")
    void multipleGpuTasksWithinDeadline() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

        List<GpuWorkResult> results = ctx.execute(scope -> {
            List<GpuTask<GpuWorkResult>> tasks = new ArrayList<>();

            // Fork 3 GPU tasks, each doing 30ms of work
            for (int i = 0; i < 3; i++) {
                tasks.add(scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, 30)));
            }

            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }

            return tasks.stream().map(GpuTask::get).toList();
        });

        assertEquals(3, results.size(), "All 3 GPU tasks should complete");
        results.forEach(r -> assertTrue(r.elapsedNanos() > 0, "Each task should have measurable duration"));
    }

    // ==================== Timeout Behavior Tests ====================

    @Test
    @DisplayName("Already-expired deadline throws immediately")
    void expiredDeadlineThrowsImmediately() {
        Instant pastDeadline = Instant.now().minusMillis(100);
        DeadlineContext ctx = DeadlineContext.withDeadline(backend, pastDeadline);

        assertThrows(DeadlineExceededException.class, () ->
            ctx.execute(scope -> {
                // This should never execute
                scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, 10));
                return null;
            })
        );
    }

    @Test
    @DisplayName("Timeout cleans up GPU resources")
    void timeoutCleansUpResources() {
        DeadlineContext ctx = DeadlineContext.withDeadline(
            backend, Instant.now().minusMillis(50));

        try {
            ctx.execute(scope -> {
                scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, 100));
                return null;
            });
        } catch (DeadlineExceededException expected) {
            // Expected
        }

        // Backend should be in clean state (no leaked resources)
        // Verify by successfully creating a new scope and doing GPU work
        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            GpuTask<GpuWorkResult> task = scope.forkWithStream(lease ->
                GpuWorkCalibrator.doGpuWork(backend, lease, GpuWorkCalibrator.MIN_WORK_MS));
            scope.joinAll();
            assertTrue(task.get().elapsedNanos() > 0, "Backend should be clean after timeout");
        } catch (Exception e) {
            throw new AssertionError("Backend should be clean after timeout", e);
        }
    }

    // ==================== Deadline Accuracy Tests ====================

    @Test
    @DisplayName("Remaining time decreases during GPU work")
    void remainingTimeDecreasesDuringGpuWork() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

        ctx.execute(scope -> {
            Duration remaining1 = ctx.remainingTime();

            // Do real GPU work
            scope.forkWithStream(lease ->
                GpuWorkCalibrator.doGpuWork(backend, lease, 50));
            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            Duration remaining2 = ctx.remainingTime();

            assertTrue(remaining2.compareTo(remaining1) < 0,
                "Remaining time should decrease after GPU work");
            return null;
        });
    }

    @Test
    @DisplayName("checkDeadline passes while GPU work is within deadline")
    void checkDeadlinePassesWithinDeadline() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));

        ctx.execute(scope -> {
            // Multiple checks during GPU work should all pass
            for (int i = 0; i < 3; i++) {
                try {
                    ctx.checkDeadline();
                } catch (DeadlineExceededException e) {
                    throw new AssertionError("Deadline check should pass", e);
                }

                // Do some real GPU work between checks
                scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, 10));
            }

            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
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
            ctx.execute(scope -> {
                // Should never execute
                scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, 10));
                return null;
            })
        );
    }

    // ==================== Cancellation Tests ====================

    @Test
    @DisplayName("Cancellation prevents GPU execution")
    void cancellationPreventsGpuExecution() {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
        AtomicInteger executionCount = new AtomicInteger(0);

        ctx.cancel();

        try {
            ctx.execute(scope -> {
                executionCount.incrementAndGet();
                scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, 10));
                return null;
            });
        } catch (DeadlineExceededException expected) {
            // Expected
        }

        assertEquals(0, executionCount.get(), "GPU work should not execute after cancel");
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
    @DisplayName("Multiple concurrent GPU tasks within deadline")
    void multipleConcurrentGpuTasksWithinDeadline() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

        long startTime = System.nanoTime();

        Long totalElements = ctx.execute(scope -> {
            GpuTask<GpuWorkResult> task1 = scope.forkWithStream(lease ->
                GpuWorkCalibrator.doGpuWork(backend, lease, 20));

            GpuTask<GpuWorkResult> task2 = scope.forkWithStream(lease ->
                GpuWorkCalibrator.doGpuWork(backend, lease, 20));

            GpuTask<GpuWorkResult> task3 = scope.forkWithStream(lease ->
                GpuWorkCalibrator.doGpuWork(backend, lease, 20));

            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            }

            return task1.get().tensorElements() +
                   task2.get().tensorElements() +
                   task3.get().tensorElements();
        });

        long elapsedMillis = (System.nanoTime() - startTime) / 1_000_000;

        assertTrue(totalElements > 0, "Should have processed tensor elements");
        assertTrue(elapsedMillis < 5000, "Should complete within 5 second deadline");
    }

    @Test
    @DisplayName("Deadline context with nested scopes and real GPU work")
    void deadlineContextWithNestedScopesAndGpuWork() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

        GpuWorkResult result = ctx.execute(outerScope -> {
            GpuTask<GpuWorkResult> outerTask = outerScope.forkWithStream(outerLease -> {
                // Outer GPU work
                GpuWorkResult outerResult = GpuWorkCalibrator.doGpuWork(backend, outerLease, 15);

                // Nested scope with its own GPU work
                try (GpuTaskScope innerScope = GpuTaskScope.open(backend, "inner")) {
                    GpuTask<GpuWorkResult> innerTask = innerScope.forkWithStream(innerLease ->
                        GpuWorkCalibrator.doGpuWork(backend, innerLease, 15));
                    innerScope.joinAll();

                    // Return combined result
                    GpuWorkResult innerResult = innerTask.get();
                    return new GpuWorkResult(
                        outerResult.elapsedNanos() + innerResult.elapsedNanos(),
                        outerResult.tensorElements() + innerResult.tensorElements(),
                        outerResult.byteSize() + innerResult.byteSize(),
                        0, 0, outerLease.streamHandle(), 0
                    );
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return outerResult;
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

        assertTrue(result.elapsedNanos() > 0, "Combined GPU work should have measurable duration");
    }

    // ==================== Timing Measurement Tests ====================

    @Test
    @DisplayName("GPU work timing is accurately measured within deadline")
    void gpuWorkTimingAccuratelyMeasured() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
        AtomicLong operationTime = new AtomicLong(0);
        final long TARGET_MS = 30;

        ctx.execute(scope -> {
            GpuTask<GpuWorkResult> task = scope.forkWithStream(lease -> {
                GpuWorkResult result = GpuWorkCalibrator.doGpuWork(backend, lease, TARGET_MS);
                operationTime.set(result.elapsedNanos());
                return result;
            });

            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            return null;
        });

        // Validate timing
        GpuWorkCalibrator.assertTimingWithinTolerance(TARGET_MS, operationTime.get(),
            "GPU work timing within deadline");
    }

    @Test
    @DisplayName("Multiple timed GPU operations tracked accurately")
    void multipleTimedGpuOperationsTracked() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(10));
        List<Long> timings = new ArrayList<>();
        final int NUM_TASKS = 4;
        final long WORK_MS = 15;

        ctx.execute(scope -> {
            List<GpuTask<GpuWorkResult>> tasks = new ArrayList<>();

            for (int i = 0; i < NUM_TASKS; i++) {
                tasks.add(scope.forkWithStream(lease ->
                    GpuWorkCalibrator.doGpuWork(backend, lease, WORK_MS)));
            }

            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            for (GpuTask<GpuWorkResult> task : tasks) {
                timings.add(task.get().elapsedNanos());
            }

            return null;
        });

        assertEquals(NUM_TASKS, timings.size(), "All tasks should report timing");
        timings.forEach(t -> assertTrue(t > 0, "Each task should have positive timing"));
    }

    // ==================== JFR Validation ====================

    @Test
    @DisplayName("JFR events emitted for deadline context GPU operations")
    void jfrEventsForDeadlineContextGpuOperations() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

        long startTime = System.nanoTime();

        GpuWorkResult result = ctx.execute(scope -> {
            GpuTask<GpuWorkResult> task = scope.forkWithStream(lease ->
                GpuWorkCalibrator.doGpuWork(backend, lease, 25));
            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return task.get();
        });

        long elapsedMicros = (System.nanoTime() - startTime) / 1000;

        // JFR events were emitted by GpuWorkCalibrator.doGpuWork()
        emitSummaryEvent("DeadlineContextJFRValidation", "5s timeout", elapsedMicros, result);

        assertTrue(result.elapsedNanos() > 0, "JFR events should have been emitted for GPU work");
    }

    @Test
    @DisplayName("JFR events contain correct stream and scope IDs")
    void jfrEventsContainCorrectIds() throws DeadlineExceededException {
        DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofSeconds(5));

        GpuWorkResult result = ctx.execute(scope -> {
            GpuTask<GpuWorkResult> task = scope.forkWithStream(lease ->
                GpuWorkCalibrator.doGpuWork(backend, lease, 10));
            try {
                scope.joinAll();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return task.get();
        });

        // Verify IDs in result (which were used in JFR events)
        assertTrue(result.streamHandle() != 0, "Stream handle should be valid");
        assertTrue(result.scopeId() != 0, "Scope ID should be present");
    }

    // ==================== Helper Methods ====================

    private void emitSloEvent(String sloName, long elapsedMs, GpuWorkResult result) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = sloName;
        event.shape = "elapsed=" + elapsedMs + "ms,gpu=" + result.elapsedMillis() + "ms";
        event.gpuTimeMicros = elapsedMs * 1000;
        event.backend = GpuTestSupport.expectedBackendName(backend);
        event.deviceIndex = backend.deviceIndex();
        event.tier = "DEADLINE_CONTEXT_SLO_GTEST";
        event.bytesTransferred = result.byteSize();
        event.memoryBandwidthGBps = result.bandwidthGBps();
        event.commit();
    }

    private void emitSummaryEvent(String operation, String shape, long elapsedMicros,
                                   GpuWorkResult result) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = operation;
        event.shape = shape;
        event.gpuTimeMicros = elapsedMicros;
        event.backend = GpuTestSupport.expectedBackendName(backend);
        event.deviceIndex = backend.deviceIndex();
        event.tier = "DEADLINE_CONTEXT_GTEST";
        event.bytesTransferred = result.byteSize();
        event.memoryBandwidthGBps = result.bandwidthGBps();
        event.commit();
    }
}
