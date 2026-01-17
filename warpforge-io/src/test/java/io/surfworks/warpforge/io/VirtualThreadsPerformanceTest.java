package io.surfworks.warpforge.io;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Performance tests for VirtualThreads utility class.
 *
 * <p>These tests verify that virtual threads provide expected scalability
 * and performance characteristics for I/O-bound workloads.
 */
@Tag("performance")
@DisplayName("VirtualThreads Performance Tests")
class VirtualThreadsPerformanceTest {

    @Test
    @DisplayName("Should scale to thousands of concurrent blocking tasks")
    void testScalabilityWithBlockingTasks() throws Exception {
        int taskCount = 10_000;
        CountDownLatch allStarted = new CountDownLatch(taskCount);
        CountDownLatch allComplete = new CountDownLatch(taskCount);
        AtomicInteger maxConcurrent = new AtomicInteger(0);
        AtomicInteger currentRunning = new AtomicInteger(0);

        Instant start = Instant.now();

        List<CompletableFuture<Void>> futures = new ArrayList<>(taskCount);
        for (int i = 0; i < taskCount; i++) {
            futures.add(VirtualThreads.runAsync(() -> {
                int running = currentRunning.incrementAndGet();
                maxConcurrent.updateAndGet(max -> Math.max(max, running));
                allStarted.countDown();
                try {
                    // Simulate blocking I/O
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    currentRunning.decrementAndGet();
                    allComplete.countDown();
                }
            }));
        }

        // Wait for all tasks to complete
        assertTrue(allComplete.await(60, TimeUnit.SECONDS),
                "All tasks should complete within timeout");

        Duration elapsed = Duration.between(start, Instant.now());

        // With virtual threads, we should achieve high concurrency
        // even with blocking operations
        assertTrue(maxConcurrent.get() > 100,
                "Should have high concurrency, but max was: " + maxConcurrent.get());

        // Total time should be much less than taskCount * sleepTime
        // because tasks run concurrently
        long expectedSequentialMs = taskCount * 50L;
        assertTrue(elapsed.toMillis() < expectedSequentialMs / 10,
                "Should complete much faster than sequential execution. " +
                        "Elapsed: " + elapsed.toMillis() + "ms, Sequential would be: " + expectedSequentialMs + "ms");
    }

    @Test
    @DisplayName("Should handle burst of CPU-bound work")
    void testCpuBoundBurst() throws Exception {
        int taskCount = 1000;
        LongAdder totalWork = new LongAdder();

        Instant start = Instant.now();

        List<CompletableFuture<Long>> futures = new ArrayList<>(taskCount);
        for (int i = 0; i < taskCount; i++) {
            final int taskId = i;
            futures.add(VirtualThreads.supplyAsync(() -> {
                // Simulate some CPU work
                long sum = 0;
                for (int j = 0; j < 10_000; j++) {
                    sum += taskId + j;
                }
                totalWork.add(sum);
                return sum;
            }));
        }

        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .get(30, TimeUnit.SECONDS);

        Duration elapsed = Duration.between(start, Instant.now());

        // All work should complete
        assertTrue(totalWork.sum() > 0, "Work should have been done");

        // Should complete in reasonable time
        assertTrue(elapsed.toSeconds() < 30,
                "CPU burst should complete in reasonable time");
    }

    @Test
    @DisplayName("Should maintain throughput under sustained load")
    void testSustainedThroughput() throws Exception {
        int operationsPerBatch = 100;
        int batches = 50;
        AtomicLong totalOperations = new AtomicLong(0);

        Instant start = Instant.now();

        for (int batch = 0; batch < batches; batch++) {
            List<CompletableFuture<Void>> futures = new ArrayList<>(operationsPerBatch);

            for (int i = 0; i < operationsPerBatch; i++) {
                futures.add(VirtualThreads.runAsync(() -> {
                    try {
                        // Simulate I/O operation
                        Thread.sleep(10);
                        totalOperations.incrementAndGet();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            }

            // Wait for batch to complete
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                    .get(30, TimeUnit.SECONDS);
        }

        Duration elapsed = Duration.between(start, Instant.now());

        assertEquals(batches * operationsPerBatch, totalOperations.get(),
                "All operations should complete");

        // Calculate throughput
        double throughput = totalOperations.get() / (elapsed.toMillis() / 1000.0);
        assertTrue(throughput > 100,
                "Should maintain reasonable throughput, got: " + throughput + " ops/sec");
    }

    @Test
    @DisplayName("Should handle many concurrent suppliers returning results")
    void testConcurrentSuppliers() throws Exception {
        int taskCount = 5000;
        List<CompletableFuture<Integer>> futures = new ArrayList<>(taskCount);

        Instant start = Instant.now();

        for (int i = 0; i < taskCount; i++) {
            final int value = i;
            futures.add(VirtualThreads.supplyAsync(() -> {
                try {
                    Thread.sleep(20);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                return value * 2;
            }));
        }

        // Collect all results
        long sum = 0;
        for (CompletableFuture<Integer> future : futures) {
            sum += future.get(30, TimeUnit.SECONDS);
        }

        Duration elapsed = Duration.between(start, Instant.now());

        // Verify results
        long expectedSum = 0;
        for (int i = 0; i < taskCount; i++) {
            expectedSum += i * 2;
        }
        assertEquals(expectedSum, sum, "All results should be correct");

        // Should complete much faster than sequential
        assertTrue(elapsed.toSeconds() < 30,
                "Should complete in reasonable time, took: " + elapsed.toSeconds() + "s");
    }

    @Test
    @DisplayName("Should efficiently handle task chaining")
    void testTaskChainingPerformance() throws Exception {
        int chainLength = 100;
        int numChains = 100;

        Instant start = Instant.now();

        List<CompletableFuture<Integer>> finalFutures = new ArrayList<>(numChains);

        for (int chain = 0; chain < numChains; chain++) {
            CompletableFuture<Integer> future = VirtualThreads.supplyAsync(() -> 0);

            for (int i = 0; i < chainLength; i++) {
                future = future.thenApplyAsync(val -> val + 1, VirtualThreads.executor());
            }

            finalFutures.add(future);
        }

        // Collect all results
        int totalSum = 0;
        for (CompletableFuture<Integer> future : finalFutures) {
            totalSum += future.get(30, TimeUnit.SECONDS);
        }

        Duration elapsed = Duration.between(start, Instant.now());

        // Each chain should result in chainLength
        assertEquals(numChains * chainLength, totalSum,
                "All chains should complete with correct values");

        // Should complete in reasonable time
        assertTrue(elapsed.toSeconds() < 30,
                "Task chaining should be efficient");
    }

    @Test
    @DisplayName("Should handle rapid task submission and completion")
    void testRapidSubmission() throws Exception {
        int iterations = 1000;
        AtomicInteger completed = new AtomicInteger(0);

        Instant start = Instant.now();

        for (int i = 0; i < iterations; i++) {
            // Submit and immediately get result
            Integer result = VirtualThreads.supplyAsync(() -> {
                completed.incrementAndGet();
                return 1;
            }).get(5, TimeUnit.SECONDS);

            assertEquals(1, result);
        }

        Duration elapsed = Duration.between(start, Instant.now());

        assertEquals(iterations, completed.get(), "All tasks should complete");

        // Calculate rate
        double rate = iterations / (elapsed.toMillis() / 1000.0);
        assertTrue(rate > 100,
                "Should handle rapid submissions efficiently, got: " + rate + " ops/sec");
    }

    @Test
    @DisplayName("Virtual threads should not block carrier threads during sleep")
    void testCarrierThreadNotBlocked() throws Exception {
        int taskCount = 1000;
        long sleepMs = 100;
        CountDownLatch allComplete = new CountDownLatch(taskCount);

        Instant start = Instant.now();

        // Submit many sleeping tasks
        for (int i = 0; i < taskCount; i++) {
            VirtualThreads.runAsync(() -> {
                try {
                    Thread.sleep(sleepMs);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    allComplete.countDown();
                }
            });
        }

        assertTrue(allComplete.await(30, TimeUnit.SECONDS),
                "All tasks should complete");

        Duration elapsed = Duration.between(start, Instant.now());

        // With virtual threads, 1000 tasks sleeping 100ms should complete
        // in roughly 100ms + overhead, not 1000 * 100ms
        // Allow generous margin for test stability
        assertTrue(elapsed.toMillis() < 5000,
                "Sleeping should not block carrier threads. " +
                        "Expected ~100ms, got: " + elapsed.toMillis() + "ms");
    }

    @Test
    @DisplayName("Should handle mixed I/O and compute workloads")
    void testMixedWorkload() throws Exception {
        int taskCount = 500;
        LongAdder ioTasks = new LongAdder();
        LongAdder computeTasks = new LongAdder();

        Instant start = Instant.now();

        List<CompletableFuture<Void>> futures = new ArrayList<>(taskCount);

        for (int i = 0; i < taskCount; i++) {
            if (i % 2 == 0) {
                // I/O-bound task
                futures.add(VirtualThreads.runAsync(() -> {
                    try {
                        Thread.sleep(20);
                        ioTasks.increment();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }));
            } else {
                // Compute-bound task
                futures.add(VirtualThreads.runAsync(() -> {
                    long sum = 0;
                    for (int j = 0; j < 50_000; j++) {
                        sum += j;
                    }
                    computeTasks.increment();
                    // Prevent optimization
                    if (sum < 0) throw new RuntimeException("Unexpected");
                }));
            }
        }

        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .get(60, TimeUnit.SECONDS);

        Duration elapsed = Duration.between(start, Instant.now());

        assertEquals(taskCount / 2, ioTasks.sum(), "All I/O tasks should complete");
        assertEquals(taskCount / 2, computeTasks.sum(), "All compute tasks should complete");

        assertTrue(elapsed.toSeconds() < 30,
                "Mixed workload should complete efficiently");
    }
}
