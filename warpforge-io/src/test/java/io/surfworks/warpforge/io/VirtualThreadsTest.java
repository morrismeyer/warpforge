package io.surfworks.warpforge.io;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for VirtualThreads utility class.
 */
@Tag("unit")
@DisplayName("VirtualThreads Unit Tests")
class VirtualThreadsTest {

    @Test
    @DisplayName("Should return virtual thread executor")
    void testExecutor() {
        ExecutorService executor = VirtualThreads.executor();
        assertNotNull(executor);
        assertFalse(executor.isShutdown());
        assertFalse(executor.isTerminated());
    }

    @Test
    @DisplayName("Should execute supplier asynchronously on virtual thread")
    void testSupplyAsync() throws Exception {
        String threadName = VirtualThreads.supplyAsync(() -> Thread.currentThread().getName())
                .get(5, TimeUnit.SECONDS);

        assertNotNull(threadName);
        // Virtual threads have names like "VirtualThread[#X]/"
        assertTrue(threadName.contains("VirtualThread") || threadName.contains("virtual"),
                "Expected virtual thread but got: " + threadName);
    }

    @Test
    @DisplayName("Should return value from supplier")
    void testSupplyAsyncReturnsValue() throws Exception {
        int result = VirtualThreads.supplyAsync(() -> 42).get(5, TimeUnit.SECONDS);
        assertEquals(42, result);
    }

    @Test
    @DisplayName("Should execute runnable asynchronously on virtual thread")
    void testRunAsync() throws Exception {
        AtomicInteger counter = new AtomicInteger(0);

        VirtualThreads.runAsync(() -> counter.incrementAndGet()).get(5, TimeUnit.SECONDS);

        assertEquals(1, counter.get());
    }

    @Test
    @DisplayName("Should propagate exception from supplier")
    void testSupplyAsyncException() {
        CompletableFuture<String> future = VirtualThreads.supplyAsync(() -> {
            throw new RuntimeException("Test exception");
        });

        ExecutionException ex = assertThrows(ExecutionException.class,
                () -> future.get(5, TimeUnit.SECONDS));
        assertTrue(ex.getCause() instanceof RuntimeException);
        assertEquals("Test exception", ex.getCause().getMessage());
    }

    @Test
    @DisplayName("Should propagate exception from runnable")
    void testRunAsyncException() {
        CompletableFuture<Void> future = VirtualThreads.runAsync(() -> {
            throw new RuntimeException("Test exception");
        });

        ExecutionException ex = assertThrows(ExecutionException.class,
                () -> future.get(5, TimeUnit.SECONDS));
        assertTrue(ex.getCause() instanceof RuntimeException);
    }

    @Test
    @DisplayName("Should handle null result from supplier")
    void testSupplyAsyncNullResult() throws Exception {
        String result = VirtualThreads.supplyAsync(() -> (String) null)
                .get(5, TimeUnit.SECONDS);
        assertNull(result);
    }

    @Test
    @DisplayName("Should execute multiple tasks concurrently")
    void testConcurrentExecution() throws Exception {
        int taskCount = 100;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(taskCount);
        AtomicInteger runningCount = new AtomicInteger(0);
        AtomicInteger maxConcurrent = new AtomicInteger(0);

        List<CompletableFuture<Void>> futures = new ArrayList<>();
        for (int i = 0; i < taskCount; i++) {
            futures.add(VirtualThreads.runAsync(() -> {
                try {
                    startLatch.await();
                    int current = runningCount.incrementAndGet();
                    maxConcurrent.updateAndGet(max -> Math.max(max, current));
                    Thread.sleep(10); // Simulate work
                    runningCount.decrementAndGet();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    doneLatch.countDown();
                }
            }));
        }

        startLatch.countDown(); // Start all tasks
        assertTrue(doneLatch.await(30, TimeUnit.SECONDS), "Tasks should complete");

        // With virtual threads, many should run concurrently
        assertTrue(maxConcurrent.get() > 1,
                "Expected concurrent execution, max concurrent was: " + maxConcurrent.get());
    }

    @Test
    @DisplayName("Should chain async operations")
    void testChainedOperations() throws Exception {
        String result = VirtualThreads.supplyAsync(() -> "Hello")
                .thenApply(s -> s + " World")
                .thenApply(String::toUpperCase)
                .get(5, TimeUnit.SECONDS);

        assertEquals("HELLO WORLD", result);
    }

    @Test
    @DisplayName("Should compose multiple async operations")
    void testComposeOperations() throws Exception {
        CompletableFuture<Integer> future1 = VirtualThreads.supplyAsync(() -> 10);
        CompletableFuture<Integer> future2 = VirtualThreads.supplyAsync(() -> 20);

        int result = future1.thenCombine(future2, Integer::sum).get(5, TimeUnit.SECONDS);

        assertEquals(30, result);
    }

    @Test
    @DisplayName("Executor should be singleton")
    void testExecutorSingleton() {
        ExecutorService executor1 = VirtualThreads.executor();
        ExecutorService executor2 = VirtualThreads.executor();
        assertSame(executor1, executor2);
    }
}
