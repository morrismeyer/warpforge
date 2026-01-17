package io.surfworks.warpforge.io;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Supplier;

/**
 * Virtual thread utilities for async I/O operations.
 *
 * <p>This class provides a shared virtual thread executor for use by RDMA and
 * collective operations. Virtual threads (JEP 444) provide better scalability
 * than platform threads for I/O-bound operations like network communication.
 *
 * <h2>Benefits over ForkJoinPool</h2>
 * <ul>
 *   <li>Scales to millions of concurrent operations</li>
 *   <li>Lower memory footprint per task</li>
 *   <li>Better suited for blocking I/O operations</li>
 *   <li>Simpler mental model (one thread per task)</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Instead of:
 * CompletableFuture.supplyAsync(() -> doWork());
 *
 * // Use:
 * VirtualThreads.supplyAsync(() -> doWork());
 * }</pre>
 */
public final class VirtualThreads {

    private static final ExecutorService EXECUTOR = Executors.newVirtualThreadPerTaskExecutor();

    private VirtualThreads() {}

    /**
     * Returns the shared virtual thread executor.
     *
     * <p>This executor creates a new virtual thread for each submitted task.
     * Virtual threads are lightweight and can scale to millions of concurrent tasks.
     *
     * @return virtual thread executor
     */
    public static ExecutorService executor() {
        return EXECUTOR;
    }

    /**
     * Executes a supplier asynchronously on a virtual thread.
     *
     * @param supplier the supplier to execute
     * @param <T> the result type
     * @return a CompletableFuture that completes with the supplier's result
     */
    public static <T> CompletableFuture<T> supplyAsync(Supplier<T> supplier) {
        return CompletableFuture.supplyAsync(supplier, EXECUTOR);
    }

    /**
     * Executes a runnable asynchronously on a virtual thread.
     *
     * @param runnable the runnable to execute
     * @return a CompletableFuture that completes when the runnable finishes
     */
    public static CompletableFuture<Void> runAsync(Runnable runnable) {
        return CompletableFuture.runAsync(runnable, EXECUTOR);
    }
}
