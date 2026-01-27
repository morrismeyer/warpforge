package io.surfworks.warpforge.core.concurrency;

import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Factory for creating named virtual threads for GPU operations.
 *
 * <p><b>Critical for observability:</b> JFR events recorded on unnamed virtual threads
 * have empty 'Event Thread' fields, causing Java Mission Control to collapse all
 * events into a single unnamed thread. This makes profiling nearly impossible.
 *
 * <p>By naming virtual threads with scope and operation context, WarpForge enables:
 * <ul>
 *   <li>JFR events traceable to specific GPU operations</li>
 *   <li>Mission Control grouping events by virtual thread</li>
 *   <li>End-to-end visibility from VThread to GPU kernel execution</li>
 * </ul>
 *
 * <p>This is a key competitive differentiator. See {@code architecture/LOOM-DEBUGGING.md}
 * for research findings from the loom-dev mailing list.
 *
 * <p>Naming convention: {@code warpforge-gpu-<scopeName>-<scopeId>-task-<N>}
 *
 * <p>Example thread names:
 * <pre>
 * warpforge-gpu-inference-batch-42-task-0
 * warpforge-gpu-inference-batch-42-task-1
 * warpforge-gpu-attention-layer-17-task-0
 * warpforge-gpu-gemm-4096x4096-99-task-0
 * </pre>
 *
 * <p>Usage with GpuTaskScope (automatic):
 * <pre>{@code
 * // GpuTaskScope automatically uses GpuThreadFactory internally
 * try (GpuTaskScope scope = GpuTaskScope.open(backend, "inference-batch")) {
 *     scope.fork(() -> compute()); // Thread named: warpforge-gpu-inference-batch-1-task-0
 * }
 * }</pre>
 *
 * <p>Manual usage:
 * <pre>{@code
 * ThreadFactory factory = GpuThreadFactory.forScope("gemm-4096x4096", 42);
 * Thread t = factory.newThread(() -> runKernel());
 * // Thread named: warpforge-gpu-gemm-4096x4096-42-task-0
 * }</pre>
 *
 * @see GpuTaskScope
 */
public final class GpuThreadFactory {

    private static final String PREFIX = "warpforge-gpu-";

    private GpuThreadFactory() {
        // Utility class
    }

    /**
     * Creates a ThreadFactory for a GPU task scope.
     *
     * <p>Threads created by this factory will be named:
     * {@code warpforge-gpu-<scopeName>-<scopeId>-task-<N>}
     *
     * <p>The task number auto-increments for each thread created.
     *
     * @param scopeName the scope name (e.g., "inference-batch", "attention-layer")
     * @param scopeId the unique scope identifier
     * @return a ThreadFactory that creates named virtual threads
     */
    public static ThreadFactory forScope(String scopeName, long scopeId) {
        String prefix = PREFIX + sanitize(scopeName) + "-" + scopeId + "-task-";
        AtomicLong counter = new AtomicLong(0);

        return runnable -> Thread.ofVirtual()
            .name(prefix + counter.getAndIncrement())
            .unstarted(runnable);
    }

    /**
     * Creates a Thread.Builder for custom virtual thread creation.
     *
     * <p>Use this when you need more control over thread creation than
     * a ThreadFactory provides.
     *
     * @param scopeName the scope name
     * @param scopeId the unique scope identifier
     * @return a Thread.Builder configured with the naming convention
     */
    public static Thread.Builder.OfVirtual builderForScope(String scopeName, long scopeId) {
        String prefix = PREFIX + sanitize(scopeName) + "-" + scopeId + "-task-";
        return Thread.ofVirtual().name(prefix, 0);
    }

    /**
     * Creates a ThreadFactory for a specific operation within a scope.
     *
     * <p>Threads will be named:
     * {@code warpforge-gpu-<scopeName>-<scopeId>-<operation>-<N>}
     *
     * @param scopeName the scope name
     * @param scopeId the unique scope identifier
     * @param operation the operation name (e.g., "chunk", "stream")
     * @return a ThreadFactory that creates named virtual threads
     */
    public static ThreadFactory forOperation(String scopeName, long scopeId, String operation) {
        String prefix = PREFIX + sanitize(scopeName) + "-" + scopeId + "-" + sanitize(operation) + "-";
        AtomicLong counter = new AtomicLong(0);

        return runnable -> Thread.ofVirtual()
            .name(prefix + counter.getAndIncrement())
            .unstarted(runnable);
    }

    /**
     * Creates a single named virtual thread and starts it.
     *
     * <p>Convenience method for one-off GPU tasks outside a scope.
     *
     * @param name the thread name suffix (will be prefixed with "warpforge-gpu-")
     * @param task the task to run
     * @return the started thread
     */
    public static Thread startNamed(String name, Runnable task) {
        return Thread.ofVirtual()
            .name(PREFIX + sanitize(name))
            .start(task);
    }

    /**
     * Sanitizes a name component for use in thread names.
     *
     * <p>Replaces spaces and special characters with hyphens.
     */
    private static String sanitize(String name) {
        if (name == null || name.isEmpty()) {
            return "unnamed";
        }
        return name.toLowerCase()
            .replaceAll("[^a-z0-9-]", "-")
            .replaceAll("-+", "-")
            .replaceAll("^-|-$", "");
    }
}
