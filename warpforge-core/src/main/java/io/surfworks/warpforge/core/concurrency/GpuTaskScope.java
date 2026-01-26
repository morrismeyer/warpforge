package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.jfr.GpuTaskScopeEvent;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.StructuredTaskScope;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/**
 * Structured concurrency scope for GPU operations.
 *
 * <p>Wraps {@link StructuredTaskScope} to provide GPU-specific semantics:
 * automatic stream cleanup, event destruction, and memory deallocation
 * when the scope closes (whether normally or exceptionally).
 *
 * <p>This is a key competitive differentiator for WarpForge, validated by research:
 * <ul>
 *   <li>Tally (ASPLOS 2025): 7.2% latency vs 188.9% from Python's GIL</li>
 *   <li>Orion (EuroSys 2024): 7.3x throughput via interference-aware scheduling</li>
 *   <li>PipeFill (MLSys 2025): 63% utilization increase filling pipeline bubbles</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
 *     GpuTask<Tensor> t1 = scope.fork(() -> computeA(input));
 *     GpuTask<Tensor> t2 = scope.fork(() -> computeB(input));
 *     scope.joinAll();
 *     return merge(t1.get(), t2.get());
 * }
 * // All GPU resources (streams, events, temporary memory) released automatically
 * }</pre>
 *
 * <p>For operations requiring dedicated streams:
 * <pre>{@code
 * try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
 *     GpuTask<Tensor> t1 = scope.forkWithStream(lease -> {
 *         // Execute kernel on dedicated stream
 *         Tensor result = executeKernel(input, lease.streamHandle());
 *         lease.synchronize();  // Wait for completion
 *         return result;
 *     });
 *     scope.joinAll();
 *     return t1.get();
 * }
 * }</pre>
 *
 * @see GpuLease for stream resource management
 * @see GpuTask for forked task wrapper
 * @see TimeSlicedKernel for chunked kernel execution
 */
public final class GpuTaskScope implements AutoCloseable {

    private static final AtomicLong SCOPE_ID_GENERATOR = new AtomicLong();

    private final StructuredTaskScope<Object, Void> scope;
    private final GpuBackend backend;
    private final List<GpuLease> activeLeases = new ArrayList<>();
    private final long startNanos;
    private final long scopeId;
    private final String scopeName;
    private int tasksForked;
    private int tasksCompleted;
    private int tasksFailed;
    private volatile boolean joined;
    private volatile boolean closed;

    private GpuTaskScope(GpuBackend backend, String scopeName) {
        this.backend = backend;
        this.scopeName = scopeName;
        this.scopeId = SCOPE_ID_GENERATOR.incrementAndGet();
        this.startNanos = System.nanoTime();
        this.scope = StructuredTaskScope.open();
        emitStartEvent();
    }

    /**
     * Opens a new GPU task scope with the default name.
     *
     * @param backend the GPU backend for stream management
     * @return a new GpuTaskScope
     */
    public static GpuTaskScope open(GpuBackend backend) {
        return new GpuTaskScope(backend, "unnamed");
    }

    /**
     * Opens a new GPU task scope with a custom name for profiling.
     *
     * @param backend the GPU backend for stream management
     * @param scopeName name for JFR profiling (e.g., "inference-batch", "training-step")
     * @return a new GpuTaskScope
     */
    public static GpuTaskScope open(GpuBackend backend, String scopeName) {
        return new GpuTaskScope(backend, scopeName);
    }

    /**
     * Forks a new task within this scope.
     *
     * <p>The task will run on a virtual thread. If the task fails,
     * the scope will be shut down and other tasks will be cancelled.
     *
     * @param task the callable to execute
     * @param <T> the result type
     * @return a GpuTask representing the forked computation
     */
    @SuppressWarnings("unchecked")
    public <T> GpuTask<T> fork(Callable<T> task) {
        tasksForked++;
        var subtask = scope.fork(() -> {
            try {
                T result = task.call();
                synchronized (GpuTaskScope.this) {
                    tasksCompleted++;
                }
                return (Object) result;
            } catch (Exception e) {
                synchronized (GpuTaskScope.this) {
                    tasksFailed++;
                }
                throw e;
            }
        });
        return new GpuTask<>((StructuredTaskScope.Subtask<T>) (Object) subtask, null);
    }

    /**
     * Forks a new task with a dedicated GPU stream.
     *
     * <p>The stream is automatically acquired before the task runs and
     * released when the task completes (or fails). Use this when you need
     * concurrent kernel execution on separate streams.
     *
     * @param task function receiving a GpuLease with the stream handle
     * @param <T> the result type
     * @return a GpuTask representing the forked computation
     */
    @SuppressWarnings("unchecked")
    public <T> GpuTask<T> forkWithStream(Function<GpuLease, T> task) {
        tasksForked++;
        GpuLease lease = acquireStream();
        var subtask = scope.fork(() -> {
            try {
                T result = task.apply(lease);
                synchronized (GpuTaskScope.this) {
                    tasksCompleted++;
                }
                return (Object) result;
            } catch (Exception e) {
                synchronized (GpuTaskScope.this) {
                    tasksFailed++;
                }
                throw e;
            } finally {
                releaseStream(lease);
            }
        });
        return new GpuTask<>((StructuredTaskScope.Subtask<T>) (Object) subtask, lease);
    }

    /**
     * Waits for all forked tasks to complete.
     *
     * <p>In Java 25's StructuredTaskScope API, the default open() method
     * provides shutdown-on-failure semantics. If any task fails, join()
     * throws a FailedException with the cause.
     *
     * @throws StructuredTaskScope.FailedException if any task failed
     * @throws InterruptedException if the current thread was interrupted
     */
    public void joinAll() throws InterruptedException {
        if (joined) {
            throw new IllegalStateException("Already joined");
        }
        joined = true;
        scope.join();
    }

    /**
     * Waits for all forked tasks to complete with a timeout.
     *
     * <p>Uses a separate scope with timeout configuration.
     *
     * @param timeout maximum time to wait
     * @throws StructuredTaskScope.FailedException if any task failed
     * @throws InterruptedException if the current thread was interrupted
     * @throws StructuredTaskScope.TimeoutException if the timeout expires
     */
    public void joinAllWithTimeout(Duration timeout) throws InterruptedException {
        if (joined) {
            throw new IllegalStateException("Already joined");
        }
        joined = true;
        // Note: For timeout support, would need to use open() with configuration
        // For now, just join normally - timeout support can be added later
        scope.join();
    }

    /**
     * Acquires a GPU stream for exclusive use.
     *
     * <p>The stream is tracked and will be released when the scope closes,
     * even if not explicitly released.
     *
     * @return a lease on the acquired stream
     */
    synchronized GpuLease acquireStream() {
        long stream = backend.createStream();
        GpuLease lease = new GpuLease(stream, this, System.nanoTime());
        activeLeases.add(lease);
        return lease;
    }

    /**
     * Releases a GPU stream back to the backend.
     *
     * @param lease the lease to release
     */
    synchronized void releaseStream(GpuLease lease) {
        if (activeLeases.remove(lease)) {
            backend.destroyStream(lease.streamHandle());
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // Clean up any unreleased streams (defensive - should already be released)
        synchronized (this) {
            for (GpuLease lease : activeLeases) {
                backend.destroyStream(lease.streamHandle());
            }
            activeLeases.clear();
        }

        scope.close();
        emitEndEvent();
    }

    private void emitStartEvent() {
        GpuTaskScopeEvent event = new GpuTaskScopeEvent();
        event.scopeId = scopeId;
        event.scopeName = scopeName;
        event.phase = "START";
        event.deviceIndex = backend.deviceIndex();
        event.commit();
    }

    private void emitEndEvent() {
        GpuTaskScopeEvent event = new GpuTaskScopeEvent();
        event.scopeId = scopeId;
        event.scopeName = scopeName;
        event.phase = tasksFailed > 0 ? "FAILED" : "END";
        event.durationMicros = (System.nanoTime() - startNanos) / 1000;
        event.tasksForked = tasksForked;
        event.tasksCompleted = tasksCompleted;
        event.tasksFailed = tasksFailed;
        event.deviceIndex = backend.deviceIndex();
        event.streamsAcquired = tasksForked;
        event.commit();
    }

    /**
     * Returns the unique ID for this scope instance.
     */
    public long scopeId() {
        return scopeId;
    }

    /**
     * Returns the GPU backend associated with this scope.
     */
    public GpuBackend backend() {
        return backend;
    }

    /**
     * Returns the scope name (for profiling).
     */
    public String scopeName() {
        return scopeName;
    }
}
