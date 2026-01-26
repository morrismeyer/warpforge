package io.surfworks.warpforge.core.concurrency;

import java.util.concurrent.StructuredTaskScope;

/**
 * A task forked within a {@link GpuTaskScope}.
 *
 * <p>GpuTask wraps a {@link StructuredTaskScope.Subtask} with GPU-specific
 * semantics, providing access to the task's result, state, and optional
 * stream lease.
 *
 * <p>Example usage:
 * <pre>{@code
 * try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
 *     GpuTask<Tensor> task = scope.fork(() -> computeTensor(input));
 *     scope.joinAll();
 *
 *     if (task.isSuccess()) {
 *         Tensor result = task.get();
 *         // Use result
 *     }
 * }
 * }</pre>
 *
 * @param <T> the result type of the task
 * @see GpuTaskScope#fork(java.util.concurrent.Callable)
 * @see GpuTaskScope#forkWithStream(java.util.function.Function)
 */
public final class GpuTask<T> {

    private final StructuredTaskScope.Subtask<T> subtask;
    private final GpuLease lease;

    GpuTask(StructuredTaskScope.Subtask<T> subtask, GpuLease lease) {
        this.subtask = subtask;
        this.lease = lease;
    }

    /**
     * Returns the result of this task.
     *
     * <p>This method should only be called after {@link GpuTaskScope#joinAll()}
     * has completed successfully.
     *
     * @return the task result
     * @throws IllegalStateException if the task has not completed or failed
     */
    public T get() {
        return subtask.get();
    }

    /**
     * Returns the current state of this task.
     *
     * @return the task state (UNAVAILABLE, SUCCESS, or FAILED)
     */
    public StructuredTaskScope.Subtask.State state() {
        return subtask.state();
    }

    /**
     * Returns true if this task completed successfully.
     */
    public boolean isSuccess() {
        return subtask.state() == StructuredTaskScope.Subtask.State.SUCCESS;
    }

    /**
     * Returns true if this task failed with an exception.
     */
    public boolean isFailed() {
        return subtask.state() == StructuredTaskScope.Subtask.State.FAILED;
    }

    /**
     * Returns true if this task has not yet completed.
     */
    public boolean isUnavailable() {
        return subtask.state() == StructuredTaskScope.Subtask.State.UNAVAILABLE;
    }

    /**
     * Returns the exception that caused this task to fail, if any.
     *
     * @return the exception, or null if the task succeeded or hasn't completed
     */
    public Throwable exception() {
        if (isFailed()) {
            return subtask.exception();
        }
        return null;
    }

    /**
     * Returns the GPU stream lease associated with this task, if any.
     *
     * <p>Only tasks created via {@link GpuTaskScope#forkWithStream} will have
     * a lease. Tasks created via {@link GpuTaskScope#fork} will return null.
     *
     * @return the lease, or null if this task doesn't have a dedicated stream
     */
    public GpuLease lease() {
        return lease;
    }
}
