package io.surfworks.warpforge.core.concurrency;

/**
 * A lease on GPU execution resources (stream handle).
 *
 * <p>GpuLease is obtained from a {@link GpuTaskScope} and represents exclusive
 * access to a CUDA/HIP stream. The lease is automatically returned when the
 * scope closes or when explicitly closed via {@link #close()}.
 *
 * <p>Example usage:
 * <pre>{@code
 * try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
 *     scope.forkWithStream(lease -> {
 *         // Execute kernel on this stream
 *         backend.launchKernel(kernel, lease.streamHandle());
 *         lease.synchronize();  // Wait for completion
 *         return result;
 *     });
 *     scope.joinAll();
 * }
 * }</pre>
 *
 * @see GpuTaskScope#forkWithStream(java.util.function.Function)
 */
public final class GpuLease implements AutoCloseable {

    private final long streamHandle;
    private final GpuTaskScope parentScope;
    private final long acquireTimeNanos;

    GpuLease(long streamHandle, GpuTaskScope parentScope, long acquireTimeNanos) {
        this.streamHandle = streamHandle;
        this.parentScope = parentScope;
        this.acquireTimeNanos = acquireTimeNanos;
    }

    /**
     * Returns the native stream handle (CUDA stream or HIP stream).
     *
     * @return the stream handle for use in kernel launches
     */
    public long streamHandle() {
        return streamHandle;
    }

    /**
     * Synchronizes the stream, blocking until all operations complete.
     *
     * <p>This is a yield point for time-sliced kernels, allowing other
     * work to execute between chunks.
     */
    public void synchronize() {
        parentScope.backend().synchronizeStream(streamHandle);
    }

    /**
     * Returns the time (in nanoseconds) when this lease was acquired.
     *
     * <p>Useful for profiling stream utilization.
     */
    public long acquireTimeNanos() {
        return acquireTimeNanos;
    }

    /**
     * Returns the parent scope that owns this lease.
     */
    public GpuTaskScope parentScope() {
        return parentScope;
    }

    /**
     * Releases this lease back to the parent scope.
     *
     * <p>The stream will be destroyed. After calling close(), this lease
     * should not be used.
     */
    @Override
    public void close() {
        parentScope.releaseStream(this);
    }
}
