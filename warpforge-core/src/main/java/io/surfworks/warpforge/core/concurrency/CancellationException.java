package io.surfworks.warpforge.core.concurrency;

/**
 * Thrown when a GPU operation is cancelled.
 *
 * <p>This is a clean cancellation signal, distinct from
 * {@link InterruptedException}. It indicates that the operation
 * was intentionally cancelled (e.g., due to timeout, user request,
 * or scope shutdown) rather than encountering an error.
 *
 * <p>Example usage in a time-sliced kernel:
 * <pre>{@code
 * protected T executeChunk(int chunkIndex, int totalChunks,
 *                          List<Tensor> inputs, GpuLease lease) {
 *     checkCancellation();  // Throws CancellationException if interrupted
 *     // ... execute chunk
 * }
 * }</pre>
 *
 * @see TimeSlicedKernel#checkCancellation()
 * @see GpuTaskScope
 */
public class CancellationException extends RuntimeException {

    /**
     * Creates a new CancellationException with the given message.
     *
     * @param message description of why the operation was cancelled
     */
    public CancellationException(String message) {
        super(message);
    }

    /**
     * Creates a new CancellationException with a message and cause.
     *
     * @param message description of why the operation was cancelled
     * @param cause the underlying cause of the cancellation
     */
    public CancellationException(String message, Throwable cause) {
        super(message, cause);
    }
}
