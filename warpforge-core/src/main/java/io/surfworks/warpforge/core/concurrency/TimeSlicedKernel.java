package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class for GPU kernels executed in time-bounded slices.
 *
 * <p>Long-running GPU operations (large matmul, reduce over millions of elements)
 * can block other work and prevent responsive cancellation. TimeSlicedKernel
 * breaks these into chunks with explicit synchronization between slices,
 * enabling:
 * <ul>
 *   <li>Responsive preemption via cancellation checks between chunks</li>
 *   <li>Prevents GPU starvation by yielding between slices</li>
 *   <li>Better resource utilization by allowing other work to interleave</li>
 * </ul>
 *
 * <p>This pattern is validated by research:
 * <ul>
 *   <li>PipeFill (MLSys 2025): 63% utilization increase filling pipeline bubbles</li>
 *   <li>Orion (EuroSys 2024): 7.3x throughput via interference-aware scheduling</li>
 * </ul>
 *
 * <p>Example implementation:
 * <pre>{@code
 * public class SlicedMatmul extends TimeSlicedKernel<Tensor> {
 *
 *     private static final int ROWS_PER_CHUNK = 512;
 *
 *     @Override
 *     protected int estimateChunks(List<Tensor> inputs) {
 *         int M = inputs.get(0).shape()[0];  // Rows of A
 *         return Math.max(1, (M + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK);
 *     }
 *
 *     @Override
 *     protected Tensor executeChunk(int chunkIndex, int totalChunks,
 *                                    List<Tensor> inputs, GpuLease lease) {
 *         Tensor A = inputs.get(0);
 *         Tensor B = inputs.get(1);
 *
 *         int M = A.shape()[0];
 *         int startRow = chunkIndex * ROWS_PER_CHUNK;
 *         int endRow = Math.min(startRow + ROWS_PER_CHUNK, M);
 *
 *         // Compute partial result for rows [startRow, endRow)
 *         return backend.gemm(A.sliceRows(startRow, endRow), B, lease);
 *     }
 *
 *     @Override
 *     protected Tensor mergeResults(List<Tensor> chunks) {
 *         return Tensor.concatenate(0, chunks);  // Stack along row dimension
 *     }
 * }
 * }</pre>
 *
 * @param <T> the result type of the kernel
 * @see GpuTaskScope
 * @see GpuLease
 */
public abstract class TimeSlicedKernel<T> {

    /**
     * Executes this kernel within the given scope.
     *
     * <p>The kernel is automatically divided into chunks based on
     * {@link #estimateChunks(List)}. Each chunk runs on a dedicated stream
     * via {@link GpuTaskScope#forkWithStream}, with synchronization after
     * each chunk to provide yield points.
     *
     * @param scope the GPU task scope for forking chunks
     * @param inputs the input tensors for the operation
     * @return the merged result from all chunks
     * @throws CancellationException if cancelled between chunks
     * @throws RuntimeException if any chunk fails
     */
    public final T execute(GpuTaskScope scope, List<Tensor> inputs) {
        int numChunks = estimateChunks(inputs);
        List<GpuTask<T>> chunkTasks = new ArrayList<>();

        for (int i = 0; i < numChunks; i++) {
            int chunk = i;
            chunkTasks.add(scope.forkWithStream(lease -> {
                checkCancellation();
                T result = executeChunk(chunk, numChunks, inputs, lease);
                lease.synchronize(); // Yield point - allows other work to run
                return result;
            }));
        }

        try {
            scope.joinAll();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new CancellationException("Time-sliced kernel interrupted");
        } catch (Exception e) {
            throw new RuntimeException("Time-sliced kernel failed", e);
        }

        return mergeResults(chunkTasks.stream()
                .map(GpuTask::get)
                .toList());
    }

    /**
     * Estimates the number of chunks to divide this operation into.
     *
     * <p>Override this to provide operation-specific chunking logic.
     * Consider factors like:
     * <ul>
     *   <li>Input tensor sizes</li>
     *   <li>Target chunk duration (typically 10-50ms for responsive preemption)</li>
     *   <li>Memory constraints per chunk</li>
     * </ul>
     *
     * @param inputs the input tensors
     * @return the number of chunks (must be &gt;= 1)
     */
    protected abstract int estimateChunks(List<Tensor> inputs);

    /**
     * Executes a single chunk of the operation.
     *
     * <p>This method is called on a virtual thread with a dedicated GPU stream.
     * The lease should be used for all GPU operations in this chunk.
     *
     * @param chunkIndex the zero-based index of this chunk
     * @param totalChunks the total number of chunks
     * @param inputs the input tensors (shared across all chunks)
     * @param lease the GPU stream lease for this chunk
     * @return the partial result for this chunk
     */
    protected abstract T executeChunk(int chunkIndex, int totalChunks,
                                       List<Tensor> inputs, GpuLease lease);

    /**
     * Merges the results from all chunks into the final result.
     *
     * @param chunks the partial results from each chunk, in order
     * @return the merged final result
     */
    protected abstract T mergeResults(List<T> chunks);

    /**
     * Checks if the current thread has been interrupted.
     *
     * <p>Call this at the start of each chunk to support responsive cancellation.
     *
     * @throws CancellationException if the thread has been interrupted
     */
    protected void checkCancellation() {
        if (Thread.currentThread().isInterrupted()) {
            throw new CancellationException("Kernel execution cancelled");
        }
    }
}
