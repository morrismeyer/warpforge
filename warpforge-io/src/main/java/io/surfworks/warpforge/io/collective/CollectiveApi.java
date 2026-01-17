package io.surfworks.warpforge.io.collective;

import io.surfworks.warpforge.core.tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.CompletableFuture;

/**
 * Collective communication primitives for distributed AI/ML training.
 *
 * <p>This interface provides high-level collective operations backed by
 * UCC (Unified Collective Communications) over UCX transport. All operations
 * are optimized for RDMA when available, falling back to TCP otherwise.
 *
 * <h2>Collective Operations</h2>
 * <ul>
 *   <li><b>AllReduce</b>: Combine tensors from all ranks using a reduction
 *       operation (sum, max, etc.) and distribute the result to all ranks.</li>
 *   <li><b>AllGather</b>: Gather tensors from all ranks and distribute the
 *       concatenated result to all ranks.</li>
 *   <li><b>Broadcast</b>: Send a tensor from one rank (root) to all others.</li>
 *   <li><b>ReduceScatter</b>: Reduce tensors across ranks, then scatter
 *       different portions of the result to each rank.</li>
 *   <li><b>AllToAll</b>: Exchange data between all pairs of ranks.</li>
 *   <li><b>Reduce</b>: Combine tensors to a single rank.</li>
 *   <li><b>Scatter</b>: Distribute portions of a tensor from root to all ranks.</li>
 *   <li><b>Gather</b>: Collect tensors from all ranks to root.</li>
 * </ul>
 *
 * <h2>Zero-Copy Guarantee</h2>
 * <p>When input tensors are backed by RDMA-registered memory (via
 * {@link io.surfworks.warpforge.io.buffer.RegisteredBuffer}), collective
 * operations perform zero-copy transfers. The underlying MemorySegment
 * is used directly for RDMA operations.
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * CollectiveConfig config = CollectiveConfig.of(worldSize, rank, masterAddr, port);
 * try (CollectiveApi collective = Collective.load(config)) {
 *
 *     // Gradient all-reduce during training
 *     Tensor gradients = computeGradients();
 *     Tensor reducedGradients = collective.allReduce(gradients, AllReduceOp.SUM).join();
 *
 *     // Apply gradients (now averaged across all workers)
 *     applyGradients(reducedGradients);
 * }
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 * <p>CollectiveApi instances are thread-safe. Multiple threads can submit
 * collective operations concurrently. Operations are serialized internally
 * to maintain correct ordering semantics.
 *
 * @see Collective#load(CollectiveConfig)
 * @see AllReduceOp
 */
public interface CollectiveApi extends AutoCloseable {

    /**
     * Returns the backend name identifying this implementation.
     *
     * @return backend identifier (e.g., "ucc", "mock")
     */
    String backendName();

    /**
     * Returns the configuration used by this instance.
     *
     * @return collective configuration
     */
    CollectiveConfig config();

    /**
     * Returns the world size (total number of participating ranks).
     *
     * @return number of ranks
     */
    int worldSize();

    /**
     * Returns this process's rank.
     *
     * @return rank (0 to worldSize-1)
     */
    int rank();

    // ===== AllReduce =====

    /**
     * Performs an all-reduce operation on a tensor.
     *
     * <p>Combines tensor data from all ranks using the specified reduction
     * operation, then distributes the result to all ranks.
     *
     * @param input input tensor (same shape on all ranks)
     * @param op reduction operation
     * @return future completing with the reduced tensor
     */
    CompletableFuture<Tensor> allReduce(Tensor input, AllReduceOp op);

    /**
     * Performs an in-place all-reduce operation.
     *
     * <p>The input tensor is modified in place with the reduced result.
     *
     * @param tensor tensor to reduce in place
     * @param op reduction operation
     * @return future completing when operation is done
     */
    CompletableFuture<Void> allReduceInPlace(Tensor tensor, AllReduceOp op);

    /**
     * Performs all-reduce on raw memory.
     *
     * <p>For advanced use when working directly with MemorySegment.
     *
     * @param buffer memory buffer (same size on all ranks)
     * @param count number of elements
     * @param datatype element data type code
     * @param op reduction operation
     * @return future completing when operation is done
     */
    CompletableFuture<Void> allReduceRaw(MemorySegment buffer, long count, int datatype, AllReduceOp op);

    // ===== AllGather =====

    /**
     * Performs an all-gather operation.
     *
     * <p>Gathers tensors from all ranks and concatenates them along the
     * first dimension. Each rank receives the same concatenated result.
     *
     * @param input input tensor (can differ in first dimension across ranks)
     * @return future completing with the gathered tensor
     */
    CompletableFuture<Tensor> allGather(Tensor input);

    /**
     * Performs all-gather with explicit output tensor.
     *
     * @param input input tensor
     * @param output output tensor (pre-allocated, shape = [worldSize * inputDim0, ...])
     * @return future completing when operation is done
     */
    CompletableFuture<Void> allGather(Tensor input, Tensor output);

    // ===== Broadcast =====

    /**
     * Broadcasts a tensor from root rank to all other ranks.
     *
     * @param tensor tensor to broadcast (input on root, output on others)
     * @param root source rank
     * @return future completing with the broadcast tensor
     */
    CompletableFuture<Tensor> broadcast(Tensor tensor, int root);

    /**
     * Broadcasts a tensor in place from root rank.
     *
     * <p>On root rank, the tensor is unchanged. On other ranks, the tensor
     * is overwritten with data from root.
     *
     * @param tensor tensor to broadcast/receive
     * @param root source rank
     * @return future completing when operation is done
     */
    CompletableFuture<Void> broadcastInPlace(Tensor tensor, int root);

    // ===== ReduceScatter =====

    /**
     * Performs a reduce-scatter operation.
     *
     * <p>First reduces tensors across all ranks (like all-reduce), then
     * scatters different portions of the result to each rank. Useful for
     * distributed optimizers.
     *
     * @param input input tensor
     * @param op reduction operation
     * @return future completing with this rank's portion of the result
     */
    CompletableFuture<Tensor> reduceScatter(Tensor input, AllReduceOp op);

    /**
     * Performs reduce-scatter with explicit output tensor.
     *
     * @param input input tensor
     * @param output output tensor (pre-allocated, shape = input.shape / worldSize)
     * @param op reduction operation
     * @return future completing when operation is done
     */
    CompletableFuture<Void> reduceScatter(Tensor input, Tensor output, AllReduceOp op);

    // ===== AllToAll =====

    /**
     * Performs an all-to-all exchange.
     *
     * <p>Each rank sends a portion of its input to every other rank and
     * receives data from all other ranks. The input is split into worldSize
     * equal chunks.
     *
     * @param input input tensor (first dim must be divisible by worldSize)
     * @return future completing with exchanged tensor
     */
    CompletableFuture<Tensor> allToAll(Tensor input);

    /**
     * Performs all-to-all with explicit output tensor.
     *
     * @param input input tensor
     * @param output output tensor (same shape as input)
     * @return future completing when operation is done
     */
    CompletableFuture<Void> allToAll(Tensor input, Tensor output);

    // ===== Reduce (to single rank) =====

    /**
     * Reduces tensors to a single rank.
     *
     * @param input input tensor
     * @param op reduction operation
     * @param root destination rank
     * @return future completing with reduced tensor (valid only on root)
     */
    CompletableFuture<Tensor> reduce(Tensor input, AllReduceOp op, int root);

    // ===== Scatter =====

    /**
     * Scatters portions of a tensor from root to all ranks.
     *
     * @param input input tensor on root (ignored on other ranks)
     * @param root source rank
     * @return future completing with this rank's portion
     */
    CompletableFuture<Tensor> scatter(Tensor input, int root);

    // ===== Gather =====

    /**
     * Gathers tensors from all ranks to root.
     *
     * @param input input tensor
     * @param root destination rank
     * @return future completing with gathered tensor (valid only on root)
     */
    CompletableFuture<Tensor> gather(Tensor input, int root);

    // ===== Synchronization =====

    /**
     * Synchronizes all ranks (barrier).
     *
     * <p>Blocks until all ranks have called barrier.
     *
     * @return future completing when all ranks have synchronized
     */
    CompletableFuture<Void> barrier();

    // ===== Send/Receive (Point-to-Point) =====

    /**
     * Sends a tensor to a specific rank.
     *
     * @param tensor tensor to send
     * @param destRank destination rank
     * @param tag message tag for matching
     * @return future completing when send is initiated
     */
    CompletableFuture<Void> send(Tensor tensor, int destRank, int tag);

    /**
     * Receives a tensor from a specific rank.
     *
     * @param tensor tensor to receive into
     * @param srcRank source rank
     * @param tag message tag for matching
     * @return future completing when receive is complete
     */
    CompletableFuture<Void> recv(Tensor tensor, int srcRank, int tag);

    // ===== Lifecycle =====

    /**
     * Finalizes the collective context, releasing all resources.
     */
    @Override
    void close();

    /**
     * Returns statistics for collective operations.
     */
    CollectiveStats stats();

    /**
     * Statistics for collective operations.
     */
    record CollectiveStats(
            long allReduceCount,
            long allGatherCount,
            long broadcastCount,
            long reduceScatterCount,
            long barrierCount,
            long totalBytesTransferred,
            long totalOperations
    ) {
        public static CollectiveStats zero() {
            return new CollectiveStats(0, 0, 0, 0, 0, 0, 0);
        }
    }
}
