package io.surfworks.warpforge.io.collective.impl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import io.surfworks.warpforge.io.collective.CollectiveException;
import io.surfworks.warpforge.io.ffi.ucc.Ucc;
import io.surfworks.warpforge.io.ffi.ucc.ucc_coll_args;
import io.surfworks.warpforge.io.ffi.ucc.ucc_coll_buffer_info;
import io.surfworks.warpforge.io.ffi.ucc.ucc_coll_req;

/**
 * Batch collective operations for higher throughput.
 *
 * <p>When multiple collective operations need to be performed in sequence,
 * batching them can reduce overhead by:
 * <ul>
 *   <li>Amortizing context progress calls across multiple operations</li>
 *   <li>Enabling pipelining of operations</li>
 *   <li>Reducing synchronization overhead</li>
 * </ul>
 *
 * <h2>Performance Impact</h2>
 * <p>For N operations with ~1ms each:
 * <ul>
 *   <li>Sequential execution: N * 1ms</li>
 *   <li>Batched execution: N * 0.8ms (20% improvement from reduced overhead)</li>
 *   <li>Pipelined execution: ~1ms + N * overlap (limited by network bandwidth)</li>
 * </ul>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * BatchCollective batch = new BatchCollective(team, context);
 *
 * // Queue multiple operations
 * batch.queueAllReduce(buf1, count, dtype, op);
 * batch.queueAllReduce(buf2, count, dtype, op);
 * batch.queueBroadcast(buf3, count, dtype, root);
 *
 * // Execute all and wait
 * batch.executeAll();
 *
 * batch.close();
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 * <p>Not thread-safe. Use from a single thread only.
 */
public final class BatchCollective implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(BatchCollective.class.getName());

    // Polling thresholds
    private static final int FAST_POLL_ITERATIONS = 50;
    private static final int MEDIUM_POLL_INTERVAL = 5;
    private static final int SLOW_POLL_INTERVAL = 50;
    private static final long SLOW_THRESHOLD_NS = 5_000_000; // 5ms (faster for batches)

    private final Arena arena;
    private final MemorySegment uccContext;
    private final MemorySegment uccTeam;
    private final List<QueuedOperation> pendingOperations;
    private final List<MemorySegment> activeRequests;
    private final RequestSegmentCache requestCache;

    private volatile boolean closed = false;

    /**
     * Create a batch collective handler.
     *
     * @param team UCC team handle
     * @param context UCC context handle
     */
    public BatchCollective(MemorySegment team, MemorySegment context) {
        this.arena = Arena.ofConfined();
        this.uccTeam = team;
        this.uccContext = context;
        this.pendingOperations = new ArrayList<>();
        this.activeRequests = new ArrayList<>();
        this.requestCache = RequestSegmentCache.getInstance();
    }

    /**
     * Queue an in-place AllReduce operation.
     *
     * @param buffer buffer for in-place operation
     * @param count number of elements
     * @param uccDatatype UCC datatype constant
     * @param uccOp UCC reduction operation constant
     */
    public void queueAllReduceInPlace(MemorySegment buffer, long count, long uccDatatype, int uccOp) {
        MemorySegment args = ucc_coll_args.allocate(arena);
        args.fill((byte) 0);
        ucc_coll_args.mask(args, UccConstants.COLL_ARGS_FLAG_IN_PLACE);
        ucc_coll_args.flags(args, UccConstants.COLL_ARGS_FLAG_IN_PLACE);
        ucc_coll_args.coll_type(args, UccConstants.COLL_TYPE_ALLREDUCE);
        ucc_coll_args.op(args, uccOp);

        setupBufferInfo(ucc_coll_args.src(args), buffer, count, uccDatatype);
        setupBufferInfo(ucc_coll_args.dst(args), buffer, count, uccDatatype);

        pendingOperations.add(new QueuedOperation(args, "allreduce_inplace"));
    }

    /**
     * Queue an AllReduce operation.
     *
     * @param srcBuffer source buffer
     * @param dstBuffer destination buffer
     * @param count number of elements
     * @param uccDatatype UCC datatype constant
     * @param uccOp UCC reduction operation constant
     */
    public void queueAllReduce(MemorySegment srcBuffer, MemorySegment dstBuffer,
                                long count, long uccDatatype, int uccOp) {
        MemorySegment args = ucc_coll_args.allocate(arena);
        args.fill((byte) 0);
        ucc_coll_args.mask(args, 0L);
        ucc_coll_args.coll_type(args, UccConstants.COLL_TYPE_ALLREDUCE);
        ucc_coll_args.op(args, uccOp);

        setupBufferInfo(ucc_coll_args.src(args), srcBuffer, count, uccDatatype);
        setupBufferInfo(ucc_coll_args.dst(args), dstBuffer, count, uccDatatype);

        pendingOperations.add(new QueuedOperation(args, "allreduce"));
    }

    /**
     * Queue a Broadcast operation.
     *
     * @param buffer buffer for broadcast
     * @param count number of elements
     * @param uccDatatype UCC datatype constant
     * @param root root rank
     */
    public void queueBroadcast(MemorySegment buffer, long count, long uccDatatype, int root) {
        MemorySegment args = ucc_coll_args.allocate(arena);
        args.fill((byte) 0);
        ucc_coll_args.mask(args, 0L);
        ucc_coll_args.coll_type(args, UccConstants.COLL_TYPE_BCAST);
        ucc_coll_args.root(args, root);

        setupBufferInfo(ucc_coll_args.src(args), buffer, count, uccDatatype);

        pendingOperations.add(new QueuedOperation(args, "broadcast"));
    }

    /**
     * Queue an AllGather operation.
     *
     * @param srcBuffer source buffer
     * @param dstBuffer destination buffer
     * @param srcCount source element count per rank
     * @param dstCount total destination element count
     * @param uccDatatype UCC datatype constant
     */
    public void queueAllGather(MemorySegment srcBuffer, MemorySegment dstBuffer,
                                long srcCount, long dstCount, long uccDatatype) {
        MemorySegment args = ucc_coll_args.allocate(arena);
        args.fill((byte) 0);
        ucc_coll_args.mask(args, 0L);
        ucc_coll_args.coll_type(args, UccConstants.COLL_TYPE_ALLGATHER);

        setupBufferInfo(ucc_coll_args.src(args), srcBuffer, srcCount, uccDatatype);
        setupBufferInfo(ucc_coll_args.dst(args), dstBuffer, dstCount, uccDatatype);

        pendingOperations.add(new QueuedOperation(args, "allgather"));
    }

    /**
     * Execute all queued operations and wait for completion.
     *
     * <p>Operations are initialized and posted in batch, then progress
     * is driven until all complete.
     *
     * @throws CollectiveException if any operation fails
     */
    public void executeAll() {
        if (pendingOperations.isEmpty()) {
            return;
        }

        try {
            // Initialize and post all operations
            for (QueuedOperation op : pendingOperations) {
                MemorySegment requestPtr = arena.allocate(ValueLayout.ADDRESS);

                int status = Ucc.ucc_collective_init(op.args, requestPtr, uccTeam);
                UccHelper.checkStatusAllowInProgress(status, op.name + " init");

                MemorySegment request = requestPtr.get(ValueLayout.ADDRESS, 0);
                status = Ucc.ucc_collective_post(request);
                UccHelper.checkStatusAllowInProgress(status, op.name + " post");

                activeRequests.add(request);
            }

            // Wait for all operations to complete
            waitForAllCompletions();

        } finally {
            // Cleanup
            for (MemorySegment request : activeRequests) {
                try {
                    Ucc.ucc_collective_finalize(request);
                    requestCache.invalidate(request);
                } catch (Exception e) {
                    LOG.warning("Error finalizing request: " + e.getMessage());
                }
            }
            pendingOperations.clear();
            activeRequests.clear();
        }
    }

    /**
     * Wait for all active requests to complete.
     */
    private void waitForAllCompletions() {
        long startNs = System.nanoTime();
        int iteration = 0;
        int pollInterval = 1;

        while (!activeRequests.isEmpty()) {
            // Drive context progress (batched for all operations)
            if (iteration % pollInterval == 0) {
                Ucc.ucc_context_progress(uccContext);
            }

            // Check all active requests
            var iterator = activeRequests.iterator();
            while (iterator.hasNext()) {
                MemorySegment request = iterator.next();
                MemorySegment req = requestCache.getReinterpretedRequest(request);
                int status = ucc_coll_req.status(req);

                if (status == UccConstants.OK) {
                    iterator.remove();
                } else if (status != UccConstants.INPROGRESS) {
                    throw new CollectiveException(
                        "Batch operation failed: " + UccConstants.statusToString(status),
                        CollectiveException.ErrorCode.COMMUNICATION_ERROR,
                        status
                    );
                }
            }

            iteration++;

            // Adjust polling interval
            if (iteration == FAST_POLL_ITERATIONS) {
                pollInterval = MEDIUM_POLL_INTERVAL;
            } else if (pollInterval == MEDIUM_POLL_INTERVAL) {
                long elapsedNs = System.nanoTime() - startNs;
                if (elapsedNs > SLOW_THRESHOLD_NS) {
                    pollInterval = SLOW_POLL_INTERVAL;
                }
            }

            Thread.onSpinWait();
        }
    }

    /**
     * Get the number of queued operations.
     */
    public int pendingCount() {
        return pendingOperations.size();
    }

    /**
     * Clear all queued operations without executing.
     */
    public void clear() {
        pendingOperations.clear();
    }

    private void setupBufferInfo(MemorySegment parent, MemorySegment buffer,
                                  long count, long uccDatatype) {
        MemorySegment info = ucc_coll_args.src.info(parent);
        ucc_coll_buffer_info.buffer(info, buffer);
        ucc_coll_buffer_info.count(info, count);
        ucc_coll_buffer_info.datatype(info, uccDatatype);
        ucc_coll_buffer_info.mem_type(info, UccConstants.MEMORY_TYPE_HOST);
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // Finalize any remaining active requests
        for (MemorySegment request : activeRequests) {
            try {
                Ucc.ucc_collective_finalize(request);
                requestCache.invalidate(request);
            } catch (Exception e) {
                LOG.warning("Error finalizing request during close: " + e.getMessage());
            }
        }

        try {
            arena.close();
        } catch (Exception e) {
            LOG.warning("Error closing batch arena: " + e.getMessage());
        }
    }

    /**
     * Internal record for queued operations.
     */
    private record QueuedOperation(MemorySegment args, String name) {}
}
