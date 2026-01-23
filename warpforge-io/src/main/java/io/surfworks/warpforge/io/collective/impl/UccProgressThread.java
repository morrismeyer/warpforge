package io.surfworks.warpforge.io.collective.impl;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import java.util.logging.Logger;

import io.surfworks.warpforge.io.collective.CollectiveException;
import io.surfworks.warpforge.io.ffi.ucc.Ucc;
import io.surfworks.warpforge.io.ffi.ucc.ucc_coll_req;

/**
 * Dedicated thread for driving UCC context progress and polling completions.
 *
 * <p>UCX requires all operations on a worker to be called from the same thread
 * that created the worker. This thread handles all UCC progress driving and
 * completion polling, allowing main application threads to submit operations
 * without blocking.
 *
 * <h2>Architecture</h2>
 * <pre>
 * ┌─────────────────────────────────────┐
 * │  Main Application Thread            │
 * │  - Submits operations via submit()  │
 * │  - Receives CompletableFuture       │
 * ├─────────────────────────────────────┤
 * │  UccProgressThread                  │
 * │  - Drives ucc_context_progress()    │
 * │  - Polls pending request statuses   │
 * │  - Completes CompletableFutures     │
 * └─────────────────────────────────────┘
 * </pre>
 *
 * <h2>Performance Benefits</h2>
 * <ul>
 *   <li>Main thread never blocks during collective operations</li>
 *   <li>Single thread handles UCX affinity requirements</li>
 *   <li>Batched progress driving reduces FFM call overhead</li>
 *   <li>Enables overlapping computation with communication</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * UccProgressThread progress = new UccProgressThread(context);
 * progress.start();
 *
 * // Submit operation and get future
 * CompletableFuture<Void> future = progress.submit(request);
 *
 * // Do other work while operation completes...
 *
 * // Wait for completion
 * future.join();
 *
 * progress.shutdown();
 * }</pre>
 */
public class UccProgressThread extends Thread {

    private static final Logger LOG = Logger.getLogger(UccProgressThread.class.getName());

    /** Maximum pending operations before blocking submitters */
    private static final int MAX_PENDING_OPS = 1024;

    /** How often to call ucc_context_progress when idle (nanoseconds) */
    private static final long IDLE_PROGRESS_INTERVAL_NS = 100_000; // 100 microseconds

    /** Batch size for progress polling - process this many requests per progress call */
    private static final int PROGRESS_BATCH_SIZE = 16;

    private final MemorySegment uccContext;
    private final BlockingQueue<PendingOperation> pendingQueue;
    private final ConcurrentHashMap<Long, PendingOperation> activeOperations;
    private final AtomicLong operationIdGenerator;

    private volatile boolean running = true;
    private volatile boolean shutdown = false;

    /**
     * Create a new progress thread for the given UCC context.
     *
     * @param uccContext the UCC context handle to drive progress for
     */
    public UccProgressThread(MemorySegment uccContext) {
        super("ucc-progress");
        this.uccContext = uccContext;
        this.pendingQueue = new LinkedBlockingQueue<>(MAX_PENDING_OPS);
        this.activeOperations = new ConcurrentHashMap<>();
        this.operationIdGenerator = new AtomicLong();

        // Use daemon thread so it doesn't prevent JVM shutdown
        setDaemon(true);
    }

    /**
     * Submit a UCC collective operation for completion tracking.
     *
     * <p>The operation must already be initialized and posted. This method
     * takes ownership of polling the request status and finalizing it.
     *
     * @param request the UCC request handle from ucc_collective_post
     * @return a future that completes when the operation finishes
     */
    public CompletableFuture<Void> submit(MemorySegment request) {
        if (shutdown) {
            return CompletableFuture.failedFuture(
                new CollectiveException("Progress thread is shut down",
                    CollectiveException.ErrorCode.INVALID_STATE));
        }

        long opId = operationIdGenerator.incrementAndGet();
        CompletableFuture<Void> future = new CompletableFuture<>();
        PendingOperation op = new PendingOperation(opId, request, future);

        try {
            pendingQueue.put(op);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            future.completeExceptionally(
                new CollectiveException("Interrupted while submitting operation",
                    CollectiveException.ErrorCode.COMMUNICATION_ERROR));
        }

        return future;
    }

    /**
     * Submit a UCC collective operation with a result.
     *
     * @param request the UCC request handle
     * @param result the result to return when operation completes
     * @param <T> the result type
     * @return a future that completes with the result when the operation finishes
     */
    public <T> CompletableFuture<T> submitWithResult(MemorySegment request, T result) {
        return submit(request).thenApply(v -> result);
    }

    /**
     * Shutdown the progress thread gracefully.
     *
     * <p>Waits for all pending operations to complete before returning.
     */
    public void shutdown() {
        shutdown = true;
        running = false;

        try {
            // Wait for thread to finish
            join(5000);
            if (isAlive()) {
                LOG.warning("Progress thread did not terminate gracefully");
                interrupt();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public void run() {
        LOG.fine("Progress thread started");

        long lastProgressNs = System.nanoTime();

        while (running || !pendingQueue.isEmpty() || !activeOperations.isEmpty()) {
            try {
                // Process new submissions
                PendingOperation newOp = pendingQueue.poll();
                if (newOp != null) {
                    activeOperations.put(newOp.id, newOp);
                }

                // Drive context progress
                Ucc.ucc_context_progress(uccContext);

                // Poll active operations for completion
                int completedCount = 0;
                for (var entry : activeOperations.entrySet()) {
                    PendingOperation op = entry.getValue();

                    int status = pollRequestStatus(op.request);
                    if (status == UccConstants.OK) {
                        // Operation completed successfully
                        finalizeRequest(op.request);
                        activeOperations.remove(entry.getKey());
                        op.future.complete(null);
                        completedCount++;
                    } else if (status != UccConstants.INPROGRESS) {
                        // Operation failed
                        finalizeRequest(op.request);
                        activeOperations.remove(entry.getKey());
                        op.future.completeExceptionally(
                            new CollectiveException(
                                "Collective operation failed: " + UccConstants.statusToString(status),
                                CollectiveException.ErrorCode.COMMUNICATION_ERROR,
                                status));
                        completedCount++;
                    }

                    // Limit how many we process per iteration
                    if (completedCount >= PROGRESS_BATCH_SIZE) {
                        break;
                    }
                }

                // If idle (no operations), reduce CPU usage
                if (activeOperations.isEmpty() && pendingQueue.isEmpty()) {
                    long nowNs = System.nanoTime();
                    if (nowNs - lastProgressNs < IDLE_PROGRESS_INTERVAL_NS) {
                        Thread.onSpinWait();
                    } else {
                        lastProgressNs = nowNs;
                    }
                }

            } catch (Exception e) {
                LOG.log(Level.WARNING, "Error in progress thread", e);
            }
        }

        // Fail any remaining operations
        for (PendingOperation op : activeOperations.values()) {
            op.future.completeExceptionally(
                new CollectiveException("Progress thread shut down",
                    CollectiveException.ErrorCode.INVALID_STATE));
        }
        for (PendingOperation op : pendingQueue) {
            op.future.completeExceptionally(
                new CollectiveException("Progress thread shut down",
                    CollectiveException.ErrorCode.INVALID_STATE));
        }

        LOG.fine("Progress thread stopped");
    }

    private int pollRequestStatus(MemorySegment request) {
        // Use cached reinterpreted segment to reduce FFM overhead
        MemorySegment req = RequestSegmentCache.getInstance().getReinterpretedRequest(request);
        return ucc_coll_req.status(req);
    }

    private void finalizeRequest(MemorySegment request) {
        try {
            Ucc.ucc_collective_finalize(request);
            // Invalidate cache entry after request is finalized
            RequestSegmentCache.getInstance().invalidate(request);
        } catch (Exception e) {
            LOG.warning("Error finalizing request: " + e.getMessage());
        }
    }

    /**
     * Check if the progress thread is running.
     */
    public boolean isRunning() {
        return running && isAlive();
    }

    /**
     * Get the number of pending operations.
     */
    public int pendingCount() {
        return pendingQueue.size() + activeOperations.size();
    }

    /**
     * Internal record for tracking pending operations.
     */
    private record PendingOperation(
        long id,
        MemorySegment request,
        CompletableFuture<Void> future
    ) {}
}
