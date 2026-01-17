package io.surfworks.warpforge.io.collective.mock;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.io.collective.AllReduceOp;
import io.surfworks.warpforge.io.collective.CollectiveApi;
import io.surfworks.warpforge.io.collective.CollectiveConfig;
import io.surfworks.warpforge.io.collective.CollectiveException;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Mock implementation of CollectiveApi for testing.
 *
 * <p>This implementation simulates collective operations locally without
 * actual distributed communication. It's used for:
 * <ul>
 *   <li>Unit testing on systems without UCC/RDMA</li>
 *   <li>Development on macOS</li>
 *   <li>Single-process testing of collective semantics</li>
 * </ul>
 *
 * <h2>Behavior</h2>
 * <p>For a single rank (worldSize=1), operations behave as identity:
 * <ul>
 *   <li>AllReduce: returns input unchanged</li>
 *   <li>Broadcast: returns input unchanged</li>
 *   <li>Barrier: returns immediately</li>
 * </ul>
 */
public class CollectiveMock implements CollectiveApi {

    private final CollectiveConfig config;
    private final AtomicLong allReduceCount = new AtomicLong();
    private final AtomicLong allGatherCount = new AtomicLong();
    private final AtomicLong broadcastCount = new AtomicLong();
    private final AtomicLong reduceScatterCount = new AtomicLong();
    private final AtomicLong barrierCount = new AtomicLong();
    private final AtomicLong totalBytes = new AtomicLong();
    private final AtomicLong totalOps = new AtomicLong();

    private volatile boolean closed = false;

    public CollectiveMock(CollectiveConfig config) {
        this.config = config;
        if (config.rank() < 0 || config.rank() >= config.worldSize()) {
            throw CollectiveException.invalidRank(config.rank(), config.worldSize());
        }
    }

    @Override
    public String backendName() {
        return "mock";
    }

    @Override
    public CollectiveConfig config() {
        return config;
    }

    @Override
    public int worldSize() {
        return config.worldSize();
    }

    @Override
    public int rank() {
        return config.rank();
    }

    @Override
    public CompletableFuture<Tensor> allReduce(Tensor input, AllReduceOp op) {
        checkNotClosed();
        allReduceCount.incrementAndGet();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize());

        // For mock, just return a copy of the input
        // In real distributed case, this would reduce across ranks
        Tensor result = Tensor.zeros(input.dtype(), input.shape());
        MemorySegment.copy(input.data(), 0, result.data(), 0, input.spec().byteSize());
        return CompletableFuture.completedFuture(result);
    }

    @Override
    public CompletableFuture<Void> allReduceInPlace(Tensor tensor, AllReduceOp op) {
        checkNotClosed();
        allReduceCount.incrementAndGet();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(tensor.spec().byteSize());
        // In-place mock: no change needed for single rank
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Void> allReduceRaw(MemorySegment buffer, long count, int datatype, AllReduceOp op) {
        checkNotClosed();
        allReduceCount.incrementAndGet();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(buffer.byteSize());
        // Raw mock: no change needed for single rank
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Tensor> allGather(Tensor input) {
        checkNotClosed();
        allGatherCount.incrementAndGet();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize() * config.worldSize());

        // For mock single rank, create a tensor worldSize times larger in first dim
        int[] newShape = input.shape().clone();
        newShape[0] *= config.worldSize();
        Tensor result = Tensor.zeros(input.dtype(), newShape);

        // Copy input to each "rank's" portion
        long chunkSize = input.spec().byteSize();
        for (int r = 0; r < config.worldSize(); r++) {
            MemorySegment.copy(input.data(), 0, result.data(), r * chunkSize, chunkSize);
        }

        return CompletableFuture.completedFuture(result);
    }

    @Override
    public CompletableFuture<Void> allGather(Tensor input, Tensor output) {
        checkNotClosed();
        allGatherCount.incrementAndGet();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize() * config.worldSize());

        // Copy input to each "rank's" portion of output
        long chunkSize = input.spec().byteSize();
        for (int r = 0; r < config.worldSize(); r++) {
            MemorySegment.copy(input.data(), 0, output.data(), r * chunkSize, chunkSize);
        }
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Tensor> broadcast(Tensor tensor, int root) {
        checkNotClosed();
        validateRank(root);
        broadcastCount.incrementAndGet();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(tensor.spec().byteSize());

        // For mock, just return a copy
        Tensor result = Tensor.zeros(tensor.dtype(), tensor.shape());
        MemorySegment.copy(tensor.data(), 0, result.data(), 0, tensor.spec().byteSize());
        return CompletableFuture.completedFuture(result);
    }

    @Override
    public CompletableFuture<Void> broadcastInPlace(Tensor tensor, int root) {
        checkNotClosed();
        validateRank(root);
        broadcastCount.incrementAndGet();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(tensor.spec().byteSize());
        // In-place mock: no change needed
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Tensor> reduceScatter(Tensor input, AllReduceOp op) {
        checkNotClosed();
        reduceScatterCount.incrementAndGet();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize());

        // For mock, return the portion that would go to this rank
        int[] newShape = input.shape().clone();
        newShape[0] /= config.worldSize();
        Tensor result = Tensor.zeros(input.dtype(), newShape);

        long chunkSize = result.spec().byteSize();
        long offset = config.rank() * chunkSize;
        MemorySegment.copy(input.data(), offset, result.data(), 0, chunkSize);

        return CompletableFuture.completedFuture(result);
    }

    @Override
    public CompletableFuture<Void> reduceScatter(Tensor input, Tensor output, AllReduceOp op) {
        checkNotClosed();
        reduceScatterCount.incrementAndGet();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize());

        long chunkSize = output.spec().byteSize();
        long offset = config.rank() * chunkSize;
        MemorySegment.copy(input.data(), offset, output.data(), 0, chunkSize);
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Tensor> allToAll(Tensor input) {
        checkNotClosed();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize());

        // For mock single rank, just return a copy
        Tensor result = Tensor.zeros(input.dtype(), input.shape());
        MemorySegment.copy(input.data(), 0, result.data(), 0, input.spec().byteSize());
        return CompletableFuture.completedFuture(result);
    }

    @Override
    public CompletableFuture<Void> allToAll(Tensor input, Tensor output) {
        checkNotClosed();
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize());
        MemorySegment.copy(input.data(), 0, output.data(), 0, input.spec().byteSize());
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Tensor> reduce(Tensor input, AllReduceOp op, int root) {
        checkNotClosed();
        validateRank(root);
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize());

        // For mock, return copy only on root
        if (config.rank() == root) {
            Tensor result = Tensor.zeros(input.dtype(), input.shape());
            MemorySegment.copy(input.data(), 0, result.data(), 0, input.spec().byteSize());
            return CompletableFuture.completedFuture(result);
        } else {
            // Non-root ranks get empty tensor
            return CompletableFuture.completedFuture(Tensor.zeros(input.dtype(), new int[]{0}));
        }
    }

    @Override
    public CompletableFuture<Tensor> scatter(Tensor input, int root) {
        checkNotClosed();
        validateRank(root);
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize() / config.worldSize());

        // For mock, return this rank's portion
        int[] newShape = input.shape().clone();
        newShape[0] /= config.worldSize();
        Tensor result = Tensor.zeros(input.dtype(), newShape);

        long chunkSize = result.spec().byteSize();
        long offset = config.rank() * chunkSize;
        MemorySegment.copy(input.data(), offset, result.data(), 0, chunkSize);

        return CompletableFuture.completedFuture(result);
    }

    @Override
    public CompletableFuture<Tensor> gather(Tensor input, int root) {
        checkNotClosed();
        validateRank(root);
        totalOps.incrementAndGet();
        totalBytes.addAndGet(input.spec().byteSize());

        // For mock, only root gets the gathered tensor
        if (config.rank() == root) {
            int[] newShape = input.shape().clone();
            newShape[0] *= config.worldSize();
            Tensor result = Tensor.zeros(input.dtype(), newShape);

            // Copy input to each "rank's" portion
            long chunkSize = input.spec().byteSize();
            for (int r = 0; r < config.worldSize(); r++) {
                MemorySegment.copy(input.data(), 0, result.data(), r * chunkSize, chunkSize);
            }
            return CompletableFuture.completedFuture(result);
        } else {
            return CompletableFuture.completedFuture(Tensor.zeros(input.dtype(), new int[]{0}));
        }
    }

    @Override
    public CompletableFuture<Void> barrier() {
        checkNotClosed();
        barrierCount.incrementAndGet();
        totalOps.incrementAndGet();
        // Mock barrier completes immediately
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Void> send(Tensor tensor, int destRank, int tag) {
        checkNotClosed();
        validateRank(destRank);
        totalOps.incrementAndGet();
        totalBytes.addAndGet(tensor.spec().byteSize());
        // Mock send: no-op for single process
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CompletableFuture<Void> recv(Tensor tensor, int srcRank, int tag) {
        checkNotClosed();
        validateRank(srcRank);
        totalOps.incrementAndGet();
        totalBytes.addAndGet(tensor.spec().byteSize());
        // Mock recv: no-op for single process
        return CompletableFuture.completedFuture(null);
    }

    @Override
    public CollectiveStats stats() {
        return new CollectiveStats(
                allReduceCount.get(),
                allGatherCount.get(),
                broadcastCount.get(),
                reduceScatterCount.get(),
                barrierCount.get(),
                totalBytes.get(),
                totalOps.get()
        );
    }

    @Override
    public void close() {
        closed = true;
    }

    private void checkNotClosed() {
        if (closed) {
            throw new CollectiveException("Collective context has been closed",
                    CollectiveException.ErrorCode.INVALID_STATE);
        }
    }

    private void validateRank(int rank) {
        if (rank < 0 || rank >= config.worldSize()) {
            throw CollectiveException.invalidRank(rank, config.worldSize());
        }
    }
}
