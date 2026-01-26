package io.surfworks.warpforge.core.concurrency;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.core.backend.BackendCapabilities;
import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.backend.GpuBackendCapabilities;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Mock GPU backend for testing structured concurrency classes.
 *
 * <p>This mock tracks stream creation/destruction to verify proper cleanup.
 */
class MockGpuBackend implements GpuBackend {

    private final AtomicLong streamIdGenerator = new AtomicLong(1000);
    private final ConcurrentHashMap<Long, Boolean> activeStreams = new ConcurrentHashMap<>();
    private final int deviceIndex;

    MockGpuBackend() {
        this(0);
    }

    MockGpuBackend(int deviceIndex) {
        this.deviceIndex = deviceIndex;
    }

    @Override
    public long createStream() {
        long streamId = streamIdGenerator.incrementAndGet();
        activeStreams.put(streamId, true);
        return streamId;
    }

    @Override
    public void destroyStream(long stream) {
        activeStreams.remove(stream);
    }

    @Override
    public void synchronizeStream(long stream) {
        // Simulate synchronization delay
        try {
            Thread.sleep(1);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public void synchronizeDevice() {
        // No-op for mock
    }

    @Override
    public int deviceIndex() {
        return deviceIndex;
    }

    /**
     * Returns the number of currently active streams.
     * Used by tests to verify cleanup.
     */
    int activeStreamCount() {
        return activeStreams.size();
    }

    /**
     * Returns true if the given stream is still active.
     */
    boolean isStreamActive(long streamId) {
        return activeStreams.containsKey(streamId);
    }

    // ==================== GpuBackend methods ====================

    @Override
    public GpuBackendCapabilities gpuCapabilities() {
        return GpuBackendCapabilities.builder()
            .base(BackendCapabilities.cpu())
            .deviceMemoryBytes(8L * 1024 * 1024 * 1024)
            .computeUnits(80)
            .supportsGpuDirectRdma(false)
            .supportsFp16(true)
            .supportsBf16(true)
            .supportsTensorCores(false)
            .build();
    }

    @Override
    public Tensor allocateDevice(TensorSpec spec) {
        return Tensor.zeros(spec.dtype(), spec.shape());
    }

    @Override
    public Tensor copyToDevice(Tensor hostTensor) {
        return hostTensor;
    }

    @Override
    public Tensor copyToHost(Tensor deviceTensor) {
        return deviceTensor;
    }

    @Override
    public Tensor copyToDeviceAsync(Tensor hostTensor, long stream) {
        return hostTensor;
    }

    @Override
    public Tensor copyToHostAsync(Tensor deviceTensor, long stream) {
        return deviceTensor;
    }

    @Override
    public Tensor registerForRdma(MemorySegment segment, TensorSpec spec) {
        throw new UnsupportedOperationException("Mock does not support RDMA");
    }

    @Override
    public Tensor allocatePinned(TensorSpec spec) {
        return Tensor.zeros(spec.dtype(), spec.shape());
    }

    @Override
    public boolean isDeviceTensor(Tensor tensor) {
        return false;
    }

    @Override
    public boolean isPinnedTensor(Tensor tensor) {
        return false;
    }

    @Override
    public long totalDeviceMemory() {
        return 8L * 1024 * 1024 * 1024;
    }

    @Override
    public long freeDeviceMemory() {
        return 6L * 1024 * 1024 * 1024;
    }

    @Override
    public long usedDeviceMemory() {
        return 2L * 1024 * 1024 * 1024;
    }

    // ==================== Backend interface methods ====================

    @Override
    public String name() {
        return "mock";
    }

    @Override
    public BackendCapabilities capabilities() {
        return BackendCapabilities.cpu();
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        throw new UnsupportedOperationException("Mock does not execute operations");
    }

    @Override
    public Tensor allocate(TensorSpec spec) {
        return Tensor.zeros(spec.dtype(), spec.shape());
    }

    @Override
    public void close() {
        // No-op for mock
    }
}
