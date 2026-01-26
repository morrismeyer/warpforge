package io.surfworks.warpforge.core.concurrency;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.core.backend.BackendCapabilities;
import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.backend.GpuBackendCapabilities;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Mock GPU backend for unit testing without real GPU hardware.
 *
 * <p>This mock implementation:
 * <ul>
 *   <li>Tracks stream creation/destruction for verification</li>
 *   <li>Records all operations for assertion</li>
 *   <li>Supports configurable failures for error path testing</li>
 *   <li>Is thread-safe for concurrent test execution</li>
 * </ul>
 *
 * <p>Example usage in tests:
 * <pre>{@code
 * MockGpuBackend backend = new MockGpuBackend();
 * try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
 *     scope.forkWithStream(lease -> {
 *         lease.synchronize();
 *         return null;
 *     });
 *     scope.joinAll();
 * }
 * // Verify operations
 * assertEquals(1, backend.streamCreationCount());
 * assertEquals(1, backend.streamDestructionCount());
 * assertTrue(backend.recordedOperations().contains("synchronizeStream:1"));
 * }</pre>
 */
public class MockGpuBackend implements GpuBackend {

    private static final long MOCK_DEVICE_MEMORY = 8L * 1024 * 1024 * 1024; // 8GB

    private final AtomicLong streamIdGenerator = new AtomicLong();
    private final ConcurrentHashMap<Long, Boolean> activeStreams = new ConcurrentHashMap<>();
    private final List<String> operations = Collections.synchronizedList(new ArrayList<>());
    private final int deviceIndex;
    private final String backendName;

    // Configurable failure modes for testing error paths
    private volatile boolean failOnCreateStream = false;
    private volatile boolean failOnSynchronize = false;
    private volatile RuntimeException synchronizeException = null;
    private volatile int synchronizeDelayMs = 0;

    /**
     * Creates a mock backend with default device index 0.
     */
    public MockGpuBackend() {
        this(0, "mock");
    }

    /**
     * Creates a mock backend with specified device index and name.
     *
     * @param deviceIndex the device index to report
     * @param backendName the backend name to report (e.g., "mock", "test-nvidia", "test-amd")
     */
    public MockGpuBackend(int deviceIndex, String backendName) {
        this.deviceIndex = deviceIndex;
        this.backendName = backendName;
        recordOperation("created");
    }

    // ==================== Verification Methods ====================

    /**
     * Returns the number of streams currently active.
     */
    public int activeStreamCount() {
        return activeStreams.size();
    }

    /**
     * Returns the total number of streams created.
     */
    public long streamCreationCount() {
        return streamIdGenerator.get();
    }

    /**
     * Returns the number of streams destroyed.
     */
    public long streamDestructionCount() {
        return streamIdGenerator.get() - activeStreams.size();
    }

    /**
     * Checks if a stream is currently active.
     */
    public boolean isStreamActive(long streamId) {
        return activeStreams.containsKey(streamId);
    }

    /**
     * Returns a copy of all recorded operations.
     */
    public List<String> recordedOperations() {
        return List.copyOf(operations);
    }

    /**
     * Clears recorded operations for clean test state.
     */
    public void clearRecordedOperations() {
        operations.clear();
    }

    // ==================== Failure Configuration ====================

    /**
     * Configures the mock to fail on stream creation.
     */
    public void setFailOnCreateStream(boolean fail) {
        this.failOnCreateStream = fail;
    }

    /**
     * Configures the mock to fail on synchronize calls.
     */
    public void setFailOnSynchronize(boolean fail) {
        this.failOnSynchronize = fail;
    }

    /**
     * Sets a specific exception to throw on synchronize.
     */
    public void setSynchronizeException(RuntimeException e) {
        this.synchronizeException = e;
    }

    /**
     * Sets a delay in milliseconds for synchronize calls.
     */
    public void setSynchronizeDelayMs(int delayMs) {
        this.synchronizeDelayMs = delayMs;
    }

    // ==================== Backend Interface Implementation ====================

    @Override
    public String name() {
        return backendName;
    }

    @Override
    public BackendCapabilities capabilities() {
        return BackendCapabilities.builder()
            .supportedDtypes(Set.of(ScalarType.F32, ScalarType.F64, ScalarType.I32, ScalarType.I64))
            .supportsVectorOps(true)
            .supportsAsync(true)
            .maxTensorRank(8)
            .maxElementCount(Long.MAX_VALUE)
            .build();
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        recordOperation("execute:" + op.getClass().getSimpleName());
        // Mock implementation - just return empty list
        return List.of();
    }

    @Override
    public Tensor allocate(TensorSpec spec) {
        recordOperation("allocate:" + spec);
        return Tensor.zeros(spec.dtype(), spec.shape());
    }

    @Override
    public void close() {
        recordOperation("closed");
        // Clean up any remaining streams
        for (Long streamId : activeStreams.keySet()) {
            recordOperation("cleanup-stream:" + streamId);
        }
        activeStreams.clear();
    }

    // ==================== GpuBackend Interface Implementation ====================

    @Override
    public GpuBackendCapabilities gpuCapabilities() {
        return GpuBackendCapabilities.builder()
            .base(capabilities())
            .deviceMemoryBytes(MOCK_DEVICE_MEMORY)
            .computeUnits(80) // Simulate 80 SMs
            .supportsGpuDirectRdma(true)
            .supportsFp16(true)
            .supportsBf16(true)
            .supportsTensorCores(true)
            .maxThreadsPerBlock(1024)
            .maxSharedMemoryPerBlock(49152)
            .build();
    }

    @Override
    public int deviceIndex() {
        return deviceIndex;
    }

    @Override
    public Tensor allocateDevice(TensorSpec spec) {
        recordOperation("allocateDevice:" + spec);
        return Tensor.zeros(spec.dtype(), spec.shape());
    }

    @Override
    public Tensor copyToDevice(Tensor hostTensor) {
        recordOperation("copyToDevice:" + hostTensor.shape().length + "D");
        return hostTensor; // Mock - just return same tensor
    }

    @Override
    public Tensor copyToHost(Tensor deviceTensor) {
        recordOperation("copyToHost:" + deviceTensor.shape().length + "D");
        return deviceTensor; // Mock - just return same tensor
    }

    @Override
    public Tensor copyToDeviceAsync(Tensor hostTensor, long stream) {
        recordOperation("copyToDeviceAsync:stream=" + stream);
        return hostTensor;
    }

    @Override
    public Tensor copyToHostAsync(Tensor deviceTensor, long stream) {
        recordOperation("copyToHostAsync:stream=" + stream);
        return deviceTensor;
    }

    @Override
    public Tensor registerForRdma(MemorySegment segment, TensorSpec spec) {
        recordOperation("registerForRdma:" + spec);
        return Tensor.zeros(spec.dtype(), spec.shape());
    }

    @Override
    public Tensor allocatePinned(TensorSpec spec) {
        recordOperation("allocatePinned:" + spec);
        return Tensor.zeros(spec.dtype(), spec.shape());
    }

    @Override
    public boolean isDeviceTensor(Tensor tensor) {
        return false; // Mock - all tensors are "host" tensors
    }

    @Override
    public boolean isPinnedTensor(Tensor tensor) {
        return false;
    }

    // ==================== Stream Management ====================

    @Override
    public long createStream() {
        if (failOnCreateStream) {
            throw new RuntimeException("Mock: createStream failed");
        }
        long streamId = streamIdGenerator.incrementAndGet();
        activeStreams.put(streamId, true);
        recordOperation("createStream:" + streamId);
        return streamId;
    }

    @Override
    public void destroyStream(long stream) {
        if (activeStreams.remove(stream) != null) {
            recordOperation("destroyStream:" + stream);
        } else {
            recordOperation("destroyStream:unknown:" + stream);
        }
    }

    @Override
    public void synchronizeStream(long stream) {
        recordOperation("synchronizeStream:" + stream);

        if (synchronizeException != null) {
            throw synchronizeException;
        }

        if (failOnSynchronize) {
            throw new RuntimeException("Mock: synchronizeStream failed");
        }

        if (synchronizeDelayMs > 0) {
            try {
                Thread.sleep(synchronizeDelayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    @Override
    public void synchronizeDevice() {
        recordOperation("synchronizeDevice");
        if (synchronizeDelayMs > 0) {
            try {
                Thread.sleep(synchronizeDelayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    // ==================== Memory Info ====================

    @Override
    public long totalDeviceMemory() {
        return MOCK_DEVICE_MEMORY;
    }

    @Override
    public long freeDeviceMemory() {
        return MOCK_DEVICE_MEMORY; // Mock: always report full memory free
    }

    @Override
    public long usedDeviceMemory() {
        return 0; // Mock: no memory used
    }

    // ==================== Internal ====================

    private void recordOperation(String operation) {
        operations.add(operation);
    }
}
