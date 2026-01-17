package io.surfworks.warpforge.backend.nvidia;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;
import io.surfworks.warpforge.io.buffer.RegisteredBuffer;
import io.surfworks.warpforge.io.rdma.RdmaApi;

import java.lang.foreign.MemorySegment;

/**
 * Buffer that enables zero-copy RDMA-to-GPU transfers using GPUDirect.
 *
 * <p>This class bridges the warpforge-io RDMA layer with the NVIDIA GPU backend,
 * allowing tensors to be transferred directly from the network interface to
 * GPU memory without intermediate CPU copies.</p>
 *
 * <h2>Data Flow</h2>
 * <pre>{@code
 * Remote Node                Local Node
 * ┌─────────┐               ┌─────────────────────────────────┐
 * │   GPU   │               │  Mellanox NIC                   │
 * │  data   │  RDMA Write   │       │                         │
 * │         │──────────────>│       ▼ (GPUDirect)             │
 * └─────────┘               │  ┌─────────┐                    │
 *                           │  │   GPU   │                    │
 *                           │  │ Memory  │                    │
 *                           │  └─────────┘                    │
 *                           └─────────────────────────────────┘
 * }</pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Create buffer for receiving tensor via RDMA
 * GpuDirectBuffer buffer = GpuDirectBuffer.allocate(
 *     rdma, gpuBackend, TensorSpec.of(ScalarType.F32, 1024, 1024));
 *
 * // Buffer is RDMA-ready and GPU-accessible
 * RdmaBuffer rdmaBuffer = buffer.rdmaBuffer();  // For RDMA operations
 * Tensor gpuTensor = buffer.gpuTensor();        // For GPU compute
 *
 * // After RDMA write completes, tensor is immediately usable on GPU
 * backend.execute(someOp, gpuTensor);
 * }</pre>
 */
public final class GpuDirectBuffer implements AutoCloseable {

    private final RegisteredBuffer rdmaBuffer;
    private final Tensor gpuTensor;
    private final GpuBackend gpuBackend;

    private GpuDirectBuffer(RegisteredBuffer rdmaBuffer, Tensor gpuTensor, GpuBackend gpuBackend) {
        this.rdmaBuffer = rdmaBuffer;
        this.gpuTensor = gpuTensor;
        this.gpuBackend = gpuBackend;
    }

    /**
     * Allocate a new GPUDirect-enabled buffer.
     *
     * @param rdma The RDMA API for memory registration
     * @param gpuBackend The GPU backend for device memory access
     * @param spec The tensor specification
     * @return A new GPUDirect buffer
     * @throws UnsupportedOperationException if GPUDirect is not supported
     */
    public static GpuDirectBuffer allocate(RdmaApi rdma, GpuBackend gpuBackend, TensorSpec spec) {
        if (!gpuBackend.gpuCapabilities().supportsGpuDirectRdma()) {
            throw new UnsupportedOperationException(
                "GPUDirect RDMA not supported by backend: " + gpuBackend.name());
        }

        // 1. Allocate pinned host memory that is GPU-accessible
        Tensor pinnedTensor = gpuBackend.allocatePinned(spec);

        // 2. Register the pinned memory for RDMA
        MemorySegment segment = pinnedTensor.data();
        RegisteredBuffer rdmaBuffer = RegisteredBuffer.wrap(rdma, pinnedTensor);

        // 3. Register with GPU for device access
        Tensor gpuTensor = gpuBackend.registerForRdma(segment, spec);

        return new GpuDirectBuffer(rdmaBuffer, gpuTensor, gpuBackend);
    }

    /**
     * Wrap an existing RegisteredBuffer for GPU access.
     *
     * @param rdmaBuffer The RDMA-registered buffer
     * @param gpuBackend The GPU backend
     * @return A GPUDirect buffer wrapping the existing buffer
     */
    public static GpuDirectBuffer wrap(RegisteredBuffer rdmaBuffer, GpuBackend gpuBackend) {
        if (!gpuBackend.gpuCapabilities().supportsGpuDirectRdma()) {
            throw new UnsupportedOperationException(
                "GPUDirect RDMA not supported by backend: " + gpuBackend.name());
        }

        MemorySegment segment = rdmaBuffer.segment();
        TensorSpec spec = rdmaBuffer.tensor().spec();

        // Register the buffer's memory with the GPU
        Tensor gpuTensor = gpuBackend.registerForRdma(segment, spec);

        return new GpuDirectBuffer(rdmaBuffer, gpuTensor, gpuBackend);
    }

    /**
     * Get the underlying RDMA-registered buffer.
     * Use this for RDMA operations (send, receive, write, read).
     */
    public RegisteredBuffer rdmaBuffer() {
        return rdmaBuffer;
    }

    /**
     * Get the GPU-accessible tensor.
     * Use this for GPU compute operations.
     */
    public Tensor gpuTensor() {
        return gpuTensor;
    }

    /**
     * Get the raw memory segment.
     */
    public MemorySegment segment() {
        return rdmaBuffer.segment();
    }

    /**
     * Get the tensor specification.
     */
    public TensorSpec spec() {
        return gpuTensor.spec();
    }

    /**
     * Synchronize GPU access after RDMA transfer.
     * Call this after receiving data via RDMA to ensure the GPU sees the update.
     */
    public void synchronizeForGpu() {
        // TODO: Insert proper memory barriers / cache flushes
        // For GPUDirect, the hardware typically handles coherency,
        // but we may need explicit synchronization in some cases
        gpuBackend.synchronizeDevice();
    }

    /**
     * Synchronize for RDMA after GPU compute.
     * Call this after GPU computation if the result will be sent via RDMA.
     */
    public void synchronizeForRdma() {
        gpuBackend.synchronizeDevice();
        // TODO: Flush GPU caches if needed
    }

    @Override
    public void close() {
        // Close in reverse order of creation
        // Note: gpuTensor is a view, so we don't close it separately
        rdmaBuffer.close();
    }
}
