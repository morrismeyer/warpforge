package io.surfworks.warpforge.core.backend;

import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.lang.foreign.MemorySegment;

/**
 * Extended backend interface for GPU execution.
 * Adds GPU-specific capabilities like device memory management and RDMA integration.
 */
public interface GpuBackend extends Backend {

    /**
     * Returns the GPU-specific capabilities.
     */
    GpuBackendCapabilities gpuCapabilities();

    /**
     * Returns the device index (for multi-GPU systems).
     */
    int deviceIndex();

    // ==================== Device Memory Management ====================

    /**
     * Allocate a tensor directly in GPU device memory.
     *
     * @param spec The tensor specification
     * @return A tensor backed by GPU memory
     */
    Tensor allocateDevice(TensorSpec spec);

    /**
     * Copy a tensor from host to device.
     *
     * @param hostTensor The tensor in host memory
     * @return A new tensor in device memory with copied data
     */
    Tensor copyToDevice(Tensor hostTensor);

    /**
     * Copy a tensor from device to host.
     *
     * @param deviceTensor The tensor in device memory
     * @return A new tensor in host memory with copied data
     */
    Tensor copyToHost(Tensor deviceTensor);

    /**
     * Copy data asynchronously from host to device.
     *
     * @param hostTensor The tensor in host memory
     * @param stream The CUDA/HIP stream for async execution (0 for default)
     * @return A new tensor in device memory (copy may not be complete until sync)
     */
    Tensor copyToDeviceAsync(Tensor hostTensor, long stream);

    /**
     * Copy data asynchronously from device to host.
     *
     * @param deviceTensor The tensor in device memory
     * @param stream The CUDA/HIP stream for async execution
     * @return A new tensor in host memory (copy may not be complete until sync)
     */
    Tensor copyToHostAsync(Tensor deviceTensor, long stream);

    // ==================== Zero-Copy RDMA Integration ====================

    /**
     * Register a host MemorySegment for zero-copy access from GPU.
     * This enables GPUDirect RDMA - data can be transferred directly
     * from the RDMA NIC to GPU memory without CPU involvement.
     *
     * <p>The returned tensor shares the same underlying memory as the segment.
     * Modifications via RDMA will be visible to the GPU.</p>
     *
     * @param segment The host memory segment (must be page-aligned for best performance)
     * @param spec The tensor specification
     * @return A tensor backed by the registered memory
     * @throws UnsupportedOperationException if GPUDirect RDMA is not supported
     */
    Tensor registerForRdma(MemorySegment segment, TensorSpec spec);

    /**
     * Allocate pinned host memory that can be used for zero-copy RDMA transfers.
     * Pinned memory provides faster host-device transfers and enables GPUDirect.
     *
     * @param spec The tensor specification
     * @return A tensor backed by pinned host memory
     */
    Tensor allocatePinned(TensorSpec spec);

    /**
     * Check if a tensor is in device memory.
     *
     * @param tensor The tensor to check
     * @return true if the tensor is in device memory
     */
    boolean isDeviceTensor(Tensor tensor);

    /**
     * Check if a tensor is in pinned host memory.
     *
     * @param tensor The tensor to check
     * @return true if the tensor is in pinned memory
     */
    boolean isPinnedTensor(Tensor tensor);

    // ==================== Stream/Queue Management ====================

    /**
     * Create a new execution stream/queue.
     *
     * @return A handle to the new stream
     */
    long createStream();

    /**
     * Destroy an execution stream.
     *
     * @param stream The stream handle
     */
    void destroyStream(long stream);

    /**
     * Synchronize with a stream (wait for all operations to complete).
     *
     * @param stream The stream handle (0 for default stream)
     */
    void synchronizeStream(long stream);

    /**
     * Synchronize the entire device (wait for all streams to complete).
     */
    void synchronizeDevice();

    // ==================== Memory Info ====================

    /**
     * Get total device memory in bytes.
     */
    long totalDeviceMemory();

    /**
     * Get free device memory in bytes.
     */
    long freeDeviceMemory();

    /**
     * Get the amount of memory used by this backend.
     */
    long usedDeviceMemory();
}
