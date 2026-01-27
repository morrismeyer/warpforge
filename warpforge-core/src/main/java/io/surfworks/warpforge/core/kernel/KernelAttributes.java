package io.surfworks.warpforge.core.kernel;

/**
 * Kernel resource attributes queried from the GPU runtime.
 *
 * <p>This record holds the resource requirements of a compiled GPU kernel,
 * obtained via cudaFuncGetAttributes (NVIDIA) or hipFuncGetAttributes (AMD).
 * These values are used to calculate theoretical occupancy.
 *
 * <p>Resource constraints that limit occupancy:
 * <ul>
 *   <li><b>Registers:</b> Each thread needs registers; total registers per SM is limited</li>
 *   <li><b>Shared memory:</b> Each block needs shared memory; total per SM is limited</li>
 *   <li><b>Block count:</b> Maximum blocks per SM is hardware-limited</li>
 *   <li><b>Warp count:</b> Maximum warps per SM is hardware-limited</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * // Query attributes from CUDA
 * KernelAttributes attrs = CudaRuntime.funcGetAttributes(kernelFunction);
 *
 * // Calculate occupancy
 * OccupancyCalculator calc = new OccupancyCalculator(deviceProps);
 * OccupancyInfo info = calc.calculate(attrs, blockSize, dynamicSharedMem);
 * }</pre>
 *
 * @param sharedSizeBytes static shared memory per block (compiled into kernel)
 * @param localSizeBytes local memory per thread (register spills to L1/global)
 * @param maxThreadsPerBlock maximum threads per block for this kernel
 * @param numRegs registers per thread
 * @param ptxVersion PTX version (NVIDIA) or GCN/RDNA ISA version (AMD)
 * @param binaryVersion binary (cubin/hsaco) version
 *
 * @see OccupancyCalculator
 */
public record KernelAttributes(
    long sharedSizeBytes,
    long localSizeBytes,
    int maxThreadsPerBlock,
    int numRegs,
    int ptxVersion,
    int binaryVersion
) {
    /**
     * Creates a minimal KernelAttributes for testing or simple kernels.
     *
     * @param numRegs registers per thread
     * @param sharedSizeBytes static shared memory per block
     * @return kernel attributes with default values for other fields
     */
    public static KernelAttributes of(int numRegs, long sharedSizeBytes) {
        return new KernelAttributes(sharedSizeBytes, 0, 1024, numRegs, 0, 0);
    }

    /**
     * Returns true if this kernel uses any shared memory.
     */
    public boolean usesSharedMemory() {
        return sharedSizeBytes > 0;
    }

    /**
     * Returns true if this kernel has register spills (uses local memory).
     */
    public boolean hasRegisterSpills() {
        return localSizeBytes > 0;
    }
}
