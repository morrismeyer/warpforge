package io.surfworks.warpforge.core.kernel;

/**
 * Encapsulates GPU kernel launch configuration.
 *
 * <p>This record holds the grid and block dimensions for a kernel launch,
 * along with shared memory requirements. It is used to populate JFR events
 * and calculate derived metrics like total threads and warps.
 *
 * <p>NVIDIA terminology:
 * <ul>
 *   <li>Grid = collection of blocks</li>
 *   <li>Block = collection of threads</li>
 *   <li>Warp = 32 threads executing in lockstep</li>
 * </ul>
 *
 * <p>AMD terminology (equivalent):
 * <ul>
 *   <li>Grid = collection of workgroups</li>
 *   <li>Workgroup = collection of work-items</li>
 *   <li>Wavefront = 32 or 64 work-items executing in lockstep</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * // GEMM kernel with 256x256 blocks of 256 threads each
 * LaunchConfig config = new LaunchConfig(256, 256, 1, 256, 1, 1, 0);
 *
 * // Populate JFR event
 * config.populateEvent(event, 32); // warpSize=32 for NVIDIA
 * }</pre>
 *
 * @param gridDimX number of blocks in X dimension
 * @param gridDimY number of blocks in Y dimension
 * @param gridDimZ number of blocks in Z dimension
 * @param blockDimX threads per block in X dimension
 * @param blockDimY threads per block in Y dimension
 * @param blockDimZ threads per block in Z dimension
 * @param sharedMemBytes dynamic shared memory per block (bytes)
 */
public record LaunchConfig(
    int gridDimX,
    int gridDimY,
    int gridDimZ,
    int blockDimX,
    int blockDimY,
    int blockDimZ,
    int sharedMemBytes
) {
    /**
     * Warp size for NVIDIA GPUs.
     */
    public static final int NVIDIA_WARP_SIZE = 32;

    /**
     * Wavefront size for AMD RDNA GPUs.
     */
    public static final int AMD_RDNA_WAVEFRONT_SIZE = 32;

    /**
     * Wavefront size for AMD CDNA/GCN GPUs.
     */
    public static final int AMD_CDNA_WAVEFRONT_SIZE = 64;

    /**
     * Creates a 1D launch configuration.
     *
     * @param gridSize total number of blocks
     * @param blockSize threads per block
     * @return a 1D launch configuration
     */
    public static LaunchConfig of1D(int gridSize, int blockSize) {
        return new LaunchConfig(gridSize, 1, 1, blockSize, 1, 1, 0);
    }

    /**
     * Creates a 2D launch configuration.
     *
     * @param gridDimX blocks in X
     * @param gridDimY blocks in Y
     * @param blockDimX threads per block in X
     * @param blockDimY threads per block in Y
     * @return a 2D launch configuration
     */
    public static LaunchConfig of2D(int gridDimX, int gridDimY, int blockDimX, int blockDimY) {
        return new LaunchConfig(gridDimX, gridDimY, 1, blockDimX, blockDimY, 1, 0);
    }

    /**
     * Creates a launch configuration with dynamic shared memory.
     *
     * @param gridDimX blocks in X
     * @param gridDimY blocks in Y
     * @param gridDimZ blocks in Z
     * @param blockDimX threads per block in X
     * @param blockDimY threads per block in Y
     * @param blockDimZ threads per block in Z
     * @param sharedMemBytes dynamic shared memory per block
     * @return a launch configuration with shared memory
     */
    public static LaunchConfig withSharedMem(
            int gridDimX, int gridDimY, int gridDimZ,
            int blockDimX, int blockDimY, int blockDimZ,
            int sharedMemBytes) {
        return new LaunchConfig(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes);
    }

    /**
     * Returns the total number of blocks in the grid.
     */
    public int totalBlocks() {
        return gridDimX * gridDimY * gridDimZ;
    }

    /**
     * Returns the total threads per block.
     */
    public int threadsPerBlock() {
        return blockDimX * blockDimY * blockDimZ;
    }

    /**
     * Returns the total number of threads across all blocks.
     */
    public long totalThreads() {
        return (long) totalBlocks() * threadsPerBlock();
    }

    /**
     * Returns the total number of warps/wavefronts.
     *
     * @param warpSize the warp/wavefront size (32 for NVIDIA, 32 or 64 for AMD)
     */
    public long totalWarps(int warpSize) {
        return (totalThreads() + warpSize - 1) / warpSize;
    }

    /**
     * Populates a GpuKernelEvent with launch configuration fields.
     *
     * @param event the JFR event to populate
     * @param warpSize the warp/wavefront size
     */
    public void populateEvent(io.surfworks.warpforge.core.jfr.GpuKernelEvent event, int warpSize) {
        event.gridDimX = this.gridDimX;
        event.gridDimY = this.gridDimY;
        event.gridDimZ = this.gridDimZ;
        event.blockDimX = this.blockDimX;
        event.blockDimY = this.blockDimY;
        event.blockDimZ = this.blockDimZ;
        event.computeDerivedFields(warpSize);
    }

    @Override
    public String toString() {
        if (gridDimY == 1 && gridDimZ == 1 && blockDimY == 1 && blockDimZ == 1) {
            // 1D configuration
            return String.format("LaunchConfig[grid=%d, block=%d, threads=%d]",
                gridDimX, blockDimX, totalThreads());
        } else if (gridDimZ == 1 && blockDimZ == 1) {
            // 2D configuration
            return String.format("LaunchConfig[grid=%dx%d, block=%dx%d, threads=%d]",
                gridDimX, gridDimY, blockDimX, blockDimY, totalThreads());
        } else {
            // 3D configuration
            return String.format("LaunchConfig[grid=%dx%dx%d, block=%dx%dx%d, threads=%d]",
                gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, totalThreads());
        }
    }
}
