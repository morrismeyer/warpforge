package io.surfworks.warpforge.core.kernel;

import io.surfworks.warpforge.core.jfr.GpuKernelEvent;

/**
 * Calculates theoretical GPU occupancy based on kernel resource usage.
 *
 * <p>Occupancy is the ratio of active warps to the maximum possible warps per SM.
 * Higher occupancy generally means better latency hiding, though it's not always
 * correlated with performance (memory-bound kernels may not benefit).
 *
 * <p>The calculation considers four limiting factors:
 * <ol>
 *   <li><b>Registers:</b> Registers per SM / (registers per thread × warp size)</li>
 *   <li><b>Shared memory:</b> Shared memory per SM / shared memory per block</li>
 *   <li><b>Block count:</b> Maximum blocks per SM (hardware limit)</li>
 *   <li><b>Warp count:</b> Maximum warps per SM (hardware limit)</li>
 * </ol>
 *
 * <p>The tightest constraint determines achievable occupancy.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Create calculator for RTX 4090
 * OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
 *     128,    // SM count
 *     64,     // max warps per SM
 *     65536,  // registers per SM
 *     102400, // shared memory per SM
 *     32      // max blocks per SM
 * );
 *
 * // Calculate occupancy for a kernel
 * KernelAttributes attrs = KernelAttributes.of(32, 8192);
 * OccupancyInfo info = calc.calculate(attrs, 256, 0);
 *
 * System.out.println("Occupancy: " + info.occupancyPercent() + "%");
 * System.out.println("Limited by: " + info.limitingFactor());
 * }</pre>
 *
 * @see KernelAttributes
 * @see OccupancyInfo
 */
public final class OccupancyCalculator {

    private final int smCount;
    private final int maxWarpsPerSM;
    private final int maxRegistersPerSM;
    private final int maxSharedMemoryPerSM;
    private final int maxBlocksPerSM;
    private final int warpSize;

    /**
     * Creates an occupancy calculator with the given device properties.
     *
     * @param smCount number of SMs (Streaming Multiprocessors) on the device
     * @param maxWarpsPerSM maximum concurrent warps per SM
     * @param maxRegistersPerSM total registers available per SM
     * @param maxSharedMemoryPerSM total shared memory per SM in bytes
     * @param maxBlocksPerSM maximum concurrent blocks per SM
     * @param warpSize threads per warp (32 for NVIDIA, 32 or 64 for AMD)
     */
    public OccupancyCalculator(
            int smCount,
            int maxWarpsPerSM,
            int maxRegistersPerSM,
            int maxSharedMemoryPerSM,
            int maxBlocksPerSM,
            int warpSize) {
        this.smCount = smCount;
        this.maxWarpsPerSM = maxWarpsPerSM;
        this.maxRegistersPerSM = maxRegistersPerSM;
        this.maxSharedMemoryPerSM = maxSharedMemoryPerSM;
        this.maxBlocksPerSM = maxBlocksPerSM;
        this.warpSize = warpSize;
    }

    /**
     * Creates an occupancy calculator for a typical NVIDIA device.
     *
     * @param smCount number of SMs
     * @param maxWarpsPerSM maximum warps per SM (typically 48-64)
     * @param maxRegistersPerSM registers per SM (typically 65536)
     * @param maxSharedMemoryPerSM shared memory per SM in bytes
     * @param maxBlocksPerSM maximum blocks per SM (typically 16-32)
     * @return calculator configured for NVIDIA
     */
    public static OccupancyCalculator forNvidiaDevice(
            int smCount,
            int maxWarpsPerSM,
            int maxRegistersPerSM,
            int maxSharedMemoryPerSM,
            int maxBlocksPerSM) {
        return new OccupancyCalculator(
            smCount, maxWarpsPerSM, maxRegistersPerSM,
            maxSharedMemoryPerSM, maxBlocksPerSM, 32);
    }

    /**
     * Creates an occupancy calculator for an AMD CDNA device (64-wide wavefronts).
     *
     * @param cuCount number of Compute Units
     * @param maxWavefrontsPerCU maximum wavefronts per CU
     * @param maxVGPRsPerCU VGPRs per CU
     * @param maxLDSPerCU LDS (shared memory) per CU in bytes
     * @param maxWorkgroupsPerCU maximum workgroups per CU
     * @return calculator configured for AMD CDNA
     */
    public static OccupancyCalculator forAmdCdnaDevice(
            int cuCount,
            int maxWavefrontsPerCU,
            int maxVGPRsPerCU,
            int maxLDSPerCU,
            int maxWorkgroupsPerCU) {
        return new OccupancyCalculator(
            cuCount, maxWavefrontsPerCU, maxVGPRsPerCU,
            maxLDSPerCU, maxWorkgroupsPerCU, 64);
    }

    /**
     * Creates an occupancy calculator for an AMD RDNA device (32-wide wavefronts).
     *
     * @param cuCount number of Compute Units
     * @param maxWavefrontsPerCU maximum wavefronts per CU
     * @param maxVGPRsPerCU VGPRs per CU
     * @param maxLDSPerCU LDS (shared memory) per CU in bytes
     * @param maxWorkgroupsPerCU maximum workgroups per CU
     * @return calculator configured for AMD RDNA
     */
    public static OccupancyCalculator forAmdRdnaDevice(
            int cuCount,
            int maxWavefrontsPerCU,
            int maxVGPRsPerCU,
            int maxLDSPerCU,
            int maxWorkgroupsPerCU) {
        return new OccupancyCalculator(
            cuCount, maxWavefrontsPerCU, maxVGPRsPerCU,
            maxLDSPerCU, maxWorkgroupsPerCU, 32);
    }

    /**
     * Calculates occupancy for a kernel with the given attributes and launch config.
     *
     * @param attrs kernel resource attributes
     * @param blockSize threads per block
     * @param dynamicSharedMem dynamic shared memory per block in bytes
     * @return occupancy information including limiting factor
     */
    public OccupancyInfo calculate(KernelAttributes attrs, int blockSize, int dynamicSharedMem) {
        int warpsPerBlock = (blockSize + warpSize - 1) / warpSize;
        int totalSharedMem = (int) attrs.sharedSizeBytes() + dynamicSharedMem;

        // Constraint 1: Register limit
        int regsPerWarp = attrs.numRegs() * warpSize;
        int warpsLimitedByRegs = regsPerWarp > 0
            ? maxRegistersPerSM / regsPerWarp
            : maxWarpsPerSM;

        // Constraint 2: Shared memory limit (only applies if kernel uses shared memory)
        int blocksLimitedByShared = totalSharedMem > 0
            ? maxSharedMemoryPerSM / totalSharedMem
            : Integer.MAX_VALUE; // Not limiting when no shared memory
        int warpsLimitedByShared = totalSharedMem > 0
            ? blocksLimitedByShared * warpsPerBlock
            : Integer.MAX_VALUE;

        // Constraint 3: Block count limit
        int warpsLimitedByBlocks = maxBlocksPerSM * warpsPerBlock;

        // Constraint 4: Warp limit
        int warpsLimitedByMax = maxWarpsPerSM;

        // Find the tightest constraint
        int activeWarpsPerSM = Math.min(
            Math.min(warpsLimitedByRegs, warpsLimitedByShared),
            Math.min(warpsLimitedByBlocks, warpsLimitedByMax)
        );

        // Determine limiting factor
        String limitingFactor = determineLimitingFactor(
            activeWarpsPerSM,
            warpsLimitedByRegs,
            warpsLimitedByShared,
            warpsLimitedByBlocks,
            warpsLimitedByMax
        );

        // Calculate occupancy percentage
        int occupancyPercent = (int) ((100L * activeWarpsPerSM) / maxWarpsPerSM);

        // Calculate max active blocks per SM
        int maxActiveBlocksPerSM = warpsPerBlock > 0
            ? activeWarpsPerSM / warpsPerBlock
            : 0;

        return new OccupancyInfo(
            occupancyPercent,
            activeWarpsPerSM,
            maxActiveBlocksPerSM,
            limitingFactor
        );
    }

    /**
     * Populates a GpuKernelEvent with occupancy information.
     *
     * @param event the JFR event to populate
     * @param attrs kernel attributes
     * @param launchConfig launch configuration
     */
    public void populateEvent(GpuKernelEvent event, KernelAttributes attrs, LaunchConfig launchConfig) {
        // Resource usage
        event.registersPerThread = attrs.numRegs();
        event.staticSharedMemoryBytes = (int) attrs.sharedSizeBytes();
        event.dynamicSharedMemoryBytes = launchConfig.sharedMemBytes();
        event.localMemoryPerThread = (int) attrs.localSizeBytes();

        // Calculate occupancy
        OccupancyInfo info = calculate(attrs, launchConfig.threadsPerBlock(), launchConfig.sharedMemBytes());

        // Occupancy metrics
        event.theoreticalOccupancyPercent = info.occupancyPercent();
        event.maxActiveBlocksPerSM = info.maxActiveBlocksPerSM();
        event.occupancyLimitingFactor = info.limitingFactor();

        // Estimate active SMs and warps
        event.estimatedActiveSMs = Math.min(launchConfig.totalBlocks(), smCount);
        event.estimatedActiveWarps = (long) event.estimatedActiveSMs * info.activeWarpsPerSM();
    }

    private String determineLimitingFactor(
            int activeWarps,
            int byRegs,
            int byShared,
            int byBlocks,
            int byMax) {
        if (activeWarps == byRegs) return "registers";
        if (activeWarps == byShared) return "shared_memory";
        if (activeWarps == byBlocks) return "blocks";
        if (activeWarps == byMax) return "warps";
        return "unknown";
    }

    /**
     * Returns the number of SMs/CUs on the device.
     */
    public int smCount() {
        return smCount;
    }

    /**
     * Returns the maximum warps/wavefronts per SM/CU.
     */
    public int maxWarpsPerSM() {
        return maxWarpsPerSM;
    }

    /**
     * Returns the warp/wavefront size.
     */
    public int warpSize() {
        return warpSize;
    }

    /**
     * Occupancy calculation result.
     *
     * @param occupancyPercent theoretical occupancy as a percentage (0-100)
     * @param activeWarpsPerSM number of warps that can be active per SM
     * @param maxActiveBlocksPerSM maximum blocks that can be active per SM
     * @param limitingFactor what limits occupancy: "registers", "shared_memory", "blocks", or "warps"
     */
    public record OccupancyInfo(
        int occupancyPercent,
        int activeWarpsPerSM,
        int maxActiveBlocksPerSM,
        String limitingFactor
    ) {}
}
