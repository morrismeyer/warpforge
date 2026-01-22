package io.surfworks.warpforge.benchmark;

/**
 * Execution tier for GPU kernels, balancing performance against observability.
 *
 * <p>WarpForge provides three execution tiers:
 * <ul>
 *   <li>{@link #PRODUCTION} - Maximum performance via vendor libraries (cuBLAS/rocBLAS)</li>
 *   <li>{@link #OPTIMIZED_OBSERVABLE} - Near-production performance (~93%) with salt instrumentation</li>
 *   <li>{@link #CORRECTNESS} - Full observability for debugging (~1% performance)</li>
 * </ul>
 *
 * <p>The OPTIMIZED_OBSERVABLE tier is inspired by Simon Boehm's work showing that
 * optimized CUDA with coalescing, shared memory, register blocking, and warptiling
 * can achieve 93.7% of cuBLAS performance. This allows salt instrumentation while
 * maintaining near-production speed.
 *
 * @see <a href="https://siboehm.com/articles/22/CUDA-MMM">Simon Boehm's CUDA GEMM</a>
 */
public enum KernelTier {

    /**
     * Maximum performance via vendor libraries (cuBLAS, rocBLAS, cuDNN, MIOpen).
     * External timing only via CUDA/HIP Events. No kernel-internal visibility.
     */
    PRODUCTION(0, "vendor", 1.00),

    /**
     * Near-production performance with salt instrumentation for kernel internals.
     * Uses optimized PTX/HIP with timing instrumentation (~8-12 cycles overhead per thread).
     * Achieves approximately 93% of PRODUCTION performance.
     */
    OPTIMIZED_OBSERVABLE(1, "optimized", 0.93),

    /**
     * Full observability at any cost. For debugging numerical issues.
     * Uses naive PTX/HIP with full tracing and step-by-step verification.
     * Achieves approximately 1% of PRODUCTION performance.
     */
    CORRECTNESS(2, "naive", 0.01);

    private final int saltLevel;
    private final String backendSuffix;
    private final double expectedPerformanceRatio;

    KernelTier(int saltLevel, String backendSuffix, double expectedPerformanceRatio) {
        this.saltLevel = saltLevel;
        this.backendSuffix = backendSuffix;
        this.expectedPerformanceRatio = expectedPerformanceRatio;
    }

    /**
     * Returns the salt level for PTX instrumentation.
     * <ul>
     *   <li>0 = SALT_NONE - No instrumentation</li>
     *   <li>1 = SALT_TIMING - Cycle counters around compute sections</li>
     *   <li>2 = SALT_TRACE - Memory access patterns, warp divergence</li>
     * </ul>
     */
    public int saltLevel() {
        return saltLevel;
    }

    /**
     * Returns the backend suffix for identifying kernel implementations.
     */
    public String backendSuffix() {
        return backendSuffix;
    }

    /**
     * Returns the expected performance ratio relative to PRODUCTION tier.
     * Used for validating benchmark results fall within expected bounds.
     */
    public double expectedPerformanceRatio() {
        return expectedPerformanceRatio;
    }

    /**
     * Calculates the expected overhead percentage relative to PRODUCTION.
     * For OPTIMIZED_OBSERVABLE, this is approximately 7% (1 - 0.93).
     */
    public double expectedOverheadPercent() {
        return (1.0 - expectedPerformanceRatio) * 100.0;
    }
}
