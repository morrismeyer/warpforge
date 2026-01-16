package io.surfworks.warpforge.launch.job;

/**
 * GPU hardware types supported by WarpForge.
 */
public enum GpuType {
    /** No GPU required - CPU-only execution */
    NONE,
    /** NVIDIA GPU (CUDA) */
    NVIDIA,
    /** AMD GPU (ROCm) */
    AMD,
    /** Any available GPU */
    ANY
}
