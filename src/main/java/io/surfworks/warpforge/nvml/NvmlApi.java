package io.surfworks.warpforge.nvml;

/**
 * Minimal NVML abstraction for WarpForge.
 * Implementations:
 *  - NvmlMock: no GPU, no native calls (Mac, dev)
 *  - NvmlFFMImpl: real NVML via Java FFM (Linux + NVIDIA)
 */
public interface NvmlApi {

    /**
     * Human-readable backend identifier, e.g. "mock" or "nvml-ffm".
     */
    String backendName();

    /**
     * NVML driver version string.
     * For the mock backend this can be a placeholder like "mock-0.0".
     */
    String driverVersion();

    /**
     * Number of GPUs visible to NVML.
     * For the mock backend this is typically 0.
     */
    int deviceCount();
}

