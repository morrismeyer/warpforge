package io.surfworks.warpforge.nvml;

/**
 * Mock NVML implementation for environments without NVIDIA GPUs
 * (e.g., macOS dev machines).
 */
public final class NvmlMock implements NvmlApi {

    @Override
    public String backendName() {
        return "mock";
    }

    @Override
    public String driverVersion() {
        return "mock-0.0";
    }

    @Override
    public int deviceCount() {
        return 0;
    }

    @Override
    public String toString() {
        return "NvmlMock{driverVersion=" + driverVersion()
               + ", deviceCount=" + deviceCount() + "}";
    }
}

