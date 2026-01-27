package io.surfworks.warpforge.core.backend;

/**
 * Interface for GPU monitoring capabilities.
 *
 * <p>Provides access to real-time GPU metrics for scheduling decisions:
 * <ul>
 *   <li>Utilization rates (GPU and memory bandwidth)</li>
 *   <li>Memory usage</li>
 *   <li>Temperature and power (when available)</li>
 * </ul>
 *
 * <p><b>Important caveat:</b> GPU utilization from NVML/SMI measures the percentage
 * of time over the past sample period during which one or more kernels was executing.
 * It does NOT measure what percentage of the GPU's compute capacity (SMs/CUs) is being used.
 * A kernel using 10% of SMs still shows 100% utilization while running.
 *
 * <p>For Orion-style occupancy-based admission control, use utilization as a proxy
 * for GPU busyness, combined with active stream count and memory pressure.
 *
 * @see <a href="https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html">NVML API</a>
 * @see <a href="https://rocm.docs.amd.com/projects/amdsmi/en/docs-6.1.0/">AMD SMI</a>
 */
public interface GpuMonitoring {

    /**
     * Check if GPU monitoring is available.
     *
     * @return true if monitoring metrics can be queried
     */
    boolean isMonitoringAvailable();

    /**
     * Get GPU utilization rate.
     *
     * <p>This measures the percentage of time over the past sample period
     * (typically 166ms to 1 second) during which one or more kernels was executing.
     *
     * @return GPU utilization percentage (0-100), or -1 if not available
     */
    int getGpuUtilization();

    /**
     * Get memory bandwidth utilization rate.
     *
     * <p>This measures the percentage of time over the past sample period
     * during which GPU memory was being read or written.
     *
     * @return Memory utilization percentage (0-100), or -1 if not available
     */
    int getMemoryUtilization();

    /**
     * Get GPU temperature.
     *
     * @return Temperature in degrees Celsius, or -1 if not available
     */
    default int getTemperature() {
        return -1;
    }

    /**
     * Get GPU power usage.
     *
     * @return Power usage in milliwatts, or -1 if not available
     */
    default int getPowerUsage() {
        return -1;
    }

    /**
     * Snapshot of GPU metrics at a point in time.
     *
     * @param gpuUtilization GPU utilization percentage (0-100)
     * @param memoryUtilization Memory bandwidth utilization percentage (0-100)
     * @param memoryUsedBytes GPU memory currently in use
     * @param memoryTotalBytes Total GPU memory available
     * @param timestampNanos System.nanoTime() when metrics were captured
     */
    record GpuMetrics(
        int gpuUtilization,
        int memoryUtilization,
        long memoryUsedBytes,
        long memoryTotalBytes,
        long timestampNanos
    ) {
        /**
         * Memory usage as a percentage.
         */
        public double memoryUsedPercent() {
            return memoryTotalBytes > 0 ? (memoryUsedBytes * 100.0 / memoryTotalBytes) : 0;
        }

        /**
         * Check if GPU is considered busy (high utilization).
         *
         * @param threshold Utilization threshold (e.g., 80 for 80%)
         * @return true if GPU utilization exceeds threshold
         */
        public boolean isBusy(int threshold) {
            return gpuUtilization >= threshold;
        }

        /**
         * Check if memory is under pressure (high usage).
         *
         * @param threshold Memory usage threshold percentage (e.g., 90 for 90%)
         * @return true if memory usage exceeds threshold
         */
        public boolean isMemoryPressure(int threshold) {
            return memoryUsedPercent() >= threshold;
        }

        @Override
        public String toString() {
            return String.format("GpuMetrics[gpu=%d%%, mem=%d%%, used=%.1fGB/%.1fGB (%.1f%%)]",
                gpuUtilization, memoryUtilization,
                memoryUsedBytes / 1e9, memoryTotalBytes / 1e9, memoryUsedPercent());
        }
    }

    /**
     * Get a snapshot of all GPU metrics.
     *
     * @return Current GPU metrics, or null if monitoring is not available
     */
    default GpuMetrics getMetrics() {
        if (!isMonitoringAvailable()) {
            return null;
        }
        return new GpuMetrics(
            getGpuUtilization(),
            getMemoryUtilization(),
            usedDeviceMemory(),
            totalDeviceMemory(),
            System.nanoTime()
        );
    }

    /**
     * Get used device memory in bytes (required for metrics).
     */
    long usedDeviceMemory();

    /**
     * Get total device memory in bytes (required for metrics).
     */
    long totalDeviceMemory();
}
