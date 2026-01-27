package io.surfworks.warpforge.core.jfr;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;

/**
 * JFR event for periodic GPU occupancy snapshots.
 *
 * <p>This event captures the overall GPU utilization state at a point in time,
 * enabling visualization of concurrent kernel execution and GPU saturation.
 * It complements {@link GpuKernelEvent} which tracks individual kernel execution.
 *
 * <p>Usage pattern - emit periodically during active GPU work:
 * <pre>{@code
 * // In a monitoring thread or periodic callback
 * GpuOccupancyEvent event = new GpuOccupancyEvent();
 * event.deviceIndex = 0;
 * event.activeStreams = streamTracker.activeStreamCount();
 * event.activeKernels = streamTracker.activeKernelCount();
 * event.estimatedTotalOccupancyPercent = streamTracker.estimatedOccupancy();
 *
 * // Query NVML/ROCm-SMI for hardware metrics
 * event.gpuUtilizationPercent = nvml.getGpuUtilization();
 * event.memoryUtilizationPercent = nvml.getMemoryUtilization();
 *
 * event.commit();
 * }</pre>
 *
 * <p>This event enables:
 * <ul>
 *   <li>Visualizing GPU utilization over time in JMC</li>
 *   <li>Detecting GPU saturation (100% utilization)</li>
 *   <li>Identifying idle periods where more work could be scheduled</li>
 *   <li>Correlating GPU activity with Java virtual thread behavior</li>
 * </ul>
 *
 * @see GpuKernelEvent
 * @see io.surfworks.warpforge.core.kernel.StreamTracker
 */
@Name("io.surfworks.warpforge.GpuOccupancy")
@Label("GPU Occupancy Snapshot")
@Category({"WarpForge", "GPU", "Occupancy"})
@Description("Periodic snapshot of GPU occupancy and utilization state")
public class GpuOccupancyEvent extends Event {

    // ==================== Device Identity ====================

    @Label("Device Index")
    @Description("GPU device index")
    public int deviceIndex;

    @Label("Device Name")
    @Description("GPU device name (e.g., 'NVIDIA RTX 4090')")
    public String deviceName;

    // ==================== Stream and Kernel Counts ====================

    @Label("Active Streams")
    @Description("Number of streams with pending or executing work")
    public int activeStreams;

    @Label("Active Kernels")
    @Description("Number of kernels currently executing or queued")
    public int activeKernels;

    @Label("Queued Kernels")
    @Description("Number of kernels waiting in stream queues")
    public int queuedKernels;

    // ==================== Occupancy Estimates ====================

    @Label("Estimated Total Occupancy %")
    @Description("Estimated GPU occupancy across all active kernels (0-100)")
    public int estimatedTotalOccupancyPercent;

    @Label("Active Warps Estimate")
    @Description("Estimated total concurrent warps across all SMs")
    public long activeWarpsEstimate;

    @Label("Estimated Active SMs")
    @Description("Estimated number of SMs with active work")
    public int estimatedActiveSMs;

    // ==================== Hardware Metrics (from NVML/ROCm-SMI) ====================

    @Label("GPU Utilization %")
    @Description("GPU compute utilization from NVML/ROCm-SMI (0-100)")
    public int gpuUtilizationPercent;

    @Label("Memory Utilization %")
    @Description("GPU memory bandwidth utilization from NVML/ROCm-SMI (0-100)")
    public int memoryUtilizationPercent;

    @Label("GPU Temperature")
    @Description("GPU temperature in Celsius")
    public int temperatureCelsius;

    @Label("Power Draw Watts")
    @Description("Current GPU power consumption in watts")
    public int powerDrawWatts;

    // ==================== Device Capabilities ====================

    @Label("SM Count")
    @Description("Number of Streaming Multiprocessors (SMs) on device")
    public int smCount;

    @Label("Max Warps Per SM")
    @Description("Maximum concurrent warps per SM")
    public int maxWarpsPerSM;

    @Label("Total Memory MB")
    @Description("Total GPU memory in megabytes")
    public int totalMemoryMB;

    @Label("Used Memory MB")
    @Description("Currently used GPU memory in megabytes")
    public int usedMemoryMB;

    // ==================== Java Context ====================

    @Label("Tracked Scopes")
    @Description("Number of active GpuTaskScopes being tracked")
    public int trackedScopes;

    @Label("Virtual Threads With GPU Work")
    @Description("Number of virtual threads with pending GPU operations")
    public int virtualThreadsWithGpuWork;

    /**
     * Compute derived metrics from raw values.
     *
     * <p>Call this after setting smCount, maxWarpsPerSM, and activeWarpsEstimate.
     */
    public void computeDerivedFields() {
        if (smCount > 0 && maxWarpsPerSM > 0) {
            long maxTotalWarps = (long) smCount * maxWarpsPerSM;
            if (maxTotalWarps > 0) {
                this.estimatedTotalOccupancyPercent =
                    (int) ((100L * activeWarpsEstimate) / maxTotalWarps);
            }
        }
    }

    /**
     * Set memory usage from byte values.
     *
     * @param totalBytes total GPU memory in bytes
     * @param usedBytes used GPU memory in bytes
     */
    public void setMemoryUsage(long totalBytes, long usedBytes) {
        this.totalMemoryMB = (int) (totalBytes / (1024 * 1024));
        this.usedMemoryMB = (int) (usedBytes / (1024 * 1024));
    }
}
