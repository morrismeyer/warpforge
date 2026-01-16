package io.surfworks.warpforge.launch.scheduler;

import io.surfworks.warpforge.launch.job.GpuType;

import java.util.Objects;

/**
 * Information about a compute node in a cluster.
 *
 * @param name      Node hostname or identifier
 * @param status    Current status (e.g., "ready", "busy", "offline")
 * @param gpuType   Type of GPU on this node
 * @param gpuCount  Number of GPUs on this node
 * @param memoryMb  Total memory in megabytes
 * @param cpuCores  Number of CPU cores
 */
public record NodeInfo(
        String name,
        String status,
        GpuType gpuType,
        int gpuCount,
        long memoryMb,
        int cpuCores
) {

    public NodeInfo {
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(status, "status cannot be null");
        Objects.requireNonNull(gpuType, "gpuType cannot be null");
    }

    /**
     * Creates a CPU-only node info.
     */
    public static NodeInfo cpuNode(String name, String status, long memoryMb, int cpuCores) {
        return new NodeInfo(name, status, GpuType.NONE, 0, memoryMb, cpuCores);
    }

    /**
     * Creates an NVIDIA GPU node info.
     */
    public static NodeInfo nvidiaNode(String name, String status, int gpuCount, long memoryMb, int cpuCores) {
        return new NodeInfo(name, status, GpuType.NVIDIA, gpuCount, memoryMb, cpuCores);
    }

    /**
     * Creates an AMD GPU node info.
     */
    public static NodeInfo amdNode(String name, String status, int gpuCount, long memoryMb, int cpuCores) {
        return new NodeInfo(name, status, GpuType.AMD, gpuCount, memoryMb, cpuCores);
    }

    /**
     * Returns true if this node has GPUs.
     */
    public boolean hasGpu() {
        return gpuType != GpuType.NONE && gpuCount > 0;
    }

    /**
     * Returns true if this node is ready to accept jobs.
     */
    public boolean isReady() {
        return "ready".equalsIgnoreCase(status);
    }
}
