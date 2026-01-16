package io.surfworks.warpforge.launch.job;

import java.util.Objects;
import java.util.Set;

/**
 * Resource requirements for job execution.
 *
 * @param gpuType      Type of GPU required (NONE for CPU-only)
 * @param gpuCount     Number of GPUs required
 * @param memoryMb     Memory required in megabytes
 * @param cpuCores     Number of CPU cores required
 * @param queue        Optional queue/partition name
 * @param priority     Job priority (0-100, higher = more important)
 * @param nodeAffinity Optional set of specific node names to run on
 */
public record ResourceRequirements(
        GpuType gpuType,
        int gpuCount,
        long memoryMb,
        int cpuCores,
        String queue,
        int priority,
        Set<String> nodeAffinity
) {

    public ResourceRequirements {
        Objects.requireNonNull(gpuType, "gpuType cannot be null");
        if (gpuCount < 0) {
            throw new IllegalArgumentException("gpuCount must be non-negative");
        }
        if (gpuType == GpuType.NONE && gpuCount > 0) {
            throw new IllegalArgumentException("gpuCount must be 0 when gpuType is NONE");
        }
        if (gpuType != GpuType.NONE && gpuCount == 0) {
            throw new IllegalArgumentException("gpuCount must be positive when GPU is requested");
        }
        if (memoryMb <= 0) {
            throw new IllegalArgumentException("memoryMb must be positive");
        }
        if (cpuCores <= 0) {
            throw new IllegalArgumentException("cpuCores must be positive");
        }
        if (priority < 0 || priority > 100) {
            throw new IllegalArgumentException("priority must be between 0 and 100");
        }
        nodeAffinity = nodeAffinity == null ? Set.of() : Set.copyOf(nodeAffinity);
    }

    /**
     * Creates CPU-only resource requirements.
     */
    public static ResourceRequirements cpuOnly(int cpuCores, long memoryMb) {
        return new ResourceRequirements(GpuType.NONE, 0, memoryMb, cpuCores, null, 50, Set.of());
    }

    /**
     * Creates resource requirements for NVIDIA GPU.
     */
    public static ResourceRequirements nvidia(int gpuCount) {
        return nvidia(gpuCount, 4096, 4);
    }

    /**
     * Creates resource requirements for NVIDIA GPU with memory and CPU.
     */
    public static ResourceRequirements nvidia(int gpuCount, long memoryMb, int cpuCores) {
        return new ResourceRequirements(GpuType.NVIDIA, gpuCount, memoryMb, cpuCores, null, 50, Set.of());
    }

    /**
     * Creates resource requirements for AMD GPU.
     */
    public static ResourceRequirements amd(int gpuCount) {
        return amd(gpuCount, 4096, 4);
    }

    /**
     * Creates resource requirements for AMD GPU with memory and CPU.
     */
    public static ResourceRequirements amd(int gpuCount, long memoryMb, int cpuCores) {
        return new ResourceRequirements(GpuType.AMD, gpuCount, memoryMb, cpuCores, null, 50, Set.of());
    }

    /**
     * Creates resource requirements for any available GPU.
     */
    public static ResourceRequirements anyGpu(int gpuCount) {
        return anyGpu(gpuCount, 4096, 4);
    }

    /**
     * Creates resource requirements for any available GPU with memory and CPU.
     */
    public static ResourceRequirements anyGpu(int gpuCount, long memoryMb, int cpuCores) {
        return new ResourceRequirements(GpuType.ANY, gpuCount, memoryMb, cpuCores, null, 50, Set.of());
    }

    /**
     * Returns a new instance with the specified queue.
     */
    public ResourceRequirements withQueue(String queue) {
        return new ResourceRequirements(gpuType, gpuCount, memoryMb, cpuCores, queue, priority, nodeAffinity);
    }

    /**
     * Returns a new instance with the specified priority.
     */
    public ResourceRequirements withPriority(int priority) {
        return new ResourceRequirements(gpuType, gpuCount, memoryMb, cpuCores, queue, priority, nodeAffinity);
    }

    /**
     * Returns a new instance with the specified node affinity.
     */
    public ResourceRequirements withNodeAffinity(Set<String> nodes) {
        return new ResourceRequirements(gpuType, gpuCount, memoryMb, cpuCores, queue, priority, nodes);
    }

    /**
     * Returns true if this request requires a GPU.
     */
    public boolean requiresGpu() {
        return gpuType != GpuType.NONE;
    }
}
