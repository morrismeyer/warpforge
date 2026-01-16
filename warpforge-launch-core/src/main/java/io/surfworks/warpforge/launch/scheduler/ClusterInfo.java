package io.surfworks.warpforge.launch.scheduler;

import io.surfworks.warpforge.launch.job.GpuType;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Information about a compute cluster.
 *
 * @param schedulerName    Name of the scheduler (e.g., "ray", "kubernetes", "slurm")
 * @param schedulerVersion Version of the scheduler
 * @param totalNodes       Total number of nodes in the cluster
 * @param availableNodes   Number of nodes currently available for jobs
 * @param gpuCounts        Count of GPUs by type across the cluster
 * @param nodes            Detailed information about each node
 */
public record ClusterInfo(
        String schedulerName,
        String schedulerVersion,
        int totalNodes,
        int availableNodes,
        Map<GpuType, Integer> gpuCounts,
        List<NodeInfo> nodes
) {

    public ClusterInfo {
        Objects.requireNonNull(schedulerName, "schedulerName cannot be null");
        Objects.requireNonNull(schedulerVersion, "schedulerVersion cannot be null");
        Objects.requireNonNull(gpuCounts, "gpuCounts cannot be null");
        Objects.requireNonNull(nodes, "nodes cannot be null");

        gpuCounts = Map.copyOf(gpuCounts);
        nodes = List.copyOf(nodes);
    }

    /**
     * Creates cluster info for a single local node.
     */
    public static ClusterInfo local() {
        var localNode = NodeInfo.cpuNode("localhost", "ready",
                Runtime.getRuntime().maxMemory() / (1024 * 1024),
                Runtime.getRuntime().availableProcessors());
        return new ClusterInfo(
                "local", "1.0.0",
                1, 1,
                Map.of(GpuType.NONE, 0),
                List.of(localNode)
        );
    }

    /**
     * Returns total GPU count across all types.
     */
    public int totalGpuCount() {
        return gpuCounts.values().stream().mapToInt(Integer::intValue).sum();
    }

    /**
     * Returns GPU count for a specific type.
     */
    public int gpuCount(GpuType type) {
        return gpuCounts.getOrDefault(type, 0);
    }

    /**
     * Returns ready nodes only.
     */
    public List<NodeInfo> readyNodes() {
        return nodes.stream().filter(NodeInfo::isReady).toList();
    }

    /**
     * Returns nodes with GPUs.
     */
    public List<NodeInfo> gpuNodes() {
        return nodes.stream().filter(NodeInfo::hasGpu).toList();
    }
}
