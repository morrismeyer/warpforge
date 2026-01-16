package io.surfworks.warpforge.launch.testing;

import io.surfworks.warpforge.launch.job.GpuType;
import io.surfworks.warpforge.launch.job.JobResult;
import io.surfworks.warpforge.launch.job.JobState;
import io.surfworks.warpforge.launch.job.JobStatus;
import io.surfworks.warpforge.launch.job.JobSubmission;
import io.surfworks.warpforge.launch.scheduler.ClusterInfo;
import io.surfworks.warpforge.launch.scheduler.JobQuery;
import io.surfworks.warpforge.launch.scheduler.NodeInfo;
import io.surfworks.warpforge.launch.scheduler.Scheduler;
import io.surfworks.warpforge.launch.scheduler.SchedulerCapabilities;
import io.surfworks.warpforge.launch.scheduler.SchedulerException;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Simulates a multi-node cluster for testing scheduling algorithms and queue behavior.
 *
 * <p>Features:
 * <ul>
 *   <li>Configurable number of nodes with different GPU types</li>
 *   <li>Simulates job queuing when nodes are busy</li>
 *   <li>Simulates node failures</li>
 *   <li>Tracks resource utilization</li>
 * </ul>
 */
public final class ClusterSimulator implements Scheduler {

    private final List<SimulatedNode> nodes;
    private final AtomicInteger jobCounter = new AtomicInteger(0);
    private final Map<String, SimulatedJob> jobs = new ConcurrentHashMap<>();
    private final Queue<SimulatedJob> pendingQueue = new ConcurrentLinkedQueue<>();

    private boolean connected = true;
    private Duration defaultJobDuration = Duration.ofMillis(100);

    /**
     * Creates a cluster simulator with the specified node configuration.
     *
     * @param nodeConfigs list of node configurations
     */
    public ClusterSimulator(List<NodeConfig> nodeConfigs) {
        this.nodes = new ArrayList<>();
        for (int i = 0; i < nodeConfigs.size(); i++) {
            NodeConfig config = nodeConfigs.get(i);
            nodes.add(new SimulatedNode(
                    "node-" + i,
                    config.gpuType(),
                    config.gpuCount(),
                    config.memoryMb(),
                    config.cpuCores()
            ));
        }
    }

    /**
     * Creates a simple cluster with N homogeneous nodes.
     */
    public static ClusterSimulator homogeneous(int nodeCount, GpuType gpuType, int gpusPerNode) {
        List<NodeConfig> configs = new ArrayList<>();
        for (int i = 0; i < nodeCount; i++) {
            configs.add(new NodeConfig(gpuType, gpusPerNode, 32000, 8));
        }
        return new ClusterSimulator(configs);
    }

    /**
     * Creates a cluster matching the Holmes Mark 1 lab configuration.
     */
    public static ClusterSimulator holmesMark1() {
        return new ClusterSimulator(List.of(
                new NodeConfig(GpuType.NVIDIA, 1, 16000, 4),  // NVIDIA box
                new NodeConfig(GpuType.AMD, 1, 16000, 4)      // AMD box
        ));
    }

    /**
     * Creates a cluster matching the Holmes Mark 2 lab configuration.
     */
    public static ClusterSimulator holmesMark2() {
        List<NodeConfig> configs = new ArrayList<>();
        // 5 NVIDIA nodes
        for (int i = 0; i < 5; i++) {
            configs.add(new NodeConfig(GpuType.NVIDIA, 2, 32000, 8));
        }
        // 5 AMD nodes
        for (int i = 0; i < 5; i++) {
            configs.add(new NodeConfig(GpuType.AMD, 2, 32000, 8));
        }
        return new ClusterSimulator(configs);
    }

    @Override
    public String name() {
        return "cluster-simulator";
    }

    @Override
    public SchedulerCapabilities capabilities() {
        Set<GpuType> gpuTypes = nodes.stream()
                .map(n -> n.gpuType)
                .filter(g -> g != GpuType.NONE)
                .collect(java.util.stream.Collectors.toSet());
        return SchedulerCapabilities.fullFeatured(gpuTypes);
    }

    @Override
    public synchronized String submit(JobSubmission submission) throws SchedulerException {
        String jobId = "sim-job-" + jobCounter.incrementAndGet();

        SimulatedJob job = new SimulatedJob(
                jobId,
                submission.correlationId(),
                submission,
                defaultJobDuration
        );
        jobs.put(jobId, job);

        // Try to schedule immediately
        SimulatedNode node = findAvailableNode(submission);
        if (node != null) {
            job.assignToNode(node);
            node.runningJobs.add(job);
        } else {
            pendingQueue.add(job);
        }

        return jobId;
    }

    @Override
    public JobStatus status(String jobId) throws SchedulerException {
        SimulatedJob job = jobs.get(jobId);
        if (job == null) {
            throw new SchedulerException("Job not found: " + jobId);
        }
        advanceSimulation();
        return job.currentStatus();
    }

    @Override
    public JobResult result(String jobId) throws SchedulerException {
        JobStatus status = status(jobId);

        if (!status.state().isTerminal()) {
            throw new IllegalStateException(
                    "Job " + jobId + " is not complete (state: " + status.state() + ")");
        }

        SimulatedJob job = jobs.get(jobId);
        if (status.state() == JobState.COMPLETED) {
            return JobResult.success(jobId, job.correlationId, List.of(), status.elapsed());
        } else {
            return JobResult.failure(jobId, job.correlationId, "Simulation failure", status.elapsed());
        }
    }

    @Override
    public boolean cancel(String jobId) throws SchedulerException {
        SimulatedJob job = jobs.get(jobId);
        if (job == null) {
            return false;
        }
        job.cancel();
        if (job.node != null) {
            job.node.runningJobs.remove(job);
        }
        pendingQueue.remove(job);
        return true;
    }

    @Override
    public List<JobStatus> list(JobQuery query) throws SchedulerException {
        advanceSimulation();
        List<JobStatus> result = new ArrayList<>();
        for (SimulatedJob job : jobs.values()) {
            JobStatus status = job.currentStatus();
            if (query.states().isEmpty() || query.states().contains(status.state())) {
                result.add(status);
            }
        }
        if (query.limit() > 0 && result.size() > query.limit()) {
            return result.subList(0, query.limit());
        }
        return result;
    }

    @Override
    public boolean isConnected() {
        return connected;
    }

    @Override
    public ClusterInfo clusterInfo() throws SchedulerException {
        List<NodeInfo> nodeInfos = new ArrayList<>();
        Map<GpuType, Integer> gpuCounts = new HashMap<>();
        gpuCounts.put(GpuType.NVIDIA, 0);
        gpuCounts.put(GpuType.AMD, 0);

        for (SimulatedNode node : nodes) {
            String status = node.failed ? "failed" : "ready";
            nodeInfos.add(new NodeInfo(
                    node.name, status, node.gpuType, node.gpuCount, node.memoryMb, node.cpuCores));

            if (!node.failed && node.gpuType != GpuType.NONE) {
                gpuCounts.merge(node.gpuType, node.gpuCount, Integer::sum);
            }
        }

        int availableNodes = (int) nodes.stream().filter(n -> !n.failed).count();

        return new ClusterInfo(
                "cluster-simulator",
                "1.0.0",
                nodes.size(),
                availableNodes,
                gpuCounts,
                nodeInfos
        );
    }

    @Override
    public void close() {
        // No-op
    }

    // ===== Simulation control methods =====

    /**
     * Sets the default job execution duration.
     */
    public ClusterSimulator setDefaultJobDuration(Duration duration) {
        this.defaultJobDuration = duration;
        return this;
    }

    /**
     * Simulates a node failure.
     */
    public void failNode(String nodeName) {
        nodes.stream()
                .filter(n -> n.name.equals(nodeName))
                .findFirst()
                .ifPresent(n -> n.failed = true);
    }

    /**
     * Recovers a failed node.
     */
    public void recoverNode(String nodeName) {
        nodes.stream()
                .filter(n -> n.name.equals(nodeName))
                .findFirst()
                .ifPresent(n -> n.failed = false);
    }

    /**
     * Advances simulation time, completing jobs that are done.
     */
    public void advanceSimulation() {
        // Complete finished jobs
        for (SimulatedNode node : nodes) {
            node.runningJobs.removeIf(job -> {
                if (job.shouldComplete()) {
                    job.complete();
                    return true;
                }
                return false;
            });
        }

        // Schedule pending jobs
        SimulatedJob pending;
        while ((pending = pendingQueue.peek()) != null) {
            SimulatedNode node = findAvailableNode(pending.submission);
            if (node == null) {
                break;
            }
            pendingQueue.poll();
            pending.assignToNode(node);
            node.runningJobs.add(pending);
        }
    }

    /**
     * Returns the number of pending jobs in the queue.
     */
    public int getPendingQueueSize() {
        return pendingQueue.size();
    }

    /**
     * Returns node utilization statistics.
     */
    public Map<String, NodeStats> getNodeStats() {
        Map<String, NodeStats> stats = new HashMap<>();
        for (SimulatedNode node : nodes) {
            stats.put(node.name, new NodeStats(
                    node.runningJobs.size(),
                    node.totalJobsRun,
                    node.failed
            ));
        }
        return stats;
    }

    private SimulatedNode findAvailableNode(JobSubmission submission) {
        var resources = submission.definition().resources();

        for (SimulatedNode node : nodes) {
            if (node.failed) continue;
            if (!node.runningJobs.isEmpty()) continue; // Simple: one job per node

            // Check GPU compatibility
            if (resources.requiresGpu()) {
                if (resources.gpuType() != GpuType.ANY &&
                        resources.gpuType() != node.gpuType) {
                    continue;
                }
                if (resources.gpuCount() > node.gpuCount) {
                    continue;
                }
            }

            return node;
        }
        return null;
    }

    /**
     * Node configuration for cluster creation.
     */
    public record NodeConfig(GpuType gpuType, int gpuCount, long memoryMb, int cpuCores) {}

    /**
     * Statistics for a simulated node.
     */
    public record NodeStats(int runningJobs, int totalJobsRun, boolean failed) {}

    private static class SimulatedNode {
        final String name;
        final GpuType gpuType;
        final int gpuCount;
        final long memoryMb;
        final int cpuCores;
        final List<SimulatedJob> runningJobs = new ArrayList<>();

        boolean failed = false;
        int totalJobsRun = 0;

        SimulatedNode(String name, GpuType gpuType, int gpuCount, long memoryMb, int cpuCores) {
            this.name = name;
            this.gpuType = gpuType;
            this.gpuCount = gpuCount;
            this.memoryMb = memoryMb;
            this.cpuCores = cpuCores;
        }
    }

    private static class SimulatedJob {
        final String jobId;
        final String correlationId;
        final JobSubmission submission;
        final Duration duration;
        final Instant submitTime;

        SimulatedNode node;
        JobState state = JobState.PENDING;
        Instant startTime;
        Instant completionTime;

        SimulatedJob(String jobId, String correlationId, JobSubmission submission, Duration duration) {
            this.jobId = jobId;
            this.correlationId = correlationId;
            this.submission = submission;
            this.duration = duration;
            this.submitTime = Instant.now();
        }

        void assignToNode(SimulatedNode node) {
            this.node = node;
            this.state = JobState.RUNNING;
            this.startTime = Instant.now();
        }

        boolean shouldComplete() {
            if (state != JobState.RUNNING) return false;
            return Duration.between(startTime, Instant.now()).compareTo(duration) >= 0;
        }

        void complete() {
            this.state = JobState.COMPLETED;
            this.completionTime = Instant.now();
            if (node != null) {
                node.totalJobsRun++;
            }
        }

        void cancel() {
            if (!state.isTerminal()) {
                this.state = JobState.CANCELLED;
                this.completionTime = Instant.now();
            }
        }

        JobStatus currentStatus() {
            Duration elapsed = completionTime != null ?
                    Duration.between(submitTime, completionTime) :
                    Duration.between(submitTime, Instant.now());

            return new JobStatus(
                    jobId,
                    correlationId,
                    state,
                    submitTime,
                    node != null ? node.name : null,
                    elapsed,
                    state.name(),
                    Map.of()
            );
        }
    }
}
