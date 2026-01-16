package io.surfworks.warpforge.launch.testing;

import io.surfworks.warpforge.launch.job.GpuType;
import io.surfworks.warpforge.launch.job.InputSpec;
import io.surfworks.warpforge.launch.job.JobDefinition;
import io.surfworks.warpforge.launch.job.JobState;
import io.surfworks.warpforge.launch.job.JobStatus;
import io.surfworks.warpforge.launch.job.JobSubmission;
import io.surfworks.warpforge.launch.job.ResourceRequirements;
import io.surfworks.warpforge.launch.scheduler.ClusterInfo;
import io.surfworks.warpforge.launch.scheduler.JobQuery;
import io.surfworks.warpforge.launch.scheduler.SchedulerException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ClusterSimulator.
 */
class ClusterSimulatorTest {

    private ClusterSimulator simulator;

    @BeforeEach
    void setUp() {
        simulator = ClusterSimulator.homogeneous(2, GpuType.NVIDIA, 2);
        simulator.setDefaultJobDuration(Duration.ofMillis(10));
    }

    @Test
    void homogeneousClusterCreation() throws SchedulerException {
        ClusterInfo info = simulator.clusterInfo();

        assertEquals(2, info.totalNodes());
        assertEquals(2, info.availableNodes());
        assertEquals(4, info.gpuCounts().get(GpuType.NVIDIA)); // 2 nodes * 2 GPUs
    }

    @Test
    void holmesMark1Configuration() throws SchedulerException {
        ClusterSimulator mark1 = ClusterSimulator.holmesMark1();
        ClusterInfo info = mark1.clusterInfo();

        assertEquals(2, info.totalNodes());
        // One NVIDIA, one AMD
        assertEquals(1, info.gpuCounts().get(GpuType.NVIDIA));
        assertEquals(1, info.gpuCounts().get(GpuType.AMD));
    }

    @Test
    void holmesMark2Configuration() throws SchedulerException {
        ClusterSimulator mark2 = ClusterSimulator.holmesMark2();
        ClusterInfo info = mark2.clusterInfo();

        assertEquals(10, info.totalNodes());
        // 5 NVIDIA nodes * 2 GPUs = 10
        assertEquals(10, info.gpuCounts().get(GpuType.NVIDIA));
        // 5 AMD nodes * 2 GPUs = 10
        assertEquals(10, info.gpuCounts().get(GpuType.AMD));
    }

    @Test
    void submitAndCompleteJob() throws SchedulerException, InterruptedException {
        JobSubmission submission = createSubmission(GpuType.NVIDIA, 1);
        String jobId = simulator.submit(submission);

        // Wait for job to complete
        TimeUnit.MILLISECONDS.sleep(50);
        simulator.advanceSimulation();

        JobStatus status = simulator.status(jobId);
        assertEquals(JobState.COMPLETED, status.state());
    }

    @Test
    void jobsQueueWhenNodesAreBusy() throws SchedulerException {
        // Set long job duration
        simulator.setDefaultJobDuration(Duration.ofHours(1));

        // Submit more jobs than available nodes
        JobSubmission submission = createSubmission(GpuType.NVIDIA, 1);
        simulator.submit(submission); // Node 0
        simulator.submit(submission); // Node 1
        simulator.submit(submission); // Should queue

        // Should have one job in pending queue
        assertEquals(1, simulator.getPendingQueueSize());
    }

    @Test
    void gpuTypeMatching() throws SchedulerException {
        // Create cluster with 1 NVIDIA, 1 AMD
        ClusterSimulator mixed = ClusterSimulator.holmesMark1();
        mixed.setDefaultJobDuration(Duration.ofHours(1));

        // Submit NVIDIA job
        JobSubmission nvidiaJob = createSubmission(GpuType.NVIDIA, 1);
        String nvidiaJobId = mixed.submit(nvidiaJob);

        // Submit AMD job
        JobSubmission amdJob = createSubmission(GpuType.AMD, 1);
        String amdJobId = mixed.submit(amdJob);

        // Both should be running (on appropriate nodes)
        assertEquals(0, mixed.getPendingQueueSize());

        JobStatus nvidiaStatus = mixed.status(nvidiaJobId);
        JobStatus amdStatus = mixed.status(amdJobId);

        assertEquals(JobState.RUNNING, nvidiaStatus.state());
        assertEquals(JobState.RUNNING, amdStatus.state());
    }

    @Test
    void cancelJob() throws SchedulerException {
        simulator.setDefaultJobDuration(Duration.ofHours(1));
        JobSubmission submission = createSubmission(GpuType.NVIDIA, 1);
        String jobId = simulator.submit(submission);

        boolean cancelled = simulator.cancel(jobId);
        assertTrue(cancelled);

        JobStatus status = simulator.status(jobId);
        assertEquals(JobState.CANCELLED, status.state());
    }

    @Test
    void nodeFailure() throws SchedulerException {
        ClusterInfo before = simulator.clusterInfo();
        assertEquals(2, before.availableNodes());

        simulator.failNode("node-0");

        ClusterInfo after = simulator.clusterInfo();
        assertEquals(1, after.availableNodes());

        // Verify the failed node shows in the list
        assertTrue(after.nodes().stream()
                .anyMatch(n -> n.name().equals("node-0") && n.status().equals("failed")));
    }

    @Test
    void nodeRecovery() throws SchedulerException {
        simulator.failNode("node-0");
        assertEquals(1, simulator.clusterInfo().availableNodes());

        simulator.recoverNode("node-0");
        assertEquals(2, simulator.clusterInfo().availableNodes());
    }

    @Test
    void nodeStats() throws SchedulerException, InterruptedException {
        JobSubmission submission = createSubmission(GpuType.NVIDIA, 1);
        simulator.submit(submission);

        TimeUnit.MILLISECONDS.sleep(50);
        simulator.advanceSimulation();

        var stats = simulator.getNodeStats();
        // At least one node should have run a job
        assertTrue(stats.values().stream().anyMatch(s -> s.totalJobsRun() > 0));
    }

    @Test
    void listJobs() throws SchedulerException {
        simulator.setDefaultJobDuration(Duration.ofMillis(1));
        JobSubmission submission = createSubmission(GpuType.ANY, 1);

        simulator.submit(submission);
        simulator.submit(submission);
        simulator.submit(submission);

        List<JobStatus> jobs = simulator.list(JobQuery.all());
        assertEquals(3, jobs.size());
    }

    @Test
    void isConnected() {
        assertTrue(simulator.isConnected());
    }

    private JobSubmission createSubmission(GpuType gpuType, int gpuCount) {
        ResourceRequirements resources;
        if (gpuCount > 0 && gpuType != GpuType.NONE) {
            resources = switch (gpuType) {
                case NVIDIA -> ResourceRequirements.nvidia(gpuCount);
                case AMD -> ResourceRequirements.amd(gpuCount);
                case ANY -> ResourceRequirements.anyGpu(gpuCount);
                default -> ResourceRequirements.cpuOnly(4, 4096);
            };
        } else {
            resources = ResourceRequirements.cpuOnly(4, 4096);
        }

        JobDefinition def = JobDefinition.builder()
                .name("test-job")
                .modelSource("/test.py")
                .modelClass("TestModel")
                .inputSpecs(InputSpec.f32(1, 8))
                .resources(resources)
                .build();

        return JobSubmission.submit(def, "test-user");
    }
}
