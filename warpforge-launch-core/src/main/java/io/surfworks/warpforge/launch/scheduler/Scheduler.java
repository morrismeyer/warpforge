package io.surfworks.warpforge.launch.scheduler;

import io.surfworks.warpforge.launch.job.JobResult;
import io.surfworks.warpforge.launch.job.JobState;
import io.surfworks.warpforge.launch.job.JobStatus;
import io.surfworks.warpforge.launch.job.JobSubmission;

import java.time.Duration;
import java.util.List;

/**
 * Service Provider Interface for job schedulers.
 * Implementations handle scheduler-specific submission and monitoring.
 *
 * <p>Schedulers are responsible for:
 * <ul>
 *   <li>Submitting jobs to the underlying compute infrastructure</li>
 *   <li>Monitoring job status and collecting results</li>
 *   <li>Managing job lifecycle (cancellation, timeout)</li>
 *   <li>Providing cluster topology information</li>
 * </ul>
 *
 * <p>Implementations must be thread-safe.
 */
public interface Scheduler extends AutoCloseable {

    /**
     * Returns the name of this scheduler (e.g., "local", "ray", "kubernetes", "slurm").
     */
    String name();

    /**
     * Returns the capabilities of this scheduler.
     */
    SchedulerCapabilities capabilities();

    /**
     * Submits a job for execution.
     *
     * @param submission The job submission request
     * @return Job ID assigned by the scheduler
     * @throws SchedulerException if submission fails
     */
    String submit(JobSubmission submission) throws SchedulerException;

    /**
     * Gets the current status of a job.
     *
     * @param jobId The scheduler-assigned job ID
     * @return Current job status
     * @throws SchedulerException if status retrieval fails
     */
    JobStatus status(String jobId) throws SchedulerException;

    /**
     * Gets the result of a completed job.
     *
     * @param jobId The scheduler-assigned job ID
     * @return Job result with outputs
     * @throws SchedulerException if result retrieval fails
     * @throws IllegalStateException if the job is not in a terminal state
     */
    JobResult result(String jobId) throws SchedulerException;

    /**
     * Cancels a running or pending job.
     *
     * @param jobId The scheduler-assigned job ID
     * @return true if cancellation was successful
     * @throws SchedulerException if cancellation fails
     */
    boolean cancel(String jobId) throws SchedulerException;

    /**
     * Lists jobs matching the query criteria.
     *
     * @param query Query parameters for filtering
     * @return List of matching job statuses
     * @throws SchedulerException if listing fails
     */
    List<JobStatus> list(JobQuery query) throws SchedulerException;

    /**
     * Waits for a job to complete (blocking with timeout).
     *
     * @param jobId   The scheduler-assigned job ID
     * @param timeout Maximum time to wait
     * @return Job result when completed
     * @throws SchedulerException if waiting fails or times out
     */
    default JobResult awaitCompletion(String jobId, Duration timeout) throws SchedulerException {
        long deadlineMillis = System.currentTimeMillis() + timeout.toMillis();
        long pollIntervalMillis = 1000;

        while (System.currentTimeMillis() < deadlineMillis) {
            JobStatus s = status(jobId);
            if (s.state().isTerminal()) {
                return result(jobId);
            }
            try {
                Thread.sleep(Math.min(pollIntervalMillis, deadlineMillis - System.currentTimeMillis()));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new SchedulerException("Interrupted while waiting for job " + jobId, e);
            }
        }
        throw new SchedulerException("Timeout waiting for job " + jobId + " after " + timeout);
    }

    /**
     * Tests connection to the scheduler.
     *
     * @return true if the scheduler is reachable and operational
     */
    boolean isConnected();

    /**
     * Gets information about the cluster.
     *
     * @return Cluster topology and status information
     * @throws SchedulerException if cluster info retrieval fails
     */
    ClusterInfo clusterInfo() throws SchedulerException;

    /**
     * Closes this scheduler and releases any resources.
     */
    @Override
    void close();
}
