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
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * Mock scheduler for unit testing.
 *
 * <p>Provides deterministic behavior and records all submissions for assertions.
 * Supports configuring failures, delays, and custom behavior per job.
 */
public final class MockScheduler implements Scheduler {

    private final AtomicInteger jobCounter = new AtomicInteger(0);
    private final Map<String, MockJob> jobs = new ConcurrentHashMap<>();
    private final List<JobSubmission> submissions = new ArrayList<>();

    private boolean connected = true;
    private Duration defaultDelay = Duration.ZERO;
    private Function<JobSubmission, JobState> defaultOutcome = s -> JobState.COMPLETED;
    private SchedulerException submitException;

    @Override
    public String name() {
        return "mock";
    }

    @Override
    public SchedulerCapabilities capabilities() {
        return SchedulerCapabilities.fullFeatured(Set.of(GpuType.NVIDIA, GpuType.AMD));
    }

    @Override
    public synchronized String submit(JobSubmission submission) throws SchedulerException {
        if (submitException != null) {
            throw submitException;
        }

        String jobId = "mock-job-" + jobCounter.incrementAndGet();
        JobState outcome = defaultOutcome.apply(submission);

        MockJob job = new MockJob(
                jobId,
                submission.correlationId(),
                submission,
                outcome,
                defaultDelay
        );
        jobs.put(jobId, job);
        submissions.add(submission);

        return jobId;
    }

    @Override
    public JobStatus status(String jobId) throws SchedulerException {
        MockJob job = jobs.get(jobId);
        if (job == null) {
            throw new SchedulerException("Job not found: " + jobId);
        }
        return job.currentStatus();
    }

    @Override
    public JobResult result(String jobId) throws SchedulerException {
        JobStatus status = status(jobId);

        if (!status.state().isTerminal()) {
            throw new IllegalStateException(
                    "Job " + jobId + " is not complete (state: " + status.state() + ")");
        }

        MockJob job = jobs.get(jobId);
        if (status.state() == JobState.COMPLETED) {
            return JobResult.success(jobId, job.correlationId, List.of(), status.elapsed());
        } else {
            return JobResult.failure(jobId, job.correlationId, "Mock failure", status.elapsed());
        }
    }

    @Override
    public boolean cancel(String jobId) throws SchedulerException {
        MockJob job = jobs.get(jobId);
        if (job == null) {
            return false;
        }
        job.cancel();
        return true;
    }

    @Override
    public List<JobStatus> list(JobQuery query) throws SchedulerException {
        List<JobStatus> result = new ArrayList<>();
        for (MockJob job : jobs.values()) {
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
        return new ClusterInfo(
                "mock",
                "1.0.0",
                1,
                1,
                Map.of(GpuType.NVIDIA, 4, GpuType.AMD, 0),
                List.of(NodeInfo.nvidiaNode("mock-node", "ready", 4, 32000, 8))
        );
    }

    @Override
    public void close() {
        // No-op for mock
    }

    // ===== Test configuration methods =====

    /**
     * Sets whether the scheduler appears connected.
     */
    public MockScheduler setConnected(boolean connected) {
        this.connected = connected;
        return this;
    }

    /**
     * Sets the default delay before jobs complete.
     */
    public MockScheduler setDefaultDelay(Duration delay) {
        this.defaultDelay = delay;
        return this;
    }

    /**
     * Sets the default outcome for all submitted jobs.
     */
    public MockScheduler setDefaultOutcome(JobState state) {
        this.defaultOutcome = s -> state;
        return this;
    }

    /**
     * Sets a function to determine job outcome based on submission.
     */
    public MockScheduler setOutcomeFunction(Function<JobSubmission, JobState> function) {
        this.defaultOutcome = function;
        return this;
    }

    /**
     * Configures submit() to throw an exception.
     */
    public MockScheduler setSubmitException(SchedulerException exception) {
        this.submitException = exception;
        return this;
    }

    /**
     * Immediately completes a job.
     */
    public void completeJob(String jobId, JobState state) {
        MockJob job = jobs.get(jobId);
        if (job != null) {
            job.complete(state);
        }
    }

    // ===== Test assertion methods =====

    /**
     * Returns all submissions received.
     */
    public List<JobSubmission> getSubmissions() {
        return List.copyOf(submissions);
    }

    /**
     * Returns the number of submissions received.
     */
    public int getSubmissionCount() {
        return submissions.size();
    }

    /**
     * Returns true if a job with the given correlation ID was submitted.
     */
    public boolean hasSubmission(String correlationId) {
        return submissions.stream()
                .anyMatch(s -> s.correlationId().equals(correlationId));
    }

    /**
     * Clears all recorded submissions.
     */
    public void clearSubmissions() {
        submissions.clear();
        jobs.clear();
        jobCounter.set(0);
    }

    /**
     * Internal mock job tracking.
     */
    private static class MockJob {
        final String jobId;
        final String correlationId;
        final JobSubmission submission;
        final Duration delay;
        final Instant submitTime;

        JobState targetState;
        JobState currentState;
        Instant completionTime;

        MockJob(String jobId, String correlationId, JobSubmission submission,
                JobState targetState, Duration delay) {
            this.jobId = jobId;
            this.correlationId = correlationId;
            this.submission = submission;
            this.targetState = targetState;
            this.delay = delay;
            this.submitTime = Instant.now();
            this.currentState = JobState.PENDING;
        }

        JobStatus currentStatus() {
            updateState();
            Duration elapsed = completionTime != null ?
                    Duration.between(submitTime, completionTime) :
                    Duration.between(submitTime, Instant.now());

            return new JobStatus(
                    jobId,
                    correlationId,
                    currentState,
                    submitTime,
                    "mock-node",
                    elapsed,
                    currentState.name(),
                    Map.of()
            );
        }

        void updateState() {
            if (currentState.isTerminal()) {
                return;
            }

            Duration elapsed = Duration.between(submitTime, Instant.now());

            if (elapsed.compareTo(delay) >= 0) {
                currentState = targetState;
                completionTime = Instant.now();
            } else if (elapsed.toMillis() > 10) {
                currentState = JobState.RUNNING;
            }
        }

        void complete(JobState state) {
            this.currentState = state;
            this.completionTime = Instant.now();
        }

        void cancel() {
            if (!currentState.isTerminal()) {
                this.currentState = JobState.CANCELLED;
                this.completionTime = Instant.now();
            }
        }
    }
}
