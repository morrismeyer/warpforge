package io.surfworks.warpforge.launch.scheduler.local;

import io.surfworks.warpforge.launch.job.JobResult;
import io.surfworks.warpforge.launch.job.JobState;
import io.surfworks.warpforge.launch.job.JobStatus;
import io.surfworks.warpforge.launch.job.JobSubmission;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Internal representation of a job in the local scheduler.
 * Tracks job state and results.
 */
final class LocalJob {

    private final String jobId;
    private final JobSubmission submission;
    private final Instant submittedAt;
    private final AtomicReference<JobState> state;
    private volatile Instant startedAt;
    private volatile Instant completedAt;
    private volatile JobResult result;
    private volatile String nodeName;

    LocalJob(String jobId, JobSubmission submission) {
        this.jobId = jobId;
        this.submission = submission;
        this.submittedAt = Instant.now();
        this.state = new AtomicReference<>(JobState.PENDING);
    }

    String jobId() {
        return jobId;
    }

    JobSubmission submission() {
        return submission;
    }

    JobState state() {
        return state.get();
    }

    void markRunning() {
        if (state.compareAndSet(JobState.PENDING, JobState.RUNNING)) {
            startedAt = Instant.now();
            nodeName = "localhost";
        }
    }

    void complete(JobResult jobResult) {
        if (state.compareAndSet(JobState.RUNNING, JobState.COMPLETED)) {
            completedAt = Instant.now();
            result = jobResult;
        }
    }

    void fail(Throwable error) {
        if (state.compareAndSet(JobState.RUNNING, JobState.FAILED) ||
            state.compareAndSet(JobState.PENDING, JobState.FAILED)) {
            completedAt = Instant.now();
            result = JobResult.failure(
                    jobId,
                    submission.correlationId(),
                    error.getMessage(),
                    stackTraceToString(error),
                    elapsed()
            );
        }
    }

    void cancel() {
        JobState current = state.get();
        if (current == JobState.PENDING || current == JobState.RUNNING) {
            if (state.compareAndSet(current, JobState.CANCELLED)) {
                completedAt = Instant.now();
                result = JobResult.failure(
                        jobId,
                        submission.correlationId(),
                        "Job cancelled by user",
                        elapsed()
                );
            }
        }
    }

    void timeout() {
        if (state.compareAndSet(JobState.RUNNING, JobState.TIMEOUT)) {
            completedAt = Instant.now();
            result = JobResult.failure(
                    jobId,
                    submission.correlationId(),
                    "Job exceeded timeout: " + submission.definition().timeout(),
                    elapsed()
            );
        }
    }

    JobStatus toStatus() {
        return new JobStatus(
                jobId,
                submission.correlationId(),
                state.get(),
                completedAt != null ? completedAt : (startedAt != null ? startedAt : submittedAt),
                nodeName,
                elapsed(),
                state.get().name(),
                Map.of("scheduler", "local")
        );
    }

    JobResult result() {
        return result;
    }

    Duration elapsed() {
        if (startedAt == null) {
            return Duration.ZERO;
        }
        Instant end = completedAt != null ? completedAt : Instant.now();
        return Duration.between(startedAt, end);
    }

    private String stackTraceToString(Throwable t) {
        StringBuilder sb = new StringBuilder();
        for (StackTraceElement elem : t.getStackTrace()) {
            sb.append("\tat ").append(elem).append("\n");
        }
        return sb.toString();
    }
}
