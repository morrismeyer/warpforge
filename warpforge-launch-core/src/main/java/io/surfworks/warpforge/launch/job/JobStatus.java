package io.surfworks.warpforge.launch.job;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/**
 * Current status of a submitted job.
 *
 * @param jobId             Scheduler-assigned job ID
 * @param correlationId     Original correlation ID from submission
 * @param state             Current state of the job
 * @param stateChangedAt    When the job entered this state
 * @param nodeName          Node where job is/was running (null if not yet scheduled)
 * @param elapsed           Time elapsed since job started running
 * @param message           Human-readable status message
 * @param schedulerMetadata Scheduler-specific metadata
 */
public record JobStatus(
        String jobId,
        String correlationId,
        JobState state,
        Instant stateChangedAt,
        String nodeName,
        Duration elapsed,
        String message,
        Map<String, String> schedulerMetadata
) {

    public JobStatus {
        Objects.requireNonNull(jobId, "jobId cannot be null");
        Objects.requireNonNull(state, "state cannot be null");
        Objects.requireNonNull(stateChangedAt, "stateChangedAt cannot be null");

        elapsed = elapsed == null ? Duration.ZERO : elapsed;
        message = message == null ? state.name() : message;
        schedulerMetadata = schedulerMetadata == null ? Map.of() : Map.copyOf(schedulerMetadata);
    }

    /**
     * Creates a pending status.
     */
    public static JobStatus pending(String jobId, String correlationId) {
        return new JobStatus(
                jobId, correlationId, JobState.PENDING, Instant.now(),
                null, Duration.ZERO, "Waiting in queue", Map.of()
        );
    }

    /**
     * Creates a running status.
     */
    public static JobStatus running(String jobId, String correlationId, String nodeName) {
        return new JobStatus(
                jobId, correlationId, JobState.RUNNING, Instant.now(),
                nodeName, Duration.ZERO, "Executing", Map.of()
        );
    }

    /**
     * Creates a completed status.
     */
    public static JobStatus completed(String jobId, String correlationId, Duration elapsed) {
        return new JobStatus(
                jobId, correlationId, JobState.COMPLETED, Instant.now(),
                null, elapsed, "Completed successfully", Map.of()
        );
    }

    /**
     * Creates a failed status.
     */
    public static JobStatus failed(String jobId, String correlationId, String message) {
        return new JobStatus(
                jobId, correlationId, JobState.FAILED, Instant.now(),
                null, Duration.ZERO, message, Map.of()
        );
    }

    /**
     * Returns true if the job is in a terminal state.
     */
    public boolean isTerminal() {
        return state.isTerminal();
    }

    /**
     * Returns true if the job completed successfully.
     */
    public boolean isSuccess() {
        return state.isSuccess();
    }

    /**
     * Returns a new status with updated elapsed time.
     */
    public JobStatus withElapsed(Duration elapsed) {
        return new JobStatus(
                jobId, correlationId, state, stateChangedAt,
                nodeName, elapsed, message, schedulerMetadata
        );
    }

    /**
     * Returns a new status with additional metadata.
     */
    public JobStatus withMetadata(String key, String value) {
        var newMetadata = new java.util.HashMap<>(schedulerMetadata);
        newMetadata.put(key, value);
        return new JobStatus(
                jobId, correlationId, state, stateChangedAt,
                nodeName, elapsed, message, newMetadata
        );
    }
}
