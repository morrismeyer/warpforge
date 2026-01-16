package io.surfworks.warpforge.launch.job;

import java.time.Instant;
import java.util.Objects;
import java.util.UUID;

/**
 * A job submission request to a scheduler.
 *
 * @param definition    The job definition describing what to run
 * @param submittedBy   Username of the submitter
 * @param submittedAt   Timestamp of submission
 * @param correlationId Unique ID for tracking this submission across systems
 */
public record JobSubmission(
        JobDefinition definition,
        String submittedBy,
        Instant submittedAt,
        String correlationId
) {

    public JobSubmission {
        Objects.requireNonNull(definition, "definition cannot be null");
        Objects.requireNonNull(submittedBy, "submittedBy cannot be null");
        Objects.requireNonNull(submittedAt, "submittedAt cannot be null");
        Objects.requireNonNull(correlationId, "correlationId cannot be null");

        if (submittedBy.isBlank()) {
            throw new IllegalArgumentException("submittedBy cannot be blank");
        }
        if (correlationId.isBlank()) {
            throw new IllegalArgumentException("correlationId cannot be blank");
        }
    }

    /**
     * Creates a new submission for the given job definition.
     * Uses the current user and generates a unique correlation ID.
     */
    public static JobSubmission submit(JobDefinition definition) {
        return new JobSubmission(
                definition,
                System.getProperty("user.name", "unknown"),
                Instant.now(),
                UUID.randomUUID().toString()
        );
    }

    /**
     * Creates a new submission with a specific submitter.
     */
    public static JobSubmission submit(JobDefinition definition, String submittedBy) {
        return new JobSubmission(
                definition,
                submittedBy,
                Instant.now(),
                UUID.randomUUID().toString()
        );
    }

    /**
     * Returns a short version of the correlation ID (first 8 chars) for display.
     */
    public String shortCorrelationId() {
        return correlationId.length() > 8 ? correlationId.substring(0, 8) : correlationId;
    }
}
