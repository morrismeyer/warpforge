package io.surfworks.warpforge.launch.scheduler;

import io.surfworks.warpforge.launch.job.JobState;

import java.time.Instant;
import java.util.Objects;
import java.util.Set;

/**
 * Query parameters for listing jobs.
 *
 * @param states          Filter by job states (empty = all states)
 * @param submittedAfter  Only jobs submitted after this time
 * @param submittedBefore Only jobs submitted before this time
 * @param submittedBy     Filter by submitter username
 * @param limit           Maximum number of results (0 = unlimited)
 */
public record JobQuery(
        Set<JobState> states,
        Instant submittedAfter,
        Instant submittedBefore,
        String submittedBy,
        int limit
) {

    public JobQuery {
        Objects.requireNonNull(states, "states cannot be null");
        states = Set.copyOf(states);
        if (limit < 0) {
            throw new IllegalArgumentException("limit cannot be negative");
        }
    }

    /**
     * Creates a query that matches all jobs.
     */
    public static JobQuery all() {
        return new JobQuery(Set.of(), null, null, null, 0);
    }

    /**
     * Creates a query for running jobs only.
     */
    public static JobQuery running() {
        return new JobQuery(Set.of(JobState.RUNNING), null, null, null, 0);
    }

    /**
     * Creates a query for pending jobs only.
     */
    public static JobQuery pending() {
        return new JobQuery(Set.of(JobState.PENDING), null, null, null, 0);
    }

    /**
     * Creates a query for active (non-terminal) jobs.
     */
    public static JobQuery active() {
        return new JobQuery(Set.of(JobState.PENDING, JobState.RUNNING), null, null, null, 0);
    }

    /**
     * Creates a query for completed jobs (terminal states).
     */
    public static JobQuery completed() {
        return new JobQuery(
                Set.of(JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT),
                null, null, null, 0
        );
    }

    /**
     * Returns a new query with the specified limit.
     */
    public JobQuery withLimit(int limit) {
        return new JobQuery(states, submittedAfter, submittedBefore, submittedBy, limit);
    }

    /**
     * Returns a new query filtered by submitter.
     */
    public JobQuery byUser(String username) {
        return new JobQuery(states, submittedAfter, submittedBefore, username, limit);
    }

    /**
     * Returns a new query with time range.
     */
    public JobQuery inTimeRange(Instant after, Instant before) {
        return new JobQuery(states, after, before, submittedBy, limit);
    }

    /**
     * Returns true if this query matches all jobs (no filters).
     */
    public boolean matchesAll() {
        return states.isEmpty() &&
                submittedAfter == null &&
                submittedBefore == null &&
                submittedBy == null;
    }
}
