package io.surfworks.warpforge.launch.job;

/**
 * Possible states of a job during its lifecycle.
 */
public enum JobState {
    /** Job is waiting in queue to be scheduled */
    PENDING,

    /** Job is currently executing */
    RUNNING,

    /** Job finished successfully */
    COMPLETED,

    /** Job finished with an error */
    FAILED,

    /** Job was cancelled by user */
    CANCELLED,

    /** Job exceeded its timeout */
    TIMEOUT;

    /**
     * Returns true if this is a terminal state (job will not change state again).
     */
    public boolean isTerminal() {
        return this == COMPLETED || this == FAILED || this == CANCELLED || this == TIMEOUT;
    }

    /**
     * Returns true if this state indicates successful completion.
     */
    public boolean isSuccess() {
        return this == COMPLETED;
    }
}
