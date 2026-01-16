package io.surfworks.warpforge.launch.scheduler;

/**
 * Exception thrown when scheduler operations fail.
 */
public class SchedulerException extends Exception {

    public SchedulerException(String message) {
        super(message);
    }

    public SchedulerException(String message, Throwable cause) {
        super(message, cause);
    }

    public SchedulerException(Throwable cause) {
        super(cause);
    }
}
