package io.surfworks.warpforge.core.concurrency;

import java.time.Duration;

/**
 * Thrown when a GPU operation exceeds its deadline.
 *
 * <p>This exception contains diagnostic information about the timeout,
 * useful for logging, metrics, and determining appropriate fallback behavior.
 *
 * <p>Example handling:
 * <pre>{@code
 * try {
 *     result = ctx.execute(scope -> runInference(input, scope));
 * } catch (DeadlineExceededException e) {
 *     log.warn("Inference timeout: elapsed={}, allowed={}, overrun={}x",
 *              e.getElapsedTime(), e.getAllowedTime(), e.getOverrunRatio());
 *
 *     if (e.getOverrunRatio() < 1.5) {
 *         // Slightly over - return partial result if available
 *         return Response.partial(partialResult);
 *     } else {
 *         // Significantly over - return timeout error
 *         return Response.timeout();
 *     }
 * }
 * }</pre>
 *
 * @see DeadlineContext
 */
public class DeadlineExceededException extends Exception {

    private final Duration elapsedTime;
    private final Duration allowedTime;
    private final String lastOperation;

    /**
     * Creates a new DeadlineExceededException.
     *
     * @param elapsed time elapsed since the deadline context was created
     * @param allowed the configured timeout duration
     * @param lastOp description of the operation that was running when deadline hit
     */
    public DeadlineExceededException(Duration elapsed, Duration allowed, String lastOp) {
        super(String.format("Deadline exceeded: elapsed %s, allowed %s (%.1fx overrun)",
                elapsed, allowed, (double) elapsed.toNanos() / allowed.toNanos()));
        this.elapsedTime = elapsed;
        this.allowedTime = allowed;
        this.lastOperation = lastOp;
    }

    /**
     * Returns the time elapsed since the deadline context was created.
     */
    public Duration getElapsedTime() {
        return elapsedTime;
    }

    /**
     * Returns the configured timeout duration.
     */
    public Duration getAllowedTime() {
        return allowedTime;
    }

    /**
     * Returns a description of the operation that was running when the deadline hit.
     */
    public String getLastOperation() {
        return lastOperation;
    }

    /**
     * Returns the ratio of elapsed time to allowed time.
     *
     * <p>A value of 1.0 means the operation took exactly the allowed time.
     * Values &gt; 1.0 indicate how much the deadline was exceeded.
     *
     * @return overrun ratio (elapsed / allowed)
     */
    public double getOverrunRatio() {
        return (double) elapsedTime.toNanos() / allowedTime.toNanos();
    }

    /**
     * Returns how much time the operation exceeded the deadline by.
     *
     * @return overrun duration (elapsed - allowed), or zero if not exceeded
     */
    public Duration getOverrunDuration() {
        Duration overrun = elapsedTime.minus(allowedTime);
        return overrun.isNegative() ? Duration.ZERO : overrun;
    }
}
