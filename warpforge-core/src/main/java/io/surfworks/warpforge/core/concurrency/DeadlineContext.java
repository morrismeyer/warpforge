package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;

import java.time.Duration;
import java.time.Instant;
import java.util.function.Function;

/**
 * Deadline-aware execution context for SLA-bounded GPU operations.
 *
 * <p>DeadlineContext wraps a {@link GpuTaskScope} with timeout semantics,
 * enabling graceful degradation when operations exceed their deadline.
 * This is essential for inference serving where latency SLAs must be met.
 *
 * <p>This pattern is validated by research:
 * <ul>
 *   <li>Alibaba Aegaeon (SOSP 2025): 82% GPU reduction via proactive scheduling</li>
 *   <li>Tally (ASPLOS 2025): Deadline-aware batching for inference serving</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * DeadlineContext ctx = DeadlineContext.withTimeout(backend, Duration.ofMillis(100));
 *
 * try {
 *     Tensor result = ctx.execute(scope -> {
 *         // Execute layers with deadline checks
 *         Tensor x = input;
 *         for (Layer layer : model.layers()) {
 *             ctx.checkDeadline();  // Throws if past deadline
 *             x = layer.forward(x, scope);
 *         }
 *         return x;
 *     });
 *     return Response.success(result);
 * } catch (DeadlineExceededException e) {
 *     return Response.timeout(e.getElapsedTime(), e.getRemainingWork());
 * }
 * }</pre>
 *
 * @see GpuTaskScope
 * @see DeadlineExceededException
 */
public final class DeadlineContext {

    private final GpuBackend backend;
    private final Instant deadline;
    private final Duration timeout;
    private volatile boolean cancelled;

    private DeadlineContext(GpuBackend backend, Instant deadline, Duration timeout) {
        this.backend = backend;
        this.deadline = deadline;
        this.timeout = timeout;
    }

    /**
     * Creates a deadline context with an absolute deadline.
     *
     * @param backend the GPU backend
     * @param deadline the absolute time by which the operation must complete
     * @return a new DeadlineContext
     */
    public static DeadlineContext withDeadline(GpuBackend backend, Instant deadline) {
        return new DeadlineContext(backend, deadline,
                Duration.between(Instant.now(), deadline));
    }

    /**
     * Creates a deadline context with a relative timeout.
     *
     * @param backend the GPU backend
     * @param timeout maximum duration for the operation
     * @return a new DeadlineContext
     */
    public static DeadlineContext withTimeout(GpuBackend backend, Duration timeout) {
        return new DeadlineContext(backend, Instant.now().plus(timeout), timeout);
    }

    /**
     * Executes an operation within a deadline-bounded GPU scope.
     *
     * <p>The operation receives a {@link GpuTaskScope} for forking GPU tasks.
     * The deadline is checked before execution starts. For finer-grained
     * deadline checking, call {@link #checkDeadline()} within the operation.
     *
     * @param operation function that uses the GpuTaskScope
     * @param <T> the result type
     * @return the operation result
     * @throws DeadlineExceededException if the deadline is exceeded
     */
    public <T> T execute(Function<GpuTaskScope, T> operation) throws DeadlineExceededException {
        checkDeadline();
        try (GpuTaskScope scope = GpuTaskScope.open(backend, "deadline-context")) {
            return operation.apply(scope);
        }
    }

    /**
     * Checks if the deadline has been exceeded.
     *
     * <p>Call this periodically within long-running operations to enable
     * early termination when the deadline is exceeded.
     *
     * @throws DeadlineExceededException if the deadline has passed or the context was cancelled
     */
    public void checkDeadline() throws DeadlineExceededException {
        if (cancelled || isExpired()) {
            throw new DeadlineExceededException(
                    Duration.between(deadline.minus(timeout), Instant.now()),
                    timeout,
                    "deadline-check");
        }
    }

    /**
     * Returns the remaining time until the deadline.
     *
     * @return remaining duration (may be negative if deadline passed)
     */
    public Duration remainingTime() {
        return Duration.between(Instant.now(), deadline);
    }

    /**
     * Returns true if the deadline has passed.
     */
    public boolean isExpired() {
        return Instant.now().isAfter(deadline);
    }

    /**
     * Returns true if this context has been cancelled.
     */
    public boolean isCancelled() {
        return cancelled;
    }

    /**
     * Cancels this deadline context.
     *
     * <p>Subsequent calls to {@link #checkDeadline()} will throw
     * {@link DeadlineExceededException}.
     */
    public void cancel() {
        this.cancelled = true;
    }

    /**
     * Returns the absolute deadline instant.
     */
    public Instant deadline() {
        return deadline;
    }

    /**
     * Returns the original timeout duration.
     */
    public Duration timeout() {
        return timeout;
    }

    /**
     * Returns the GPU backend.
     */
    public GpuBackend backend() {
        return backend;
    }
}
