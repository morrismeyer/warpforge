package io.surfworks.warpforge.core.kernel;

import io.surfworks.warpforge.core.jfr.GpuKernelEvent;

/**
 * Measures GPU kernel launch latency.
 *
 * <p>Launch latency is the CPU-side overhead of submitting work to the GPU.
 * This includes driver overhead, command buffer preparation, and stream queue
 * insertion - but NOT the time spent waiting for the kernel to execute.
 *
 * <p>There are two timing components:
 * <ul>
 *   <li><b>Launch Latency:</b> CPU time spent in the kernel launch API call.
 *       Measured as the wall-clock time between starting and returning from
 *       the launch function (cuLaunchKernel, hipModuleLaunchKernel, etc.)</li>
 *   <li><b>Queue Delay:</b> Time the kernel waits in the stream queue before
 *       GPU execution begins. Requires GPU-side timing (CUDA/HIP events).</li>
 * </ul>
 *
 * <p>Usage pattern:
 * <pre>{@code
 * LaunchTimer timer = LaunchTimer.start();
 *
 * // Launch the kernel (asynchronous call)
 * CudaRuntime.launchKernel(function, gridDim, blockDim, ...);
 *
 * timer.markLaunched();
 *
 * // Later, populate the JFR event
 * timer.populateEvent(event);
 * }</pre>
 *
 * <p>For queue delay measurement, GPU events must be used:
 * <pre>{@code
 * LaunchTimer timer = LaunchTimer.start();
 *
 * // Record event BEFORE launch
 * long preEvent = CudaRuntime.eventCreate(arena);
 * CudaRuntime.eventRecord(preEvent, stream);
 *
 * // Launch the kernel
 * CudaRuntime.launchKernel(...);
 * timer.markLaunched();
 *
 * // Record event AFTER launch
 * long postEvent = CudaRuntime.eventCreate(arena);
 * CudaRuntime.eventRecord(postEvent, stream);
 *
 * // After synchronization, calculate queue delay
 * float queueDelayMs = CudaRuntime.eventElapsedTime(arena, preEvent, postEvent);
 * timer.setQueueDelayNanos((long)(queueDelayMs * 1_000_000));
 *
 * timer.populateEvent(event);
 * }</pre>
 *
 * @see GpuKernelEvent
 */
public final class LaunchTimer {

    private final long startNanos;
    private long launchNanos;
    private long launchLatencyNanos;
    private long queueDelayNanos;

    private LaunchTimer(long startNanos) {
        this.startNanos = startNanos;
        this.launchNanos = 0;
        this.launchLatencyNanos = 0;
        this.queueDelayNanos = 0;
    }

    /**
     * Start timing a kernel launch.
     *
     * <p>Call this immediately before invoking the kernel launch API.
     *
     * @return a new LaunchTimer instance
     */
    public static LaunchTimer start() {
        return new LaunchTimer(System.nanoTime());
    }

    /**
     * Start timing with a specific timestamp.
     *
     * <p>Useful for testing or when the start time is captured elsewhere.
     *
     * @param startNanos the start timestamp in nanoseconds
     * @return a new LaunchTimer instance
     */
    public static LaunchTimer startAt(long startNanos) {
        return new LaunchTimer(startNanos);
    }

    /**
     * Mark that the kernel launch has completed.
     *
     * <p>Call this immediately after the kernel launch API returns.
     * The launch latency is computed as the elapsed time since start.
     */
    public void markLaunched() {
        this.launchNanos = System.nanoTime();
        this.launchLatencyNanos = launchNanos - startNanos;
    }

    /**
     * Mark launch completed with a specific timestamp.
     *
     * <p>Useful for testing or when the end time is captured elsewhere.
     *
     * @param launchNanos the launch completion timestamp in nanoseconds
     */
    public void markLaunchedAt(long launchNanos) {
        this.launchNanos = launchNanos;
        this.launchLatencyNanos = launchNanos - startNanos;
    }

    /**
     * Set the queue delay measured by GPU events.
     *
     * <p>This is the time between when the kernel was submitted to the stream
     * and when it actually started executing on the GPU. This requires using
     * CUDA/HIP events to measure accurately.
     *
     * @param queueDelayNanos queue delay in nanoseconds
     */
    public void setQueueDelayNanos(long queueDelayNanos) {
        this.queueDelayNanos = queueDelayNanos;
    }

    /**
     * Get the start timestamp.
     *
     * @return start timestamp in nanoseconds
     */
    public long startNanos() {
        return startNanos;
    }

    /**
     * Get the launch completion timestamp.
     *
     * @return launch timestamp in nanoseconds, or 0 if not yet marked
     */
    public long launchNanos() {
        return launchNanos;
    }

    /**
     * Get the launch latency.
     *
     * @return launch latency in nanoseconds, or 0 if not yet marked
     */
    public long launchLatencyNanos() {
        return launchLatencyNanos;
    }

    /**
     * Get the queue delay.
     *
     * @return queue delay in nanoseconds, or 0 if not set
     */
    public long queueDelayNanos() {
        return queueDelayNanos;
    }

    /**
     * Check if the launch has been marked.
     *
     * @return true if markLaunched() has been called
     */
    public boolean isLaunched() {
        return launchNanos > 0;
    }

    /**
     * Populate a GpuKernelEvent with timing information.
     *
     * @param event the JFR event to populate
     */
    public void populateEvent(GpuKernelEvent event) {
        event.launchLatencyNanos = this.launchLatencyNanos;
        event.queueDelayNanos = this.queueDelayNanos;
    }

    /**
     * Get total submission overhead (launch latency + queue delay).
     *
     * <p>This is the total time from when the CPU started submitting the kernel
     * to when the GPU actually began execution.
     *
     * @return total overhead in nanoseconds
     */
    public long totalOverheadNanos() {
        return launchLatencyNanos + queueDelayNanos;
    }

    /**
     * Time a synchronous kernel execution.
     *
     * <p>This is a convenience method for simple cases where the kernel
     * execution is synchronous (the launch call blocks until completion).
     *
     * @param <T> return type of the kernel operation
     * @param operation the kernel operation to time
     * @return result containing the operation's return value and timing
     * @throws Exception if the operation throws
     */
    public static <T> TimedResult<T> time(ThrowingSupplier<T> operation) throws Exception {
        LaunchTimer timer = start();
        T result = operation.get();
        timer.markLaunched();
        return new TimedResult<>(result, timer);
    }

    /**
     * Time a void kernel execution.
     *
     * @param operation the kernel operation to time
     * @return the timer with launch latency recorded
     * @throws Exception if the operation throws
     */
    public static LaunchTimer timeVoid(ThrowingRunnable operation) throws Exception {
        LaunchTimer timer = start();
        operation.run();
        timer.markLaunched();
        return timer;
    }

    /**
     * Result of a timed operation.
     *
     * @param <T> type of the result
     * @param result the operation's return value
     * @param timer the timer with recorded latencies
     */
    public record TimedResult<T>(T result, LaunchTimer timer) {
        /**
         * Get the launch latency.
         */
        public long launchLatencyNanos() {
            return timer.launchLatencyNanos();
        }

        /**
         * Populate a JFR event with timing.
         */
        public void populateEvent(GpuKernelEvent event) {
            timer.populateEvent(event);
        }
    }

    /**
     * Functional interface for operations that return a value and may throw.
     */
    @FunctionalInterface
    public interface ThrowingSupplier<T> {
        T get() throws Exception;
    }

    /**
     * Functional interface for void operations that may throw.
     */
    @FunctionalInterface
    public interface ThrowingRunnable {
        void run() throws Exception;
    }

    @Override
    public String toString() {
        if (launchNanos == 0) {
            return String.format("LaunchTimer[started, not launched]");
        }
        return String.format("LaunchTimer[launchLatency=%dns, queueDelay=%dns, total=%dns]",
            launchLatencyNanos, queueDelayNanos, totalOverheadNanos());
    }
}
