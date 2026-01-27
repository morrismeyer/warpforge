package io.surfworks.warpforge.core.profiling;

import io.surfworks.warpforge.core.jfr.GpuProfilingEvent;

import java.lang.foreign.Arena;
import java.util.List;
import java.util.Optional;

/**
 * Interface for hardware-level GPU profiling.
 *
 * <p>Implementations of this interface provide access to hardware performance
 * counters through profiling APIs:
 * <ul>
 *   <li>NVIDIA: CUPTI (CUDA Profiling Tools Interface)</li>
 *   <li>AMD: roctracer / rocprofiler</li>
 * </ul>
 *
 * <p>Hardware profiling provides achieved metrics (as opposed to theoretical
 * estimates), including:
 * <ul>
 *   <li>Achieved occupancy (vs theoretical from launch config)</li>
 *   <li>SM/CU efficiency and warp scheduling metrics</li>
 *   <li>Cache hit rates and memory throughput</li>
 *   <li>Warp stall analysis for bottleneck identification</li>
 * </ul>
 *
 * <p>Usage pattern:
 * <pre>{@code
 * try (HardwareProfiler profiler = HardwareProfiler.create(deviceIndex)) {
 *     // Start profiling session
 *     profiler.startSession();
 *
 *     // Execute kernels...
 *     long correlationId = launchKernel(stream, kernel, args);
 *
 *     // Get metrics for completed kernel
 *     Optional<ProfilingMetrics> metrics = profiler.getMetrics(correlationId);
 *
 *     // Populate JFR event
 *     metrics.ifPresent(m -> m.populateEvent(event));
 *
 *     profiler.stopSession();
 * }
 * }</pre>
 *
 * <p>Note: Hardware profiling adds overhead to kernel execution. It is
 * intended for development and debugging, not production workloads.
 *
 * @see GpuProfilingEvent
 * @see io.surfworks.warpforge.core.kernel.OccupancyCalculator
 */
public interface HardwareProfiler extends AutoCloseable {

    /**
     * Profiling metrics for a single kernel execution.
     *
     * <p>These metrics are obtained from hardware performance counters
     * after kernel execution completes.
     */
    interface ProfilingMetrics {

        /**
         * Get the correlation ID linking to the kernel launch.
         *
         * @return correlation ID
         */
        long correlationId();

        /**
         * Get the kernel function name.
         *
         * @return kernel name
         */
        String kernelName();

        /**
         * Get achieved occupancy (0-100).
         *
         * @return achieved occupancy percentage
         */
        double achievedOccupancyPercent();

        /**
         * Get SM/CU efficiency (0-100).
         *
         * @return SM efficiency percentage
         */
        double smEfficiencyPercent();

        /**
         * Get compute throughput as percentage of peak (0-100).
         *
         * @return compute throughput percentage
         */
        double computeThroughputPercent();

        /**
         * Get memory throughput as percentage of peak (0-100).
         *
         * @return memory throughput percentage
         */
        double memoryThroughputPercent();

        /**
         * Get L1 cache hit rate (0-100).
         *
         * @return L1 cache hit rate percentage
         */
        double l1CacheHitRatePercent();

        /**
         * Get L2 cache hit rate (0-100).
         *
         * @return L2 cache hit rate percentage
         */
        double l2CacheHitRatePercent();

        /**
         * Get percentage of stalls due to memory dependencies.
         *
         * @return memory dependency stall percentage
         */
        double stallMemoryDependencyPercent();

        /**
         * Get percentage of stalls due to execution dependencies.
         *
         * @return execution dependency stall percentage
         */
        double stallExecutionDependencyPercent();

        /**
         * Get percentage of stalls due to synchronization barriers.
         *
         * @return synchronization stall percentage
         */
        double stallSynchronizationPercent();

        /**
         * Get kernel execution duration in nanoseconds.
         *
         * @return kernel duration in nanoseconds
         */
        long kernelDurationNanos();

        /**
         * Get the profiling overhead in nanoseconds.
         *
         * @return profiling overhead in nanoseconds
         */
        long profilingOverheadNanos();

        /**
         * Populate a GpuProfilingEvent with these metrics.
         *
         * @param event the event to populate
         */
        default void populateEvent(GpuProfilingEvent event) {
            event.correlationId = correlationId();
            event.kernelName = kernelName();
            event.achievedOccupancyPercent = achievedOccupancyPercent();
            event.smEfficiencyPercent = smEfficiencyPercent();
            event.computeThroughputPercent = computeThroughputPercent();
            event.memoryThroughputPercent = memoryThroughputPercent();
            event.l1CacheHitRatePercent = l1CacheHitRatePercent();
            event.l2CacheHitRatePercent = l2CacheHitRatePercent();
            event.stallMemoryDependencyPercent = stallMemoryDependencyPercent();
            event.stallExecutionDependencyPercent = stallExecutionDependencyPercent();
            event.stallSynchronizationPercent = stallSynchronizationPercent();
            event.kernelDurationNanos = kernelDurationNanos();
            event.profilingOverheadNanos = profilingOverheadNanos();
        }
    }

    /**
     * Profiling session configuration.
     */
    record SessionConfig(
        /**
         * Whether to collect occupancy metrics.
         */
        boolean collectOccupancy,

        /**
         * Whether to collect cache metrics.
         */
        boolean collectCacheMetrics,

        /**
         * Whether to collect stall analysis metrics.
         */
        boolean collectStallAnalysis,

        /**
         * Whether to collect throughput metrics.
         */
        boolean collectThroughput,

        /**
         * Maximum number of kernels to profile (0 = unlimited).
         */
        int maxKernels
    ) {
        /**
         * Default configuration that collects all metrics.
         */
        public static final SessionConfig ALL = new SessionConfig(
            true, true, true, true, 0
        );

        /**
         * Lightweight configuration for occupancy only.
         */
        public static final SessionConfig OCCUPANCY_ONLY = new SessionConfig(
            true, false, false, false, 0
        );

        /**
         * Configuration for bottleneck analysis (occupancy + stalls).
         */
        public static final SessionConfig BOTTLENECK_ANALYSIS = new SessionConfig(
            true, true, true, true, 0
        );
    }

    /**
     * Start a profiling session with default configuration.
     *
     * <p>Call this before executing kernels to be profiled.
     *
     * @throws IllegalStateException if a session is already active
     */
    default void startSession() {
        startSession(SessionConfig.ALL);
    }

    /**
     * Start a profiling session with the specified configuration.
     *
     * @param config the session configuration
     * @throws IllegalStateException if a session is already active
     */
    void startSession(SessionConfig config);

    /**
     * Stop the current profiling session.
     *
     * <p>Metrics for kernels executed during the session remain available
     * until the profiler is closed.
     *
     * @throws IllegalStateException if no session is active
     */
    void stopSession();

    /**
     * Check if a profiling session is currently active.
     *
     * @return true if a session is active
     */
    boolean isSessionActive();

    /**
     * Get profiling metrics for a kernel by correlation ID.
     *
     * <p>The kernel must have completed execution before metrics are available.
     *
     * @param correlationId the correlation ID from kernel launch
     * @return metrics if available, empty if not yet available or not found
     */
    Optional<ProfilingMetrics> getMetrics(long correlationId);

    /**
     * Get all available profiling metrics from the current or last session.
     *
     * @return list of all available metrics
     */
    List<ProfilingMetrics> getAllMetrics();

    /**
     * Flush any pending profiling data.
     *
     * <p>Call this after a synchronization point to ensure all kernel
     * metrics are available.
     */
    void flush();

    /**
     * Clear all collected metrics.
     */
    void clearMetrics();

    /**
     * Get the backend name (e.g., "CUPTI" or "roctracer").
     *
     * @return backend name
     */
    String getBackendName();

    /**
     * Get the device index this profiler is attached to.
     *
     * @return device index
     */
    int getDeviceIndex();

    /**
     * Check if hardware profiling is supported on this device.
     *
     * @return true if profiling is supported
     */
    boolean isSupported();

    /**
     * Get the profiling overhead estimate in nanoseconds.
     *
     * <p>This is a rough estimate of the overhead added to each kernel
     * execution when profiling is enabled.
     *
     * @return estimated overhead in nanoseconds
     */
    long estimatedOverheadNanos();

    /**
     * Close the profiler and release resources.
     */
    @Override
    void close();
}
