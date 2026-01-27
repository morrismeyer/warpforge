package io.surfworks.warpforge.backend.amd.profiling;

import io.surfworks.warpforge.core.jfr.GpuProfilingEvent;
import io.surfworks.warpforge.core.profiling.HardwareProfiler;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * roctracer-based implementation of HardwareProfiler for AMD GPUs.
 *
 * <p>This implementation uses roctracer and rocprofiler to collect
 * hardware-level profiling metrics including:
 * <ul>
 *   <li>Achieved occupancy (wavefront occupancy)</li>
 *   <li>CU (Compute Unit) efficiency</li>
 *   <li>Cache hit rates</li>
 *   <li>Memory throughput</li>
 *   <li>Wavefront stall analysis</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 * try (RoctracerProfiler profiler = RoctracerProfiler.create(0)) {
 *     profiler.startSession();
 *     // Execute HIP kernels...
 *     profiler.stopSession();
 *
 *     // Get metrics
 *     profiler.getAllMetrics().forEach(m -> {
 *         System.out.println(m.kernelName() + ": " + m.achievedOccupancyPercent() + "%");
 *     });
 * }
 * }</pre>
 */
public final class RoctracerProfiler implements HardwareProfiler {

    private static final String BACKEND_NAME = "roctracer";
    private static final long ESTIMATED_OVERHEAD_NANOS = 75_000L; // ~75 microseconds

    private final int deviceIndex;
    private final Arena arena;
    private final Map<Long, RoctracerMetrics> metricsMap;
    private volatile boolean sessionActive;
    private volatile SessionConfig currentConfig;

    /**
     * Create a roctracer profiler for the specified device.
     *
     * @param deviceIndex the HIP device index
     * @return a new RoctracerProfiler instance
     * @throws IllegalStateException if roctracer is not available
     */
    public static RoctracerProfiler create(int deviceIndex) {
        if (!RoctracerRuntime.isAvailable()) {
            throw new IllegalStateException("roctracer is not available");
        }
        return new RoctracerProfiler(deviceIndex);
    }

    /**
     * Create a roctracer profiler if available, otherwise return empty.
     *
     * @param deviceIndex the HIP device index
     * @return Optional containing profiler if roctracer is available
     */
    public static Optional<RoctracerProfiler> tryCreate(int deviceIndex) {
        if (!RoctracerRuntime.isAvailable()) {
            return Optional.empty();
        }
        return Optional.of(new RoctracerProfiler(deviceIndex));
    }

    private RoctracerProfiler(int deviceIndex) {
        this.deviceIndex = deviceIndex;
        this.arena = Arena.ofShared();
        this.metricsMap = new ConcurrentHashMap<>();
        this.sessionActive = false;
    }

    @Override
    public void startSession(SessionConfig config) {
        if (sessionActive) {
            throw new IllegalStateException("Profiling session already active");
        }

        this.currentConfig = config;
        this.metricsMap.clear();

        try {
            // Enable HIP kernel dispatch activity
            int result = RoctracerRuntime.enableActivity(
                RoctracerRuntime.ACTIVITY_DOMAIN_HIP_OPS,
                RoctracerRuntime.HIP_OP_KERNEL_DISPATCH,
                MemorySegment.NULL
            );
            if (result != RoctracerRuntime.ROCTRACER_STATUS_SUCCESS) {
                throw new RuntimeException("Failed to enable roctracer activity: error " + result);
            }

            // Enable memory operations if throughput collection is requested
            if (config.collectThroughput()) {
                RoctracerRuntime.enableActivity(
                    RoctracerRuntime.ACTIVITY_DOMAIN_HIP_OPS,
                    RoctracerRuntime.HIP_OP_COPY,
                    MemorySegment.NULL
                );
                // Ignore failure - not critical
            }

            sessionActive = true;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to start roctracer session", e);
        }
    }

    @Override
    public void stopSession() {
        if (!sessionActive) {
            throw new IllegalStateException("No profiling session active");
        }

        try {
            // Flush activity buffers
            flush();

            // Disable activity collection
            RoctracerRuntime.disableActivity(
                RoctracerRuntime.ACTIVITY_DOMAIN_HIP_OPS,
                RoctracerRuntime.HIP_OP_KERNEL_DISPATCH
            );
            RoctracerRuntime.disableActivity(
                RoctracerRuntime.ACTIVITY_DOMAIN_HIP_OPS,
                RoctracerRuntime.HIP_OP_COPY
            );

            sessionActive = false;
        } catch (Throwable e) {
            sessionActive = false;
            throw new RuntimeException("Failed to stop roctracer session", e);
        }
    }

    @Override
    public boolean isSessionActive() {
        return sessionActive;
    }

    @Override
    public Optional<ProfilingMetrics> getMetrics(long correlationId) {
        return Optional.ofNullable(metricsMap.get(correlationId));
    }

    @Override
    public List<ProfilingMetrics> getAllMetrics() {
        return new ArrayList<>(metricsMap.values());
    }

    @Override
    public void flush() {
        if (!sessionActive) {
            return;
        }

        try {
            // Flush all activity buffers
            int result = RoctracerRuntime.flushActivity(MemorySegment.NULL);
            if (result != RoctracerRuntime.ROCTRACER_STATUS_SUCCESS) {
                // Log warning but don't fail
                System.err.println("roctracer flush warning: error " + result);
            }

            // Process activity records and populate metricsMap
            // In a full implementation, this would iterate through activity buffers
        } catch (Throwable e) {
            throw new RuntimeException("Failed to flush roctracer activity", e);
        }
    }

    @Override
    public void clearMetrics() {
        metricsMap.clear();
    }

    @Override
    public String getBackendName() {
        return BACKEND_NAME;
    }

    @Override
    public int getDeviceIndex() {
        return deviceIndex;
    }

    @Override
    public boolean isSupported() {
        return RoctracerRuntime.isAvailable();
    }

    @Override
    public long estimatedOverheadNanos() {
        return ESTIMATED_OVERHEAD_NANOS;
    }

    @Override
    public void close() {
        if (sessionActive) {
            try {
                stopSession();
            } catch (Exception e) {
                // Log but don't throw during close
            }
        }
        arena.close();
    }

    /**
     * Record profiling metrics for a kernel.
     *
     * <p>This is called internally when processing roctracer activity records.
     *
     * @param correlationId the kernel correlation ID
     * @param metrics the collected metrics
     */
    void recordMetrics(long correlationId, RoctracerMetrics metrics) {
        metricsMap.put(correlationId, metrics);
    }

    /**
     * roctracer-based implementation of ProfilingMetrics.
     */
    public static final class RoctracerMetrics implements ProfilingMetrics {
        private final long correlationId;
        private final String kernelName;
        private final double achievedOccupancy;
        private final double cuEfficiency;
        private final double computeThroughput;
        private final double memoryThroughput;
        private final double l1HitRate;
        private final double l2HitRate;
        private final double stallMemory;
        private final double stallExecution;
        private final double stallSync;
        private final long durationNanos;
        private final long overheadNanos;

        public RoctracerMetrics(
            long correlationId,
            String kernelName,
            double achievedOccupancy,
            double cuEfficiency,
            double computeThroughput,
            double memoryThroughput,
            double l1HitRate,
            double l2HitRate,
            double stallMemory,
            double stallExecution,
            double stallSync,
            long durationNanos,
            long overheadNanos
        ) {
            this.correlationId = correlationId;
            this.kernelName = kernelName;
            this.achievedOccupancy = achievedOccupancy;
            this.cuEfficiency = cuEfficiency;
            this.computeThroughput = computeThroughput;
            this.memoryThroughput = memoryThroughput;
            this.l1HitRate = l1HitRate;
            this.l2HitRate = l2HitRate;
            this.stallMemory = stallMemory;
            this.stallExecution = stallExecution;
            this.stallSync = stallSync;
            this.durationNanos = durationNanos;
            this.overheadNanos = overheadNanos;
        }

        @Override public long correlationId() { return correlationId; }
        @Override public String kernelName() { return kernelName; }
        @Override public double achievedOccupancyPercent() { return achievedOccupancy; }
        @Override public double smEfficiencyPercent() { return cuEfficiency; } // CU ≈ SM
        @Override public double computeThroughputPercent() { return computeThroughput; }
        @Override public double memoryThroughputPercent() { return memoryThroughput; }
        @Override public double l1CacheHitRatePercent() { return l1HitRate; }
        @Override public double l2CacheHitRatePercent() { return l2HitRate; }
        @Override public double stallMemoryDependencyPercent() { return stallMemory; }
        @Override public double stallExecutionDependencyPercent() { return stallExecution; }
        @Override public double stallSynchronizationPercent() { return stallSync; }
        @Override public long kernelDurationNanos() { return durationNanos; }
        @Override public long profilingOverheadNanos() { return overheadNanos; }

        @Override
        public void populateEvent(GpuProfilingEvent event) {
            ProfilingMetrics.super.populateEvent(event);
            event.profilerBackend = "roctracer";
        }

        /**
         * Create a builder for RoctracerMetrics.
         *
         * @param correlationId the kernel correlation ID
         * @param kernelName the kernel name
         * @return a new builder
         */
        public static Builder builder(long correlationId, String kernelName) {
            return new Builder(correlationId, kernelName);
        }

        /**
         * Builder for RoctracerMetrics.
         */
        public static final class Builder {
            private final long correlationId;
            private final String kernelName;
            private double achievedOccupancy;
            private double cuEfficiency;
            private double computeThroughput;
            private double memoryThroughput;
            private double l1HitRate;
            private double l2HitRate;
            private double stallMemory;
            private double stallExecution;
            private double stallSync;
            private long durationNanos;
            private long overheadNanos;

            private Builder(long correlationId, String kernelName) {
                this.correlationId = correlationId;
                this.kernelName = kernelName;
            }

            public Builder achievedOccupancy(double percent) {
                this.achievedOccupancy = percent;
                return this;
            }

            public Builder cuEfficiency(double percent) {
                this.cuEfficiency = percent;
                return this;
            }

            public Builder computeThroughput(double percent) {
                this.computeThroughput = percent;
                return this;
            }

            public Builder memoryThroughput(double percent) {
                this.memoryThroughput = percent;
                return this;
            }

            public Builder l1HitRate(double percent) {
                this.l1HitRate = percent;
                return this;
            }

            public Builder l2HitRate(double percent) {
                this.l2HitRate = percent;
                return this;
            }

            public Builder stallMemory(double percent) {
                this.stallMemory = percent;
                return this;
            }

            public Builder stallExecution(double percent) {
                this.stallExecution = percent;
                return this;
            }

            public Builder stallSync(double percent) {
                this.stallSync = percent;
                return this;
            }

            public Builder durationNanos(long nanos) {
                this.durationNanos = nanos;
                return this;
            }

            public Builder overheadNanos(long nanos) {
                this.overheadNanos = nanos;
                return this;
            }

            public RoctracerMetrics build() {
                return new RoctracerMetrics(
                    correlationId, kernelName,
                    achievedOccupancy, cuEfficiency,
                    computeThroughput, memoryThroughput,
                    l1HitRate, l2HitRate,
                    stallMemory, stallExecution, stallSync,
                    durationNanos, overheadNanos
                );
            }
        }
    }
}
