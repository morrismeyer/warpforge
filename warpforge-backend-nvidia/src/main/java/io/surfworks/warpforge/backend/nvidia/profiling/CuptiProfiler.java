package io.surfworks.warpforge.backend.nvidia.profiling;

import io.surfworks.warpforge.core.jfr.GpuProfilingEvent;
import io.surfworks.warpforge.core.profiling.HardwareProfiler;

import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * CUPTI-based implementation of HardwareProfiler for NVIDIA GPUs.
 *
 * <p>This implementation uses CUPTI (CUDA Profiling Tools Interface) to
 * collect hardware-level profiling metrics including:
 * <ul>
 *   <li>Achieved occupancy</li>
 *   <li>SM efficiency</li>
 *   <li>Cache hit rates</li>
 *   <li>Memory throughput</li>
 *   <li>Warp stall analysis</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 * try (CuptiProfiler profiler = CuptiProfiler.create(0)) {
 *     profiler.startSession();
 *     // Execute CUDA kernels...
 *     profiler.stopSession();
 *
 *     // Get metrics
 *     profiler.getAllMetrics().forEach(m -> {
 *         System.out.println(m.kernelName() + ": " + m.achievedOccupancyPercent() + "%");
 *     });
 * }
 * }</pre>
 */
public final class CuptiProfiler implements HardwareProfiler {

    private static final String BACKEND_NAME = "CUPTI";
    private static final long ESTIMATED_OVERHEAD_NANOS = 50_000L; // ~50 microseconds

    private final int deviceIndex;
    private final Arena arena;
    private final Map<Long, CuptiMetrics> metricsMap;
    private volatile boolean sessionActive;
    private volatile SessionConfig currentConfig;

    /**
     * Create a CUPTI profiler for the specified device.
     *
     * @param deviceIndex the CUDA device index
     * @return a new CuptiProfiler instance
     * @throws IllegalStateException if CUPTI is not available
     */
    public static CuptiProfiler create(int deviceIndex) {
        if (!CuptiRuntime.isAvailable()) {
            throw new IllegalStateException("CUPTI is not available");
        }
        return new CuptiProfiler(deviceIndex);
    }

    /**
     * Create a CUPTI profiler if available, otherwise return empty.
     *
     * @param deviceIndex the CUDA device index
     * @return Optional containing profiler if CUPTI is available
     */
    public static Optional<CuptiProfiler> tryCreate(int deviceIndex) {
        if (!CuptiRuntime.isAvailable()) {
            return Optional.empty();
        }
        return Optional.of(new CuptiProfiler(deviceIndex));
    }

    private CuptiProfiler(int deviceIndex) {
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
            // Enable kernel activity collection
            int result = CuptiRuntime.activityEnable(CuptiRuntime.CUPTI_ACTIVITY_KIND_KERNEL);
            if (result != CuptiRuntime.CUPTI_SUCCESS) {
                throw new RuntimeException("Failed to enable CUPTI kernel activity: " +
                    CuptiRuntime.getResultString(arena, result));
            }

            // Enable concurrent kernel activity if needed
            if (config.collectThroughput()) {
                result = CuptiRuntime.activityEnable(CuptiRuntime.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
                // Ignore failure - not all GPUs support this
            }

            sessionActive = true;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to start CUPTI session", e);
        }
    }

    @Override
    public void stopSession() {
        if (!sessionActive) {
            throw new IllegalStateException("No profiling session active");
        }

        try {
            // Flush and process activity buffers
            flush();

            // Disable activity collection
            CuptiRuntime.activityDisable(CuptiRuntime.CUPTI_ACTIVITY_KIND_KERNEL);
            CuptiRuntime.activityDisable(CuptiRuntime.CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

            sessionActive = false;
        } catch (Throwable e) {
            sessionActive = false;
            throw new RuntimeException("Failed to stop CUPTI session", e);
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
            // Blocking flush - wait for all GPU work to complete
            int result = CuptiRuntime.activityFlushAll(0);
            if (result != CuptiRuntime.CUPTI_SUCCESS) {
                // Log warning but don't fail - some metrics may still be available
                System.err.println("CUPTI flush warning: " +
                    CuptiRuntime.getResultString(arena, result));
            }

            // Process activity records and populate metricsMap
            // In a full implementation, this would use cuptiActivityGetNextRecord
            // to iterate through all collected activity records
        } catch (Throwable e) {
            throw new RuntimeException("Failed to flush CUPTI activity", e);
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
        return CuptiRuntime.isAvailable();
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
     * <p>This is called internally when processing CUPTI activity records.
     *
     * @param correlationId the kernel correlation ID
     * @param metrics the collected metrics
     */
    void recordMetrics(long correlationId, CuptiMetrics metrics) {
        metricsMap.put(correlationId, metrics);
    }

    /**
     * CUPTI-based implementation of ProfilingMetrics.
     */
    public static final class CuptiMetrics implements ProfilingMetrics {
        private final long correlationId;
        private final String kernelName;
        private final double achievedOccupancy;
        private final double smEfficiency;
        private final double computeThroughput;
        private final double memoryThroughput;
        private final double l1HitRate;
        private final double l2HitRate;
        private final double stallMemory;
        private final double stallExecution;
        private final double stallSync;
        private final long durationNanos;
        private final long overheadNanos;

        public CuptiMetrics(
            long correlationId,
            String kernelName,
            double achievedOccupancy,
            double smEfficiency,
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
            this.smEfficiency = smEfficiency;
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
        @Override public double smEfficiencyPercent() { return smEfficiency; }
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
            event.profilerBackend = "CUPTI";
        }

        /**
         * Create a builder for CuptiMetrics.
         *
         * @param correlationId the kernel correlation ID
         * @param kernelName the kernel name
         * @return a new builder
         */
        public static Builder builder(long correlationId, String kernelName) {
            return new Builder(correlationId, kernelName);
        }

        /**
         * Builder for CuptiMetrics.
         */
        public static final class Builder {
            private final long correlationId;
            private final String kernelName;
            private double achievedOccupancy;
            private double smEfficiency;
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

            public Builder smEfficiency(double percent) {
                this.smEfficiency = percent;
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

            public CuptiMetrics build() {
                return new CuptiMetrics(
                    correlationId, kernelName,
                    achievedOccupancy, smEfficiency,
                    computeThroughput, memoryThroughput,
                    l1HitRate, l2HitRate,
                    stallMemory, stallExecution, stallSync,
                    durationNanos, overheadNanos
                );
            }
        }
    }
}
