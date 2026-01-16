package io.surfworks.warpforge.launch.scheduler;

import io.surfworks.warpforge.launch.job.GpuType;

import java.time.Duration;
import java.util.Objects;
import java.util.Set;

/**
 * Capabilities of a scheduler implementation.
 *
 * @param supportsGpuScheduling Whether the scheduler can handle GPU resource requests
 * @param supportedGpuTypes     Set of GPU types this scheduler can provision
 * @param supportsQueuePriority Whether the scheduler supports job priority
 * @param supportsNodeAffinity  Whether the scheduler supports node affinity constraints
 * @param supportsJobArrays     Whether the scheduler supports submitting job arrays
 * @param maxConcurrentJobs     Maximum number of concurrent jobs (0 = unlimited)
 * @param maxJobDuration        Maximum allowed job duration (null = unlimited)
 */
public record SchedulerCapabilities(
        boolean supportsGpuScheduling,
        Set<GpuType> supportedGpuTypes,
        boolean supportsQueuePriority,
        boolean supportsNodeAffinity,
        boolean supportsJobArrays,
        int maxConcurrentJobs,
        Duration maxJobDuration
) {

    public SchedulerCapabilities {
        Objects.requireNonNull(supportedGpuTypes, "supportedGpuTypes cannot be null");
        supportedGpuTypes = Set.copyOf(supportedGpuTypes);
        if (maxConcurrentJobs < 0) {
            throw new IllegalArgumentException("maxConcurrentJobs cannot be negative");
        }
    }

    /**
     * Creates basic capabilities for a simple scheduler.
     */
    public static SchedulerCapabilities basic() {
        return new SchedulerCapabilities(
                false,
                Set.of(),
                false,
                false,
                false,
                0,
                null
        );
    }

    /**
     * Creates capabilities for a local scheduler with limited concurrency.
     */
    public static SchedulerCapabilities local(int maxConcurrent) {
        return new SchedulerCapabilities(
                false,
                Set.of(),
                false,
                false,
                false,
                maxConcurrent,
                null
        );
    }

    /**
     * Creates capabilities for a full-featured cluster scheduler.
     */
    public static SchedulerCapabilities fullFeatured(Set<GpuType> gpuTypes) {
        return new SchedulerCapabilities(
                true,
                gpuTypes,
                true,
                true,
                true,
                0,
                null
        );
    }

    /**
     * Returns true if this scheduler supports the given GPU type.
     */
    public boolean supportsGpuType(GpuType gpuType) {
        if (gpuType == GpuType.NONE) return true;
        if (gpuType == GpuType.ANY) return !supportedGpuTypes.isEmpty();
        return supportedGpuTypes.contains(gpuType);
    }
}
