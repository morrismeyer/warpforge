package io.surfworks.warpforge.core.kernel;

import io.surfworks.warpforge.core.jfr.GpuKernelEvent;
import io.surfworks.warpforge.core.jfr.GpuOccupancyEvent;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Tracks active GPU streams and kernel executions for concurrency monitoring.
 *
 * <p>This class maintains a view of currently active GPU work across all streams,
 * enabling:
 * <ul>
 *   <li>Concurrent kernel counting for JFR events</li>
 *   <li>Stream utilization monitoring</li>
 *   <li>Periodic occupancy snapshots</li>
 *   <li>Detecting GPU saturation</li>
 * </ul>
 *
 * <p>Usage pattern:
 * <pre>{@code
 * StreamTracker tracker = StreamTracker.forDevice(0);
 *
 * // When launching a kernel
 * long kernelId = tracker.kernelStarted(streamId, occupancyInfo);
 *
 * // Populate JFR event with concurrency info
 * tracker.populateEvent(event, streamId);
 *
 * // When kernel completes
 * tracker.kernelCompleted(kernelId);
 *
 * // For periodic snapshots
 * tracker.populateOccupancyEvent(occupancyEvent);
 * }</pre>
 *
 * <p>Thread safety: All methods are thread-safe and can be called concurrently
 * from multiple virtual threads.
 *
 * @see GpuKernelEvent
 * @see GpuOccupancyEvent
 */
public final class StreamTracker {

    private final int deviceIndex;
    private final AtomicLong nextKernelId = new AtomicLong(0);

    // Active kernels: kernelId -> KernelInfo
    private final Map<Long, KernelInfo> activeKernels = new ConcurrentHashMap<>();

    // Active streams: streamId -> set of kernelIds
    private final Map<Long, Set<Long>> streamToKernels = new ConcurrentHashMap<>();

    // Aggregate metrics
    private final AtomicLong totalActiveWarps = new AtomicLong(0);
    private final AtomicInteger totalActiveSMs = new AtomicInteger(0);

    // Device capabilities (set via configure())
    private volatile int smCount = 0;
    private volatile int maxWarpsPerSM = 64;

    /**
     * Information about an active kernel.
     */
    public record KernelInfo(
        long kernelId,
        long streamId,
        long startTimeNanos,
        int estimatedWarps,
        int estimatedSMs,
        int occupancyPercent,
        String operation
    ) {}

    private StreamTracker(int deviceIndex) {
        this.deviceIndex = deviceIndex;
    }

    /**
     * Create a tracker for a specific GPU device.
     *
     * @param deviceIndex the GPU device index
     * @return a new StreamTracker instance
     */
    public static StreamTracker forDevice(int deviceIndex) {
        return new StreamTracker(deviceIndex);
    }

    /**
     * Configure device capabilities for accurate occupancy estimation.
     *
     * @param smCount number of SMs on the device
     * @param maxWarpsPerSM maximum warps per SM
     * @return this tracker for chaining
     */
    public StreamTracker configure(int smCount, int maxWarpsPerSM) {
        this.smCount = smCount;
        this.maxWarpsPerSM = maxWarpsPerSM;
        return this;
    }

    /**
     * Record that a kernel has started execution.
     *
     * @param streamId the stream the kernel is executing on
     * @param occupancyInfo occupancy information for the kernel (may be null)
     * @param operation operation name for debugging
     * @return a unique kernel ID for tracking completion
     */
    public long kernelStarted(long streamId, OccupancyCalculator.OccupancyInfo occupancyInfo, String operation) {
        long kernelId = nextKernelId.incrementAndGet();

        int warps = occupancyInfo != null ? occupancyInfo.activeWarpsPerSM() : 0;
        int sms = 1; // Conservative estimate
        int occupancy = occupancyInfo != null ? occupancyInfo.occupancyPercent() : 0;

        KernelInfo info = new KernelInfo(
            kernelId,
            streamId,
            System.nanoTime(),
            warps,
            sms,
            occupancy,
            operation
        );

        activeKernels.put(kernelId, info);
        streamToKernels.computeIfAbsent(streamId, k -> ConcurrentHashMap.newKeySet()).add(kernelId);

        totalActiveWarps.addAndGet(warps);
        totalActiveSMs.incrementAndGet();

        return kernelId;
    }

    /**
     * Record that a kernel has started execution (without occupancy info).
     *
     * @param streamId the stream the kernel is executing on
     * @return a unique kernel ID for tracking completion
     */
    public long kernelStarted(long streamId) {
        return kernelStarted(streamId, null, null);
    }

    /**
     * Record that a kernel has completed execution.
     *
     * @param kernelId the kernel ID returned by kernelStarted()
     */
    public void kernelCompleted(long kernelId) {
        KernelInfo info = activeKernels.remove(kernelId);
        if (info != null) {
            Set<Long> kernelsOnStream = streamToKernels.get(info.streamId());
            if (kernelsOnStream != null) {
                kernelsOnStream.remove(kernelId);
                if (kernelsOnStream.isEmpty()) {
                    streamToKernels.remove(info.streamId());
                }
            }

            totalActiveWarps.addAndGet(-info.estimatedWarps());
            totalActiveSMs.decrementAndGet();
        }
    }

    /**
     * Get the number of currently active kernels.
     *
     * @return count of active kernels across all streams
     */
    public int activeKernelCount() {
        return activeKernels.size();
    }

    /**
     * Get the number of currently active streams.
     *
     * @return count of streams with active kernels
     */
    public int activeStreamCount() {
        return streamToKernels.size();
    }

    /**
     * Get the number of concurrent kernels on other streams.
     *
     * <p>This is the count of kernels NOT on the specified stream,
     * used for the concurrentKernels field in GpuKernelEvent.
     *
     * @param excludeStreamId stream to exclude from the count
     * @return count of kernels on other streams
     */
    public int concurrentKernelsExcluding(long excludeStreamId) {
        Set<Long> kernelsOnStream = streamToKernels.get(excludeStreamId);
        int kernelsOnThisStream = kernelsOnStream != null ? kernelsOnStream.size() : 0;
        return activeKernels.size() - kernelsOnThisStream;
    }

    /**
     * Get the estimated total active warps across all kernels.
     *
     * @return estimated total active warps
     */
    public long estimatedTotalActiveWarps() {
        return Math.max(0, totalActiveWarps.get());
    }

    /**
     * Get the estimated total occupancy percentage.
     *
     * @return estimated occupancy (0-100)
     */
    public int estimatedTotalOccupancy() {
        if (smCount == 0 || maxWarpsPerSM == 0) {
            return 0;
        }
        long maxWarps = (long) smCount * maxWarpsPerSM;
        long activeWarps = estimatedTotalActiveWarps();
        return (int) Math.min(100, (100L * activeWarps) / maxWarps);
    }

    /**
     * Get the device index this tracker is monitoring.
     *
     * @return the device index
     */
    public int deviceIndex() {
        return deviceIndex;
    }

    /**
     * Populate a GpuKernelEvent with concurrency information.
     *
     * @param event the event to populate
     * @param streamId the stream for this kernel
     */
    public void populateEvent(GpuKernelEvent event, long streamId) {
        event.streamId = streamId;
        event.concurrentKernels = concurrentKernelsExcluding(streamId);
    }

    /**
     * Populate a GpuOccupancyEvent with current state.
     *
     * @param event the event to populate
     */
    public void populateOccupancyEvent(GpuOccupancyEvent event) {
        event.deviceIndex = deviceIndex;
        event.activeStreams = activeStreamCount();
        event.activeKernels = activeKernelCount();
        event.activeWarpsEstimate = estimatedTotalActiveWarps();
        event.estimatedTotalOccupancyPercent = estimatedTotalOccupancy();
        event.smCount = smCount;
        event.maxWarpsPerSM = maxWarpsPerSM;
        event.estimatedActiveSMs = Math.min(totalActiveSMs.get(), smCount);
    }

    /**
     * Check if the GPU appears saturated (high concurrent kernel count).
     *
     * @param threshold minimum kernel count to consider saturated
     * @return true if active kernel count exceeds threshold
     */
    public boolean isSaturated(int threshold) {
        return activeKernelCount() >= threshold;
    }

    /**
     * Check if any kernels are currently executing.
     *
     * @return true if there are active kernels
     */
    public boolean hasActiveWork() {
        return !activeKernels.isEmpty();
    }

    /**
     * Get information about all active kernels.
     *
     * @return map of kernel ID to kernel info
     */
    public Map<Long, KernelInfo> getActiveKernels() {
        return Map.copyOf(activeKernels);
    }

    /**
     * Reset all tracking state.
     *
     * <p>Use with caution - typically only for testing or recovery.
     */
    public void reset() {
        activeKernels.clear();
        streamToKernels.clear();
        totalActiveWarps.set(0);
        totalActiveSMs.set(0);
    }

    @Override
    public String toString() {
        return String.format("StreamTracker[device=%d, streams=%d, kernels=%d, occupancy=%d%%]",
            deviceIndex, activeStreamCount(), activeKernelCount(), estimatedTotalOccupancy());
    }
}
