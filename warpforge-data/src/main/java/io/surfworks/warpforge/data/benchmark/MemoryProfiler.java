package io.surfworks.warpforge.data.benchmark;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Memory profiler for tracking heap and native memory usage during benchmarks.
 *
 * <p>Tracks:
 * <ul>
 *   <li>Heap memory usage (used, committed, max)</li>
 *   <li>Peak memory usage during measurement</li>
 *   <li>Memory allocation rate</li>
 *   <li>GC pause time (if available)</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * MemoryProfiler profiler = new MemoryProfiler();
 * profiler.start();
 *
 * // Run benchmark code
 * runBenchmark();
 *
 * MemorySnapshot snapshot = profiler.stop();
 * System.out.println(snapshot.summary());
 * }</pre>
 */
public final class MemoryProfiler {

    private static final MemoryMXBean MEMORY_MX_BEAN = ManagementFactory.getMemoryMXBean();

    private volatile boolean running = false;
    private Thread samplerThread;
    private final List<MemorySample> samples = new ArrayList<>();
    private final AtomicLong allocationCount = new AtomicLong(0);

    private long startTimeNanos;
    private long startHeapUsed;
    private long startNonHeapUsed;

    /**
     * Start memory profiling.
     */
    public void start() {
        if (running) {
            throw new IllegalStateException("Profiler is already running");
        }

        samples.clear();
        allocationCount.set(0);
        running = true;

        // Record starting state
        forceGC();
        MemoryUsage heapUsage = MEMORY_MX_BEAN.getHeapMemoryUsage();
        MemoryUsage nonHeapUsage = MEMORY_MX_BEAN.getNonHeapMemoryUsage();
        startHeapUsed = heapUsage.getUsed();
        startNonHeapUsed = nonHeapUsage.getUsed();
        startTimeNanos = System.nanoTime();

        // Start background sampler
        samplerThread = new Thread(this::sampleLoop, "memory-profiler");
        samplerThread.setDaemon(true);
        samplerThread.start();
    }

    /**
     * Stop profiling and return the collected snapshot.
     */
    public MemorySnapshot stop() {
        if (!running) {
            throw new IllegalStateException("Profiler is not running");
        }

        running = false;
        try {
            samplerThread.join(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        long durationNanos = System.nanoTime() - startTimeNanos;

        // Final sample
        MemoryUsage heapUsage = MEMORY_MX_BEAN.getHeapMemoryUsage();
        MemoryUsage nonHeapUsage = MEMORY_MX_BEAN.getNonHeapMemoryUsage();

        // Calculate statistics
        long peakHeapUsed = samples.stream().mapToLong(s -> s.heapUsed).max().orElse(heapUsage.getUsed());
        long peakNonHeapUsed = samples.stream().mapToLong(s -> s.nonHeapUsed).max().orElse(nonHeapUsage.getUsed());

        long avgHeapUsed = (long) samples.stream().mapToLong(s -> s.heapUsed).average().orElse(heapUsage.getUsed());

        long heapAllocated = heapUsage.getUsed() - startHeapUsed;
        long nonHeapAllocated = nonHeapUsage.getUsed() - startNonHeapUsed;

        return new MemorySnapshot(
                startHeapUsed,
                heapUsage.getUsed(),
                peakHeapUsed,
                avgHeapUsed,
                heapUsage.getCommitted(),
                heapUsage.getMax(),
                startNonHeapUsed,
                nonHeapUsage.getUsed(),
                peakNonHeapUsed,
                nonHeapUsage.getCommitted(),
                Math.max(0, heapAllocated),
                Math.max(0, nonHeapAllocated),
                samples.size(),
                durationNanos
        );
    }

    /**
     * Record an allocation (for tracking allocation count).
     */
    public void recordAllocation() {
        allocationCount.incrementAndGet();
    }

    /**
     * Take a single memory sample (for manual sampling).
     */
    public MemorySample sample() {
        MemoryUsage heapUsage = MEMORY_MX_BEAN.getHeapMemoryUsage();
        MemoryUsage nonHeapUsage = MEMORY_MX_BEAN.getNonHeapMemoryUsage();
        return new MemorySample(
                System.nanoTime() - startTimeNanos,
                heapUsage.getUsed(),
                heapUsage.getCommitted(),
                nonHeapUsage.getUsed(),
                nonHeapUsage.getCommitted()
        );
    }

    /**
     * Get current heap usage in bytes.
     */
    public static long currentHeapUsed() {
        return MEMORY_MX_BEAN.getHeapMemoryUsage().getUsed();
    }

    /**
     * Get current non-heap usage in bytes.
     */
    public static long currentNonHeapUsed() {
        return MEMORY_MX_BEAN.getNonHeapMemoryUsage().getUsed();
    }

    /**
     * Get maximum heap size in bytes.
     */
    public static long maxHeap() {
        return MEMORY_MX_BEAN.getHeapMemoryUsage().getMax();
    }

    /**
     * Force garbage collection (best effort).
     */
    public static void forceGC() {
        System.gc();
        try {
            Thread.sleep(50);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        System.gc();
    }

    /**
     * Measure memory usage of a runnable.
     */
    public static MemorySnapshot measure(Runnable task) {
        MemoryProfiler profiler = new MemoryProfiler();
        profiler.start();
        try {
            task.run();
        } finally {
            return profiler.stop();
        }
    }

    /**
     * Create a quick snapshot without continuous sampling.
     */
    public static MemorySnapshot quickSnapshot() {
        MemoryUsage heapUsage = MEMORY_MX_BEAN.getHeapMemoryUsage();
        MemoryUsage nonHeapUsage = MEMORY_MX_BEAN.getNonHeapMemoryUsage();
        return new MemorySnapshot(
                heapUsage.getUsed(), heapUsage.getUsed(), heapUsage.getUsed(), heapUsage.getUsed(),
                heapUsage.getCommitted(), heapUsage.getMax(),
                nonHeapUsage.getUsed(), nonHeapUsage.getUsed(), nonHeapUsage.getUsed(),
                nonHeapUsage.getCommitted(),
                0, 0, 1, 0
        );
    }

    private void sampleLoop() {
        while (running) {
            try {
                MemorySample memorySample = sample();
                synchronized (samples) {
                    samples.add(memorySample);
                }
                Thread.sleep(10); // Sample every 10ms
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    /**
     * A single memory sample at a point in time.
     */
    public record MemorySample(
            long timestampNanos,
            long heapUsed,
            long heapCommitted,
            long nonHeapUsed,
            long nonHeapCommitted
    ) {}

    /**
     * Aggregated memory statistics from a profiling session.
     */
    public record MemorySnapshot(
            long startHeapUsed,
            long endHeapUsed,
            long peakHeapUsed,
            long avgHeapUsed,
            long heapCommitted,
            long heapMax,
            long startNonHeapUsed,
            long endNonHeapUsed,
            long peakNonHeapUsed,
            long nonHeapCommitted,
            long heapAllocated,
            long nonHeapAllocated,
            int sampleCount,
            long durationNanos
    ) {
        /**
         * Total peak memory (heap + non-heap).
         */
        public long totalPeakUsed() {
            return peakHeapUsed + peakNonHeapUsed;
        }

        /**
         * Total memory allocated during the session.
         */
        public long totalAllocated() {
            return heapAllocated + nonHeapAllocated;
        }

        /**
         * Heap utilization as a percentage of max.
         */
        public double heapUtilization() {
            if (heapMax <= 0) return 0;
            return (double) peakHeapUsed / heapMax * 100;
        }

        /**
         * Allocation rate in bytes per second.
         */
        public double allocationRateBytesPerSec() {
            if (durationNanos <= 0) return 0;
            return (double) heapAllocated / durationNanos * 1_000_000_000;
        }

        /**
         * Convert to BenchmarkResult.MemoryStats.
         */
        public BenchmarkResult.MemoryStats toMemoryStats() {
            return new BenchmarkResult.MemoryStats(
                    peakHeapUsed,
                    heapAllocated,
                    Math.max(0, startHeapUsed - endHeapUsed),
                    sampleCount
            );
        }

        /**
         * Generate a human-readable summary.
         */
        public String summary() {
            StringBuilder sb = new StringBuilder();
            sb.append("Memory Profile:\n");
            sb.append(String.format("  Heap:     peak=%s, avg=%s, committed=%s, max=%s%n",
                    formatBytes(peakHeapUsed), formatBytes(avgHeapUsed),
                    formatBytes(heapCommitted), formatBytes(heapMax)));
            sb.append(String.format("  Non-Heap: peak=%s, committed=%s%n",
                    formatBytes(peakNonHeapUsed), formatBytes(nonHeapCommitted)));
            sb.append(String.format("  Allocated: heap=%s, non-heap=%s%n",
                    formatBytes(heapAllocated), formatBytes(nonHeapAllocated)));
            sb.append(String.format("  Utilization: %.1f%% of max heap%n", heapUtilization()));
            if (durationNanos > 0) {
                sb.append(String.format("  Allocation rate: %s/sec%n",
                        formatBytes((long) allocationRateBytesPerSec())));
            }
            sb.append(String.format("  Samples: %d over %.1f ms%n",
                    sampleCount, durationNanos / 1_000_000.0));
            return sb.toString();
        }

        /**
         * Convert to a map for JSON serialization.
         */
        public java.util.Map<String, Object> toMap() {
            return java.util.Map.ofEntries(
                    java.util.Map.entry("peak_heap_bytes", peakHeapUsed),
                    java.util.Map.entry("avg_heap_bytes", avgHeapUsed),
                    java.util.Map.entry("heap_committed_bytes", heapCommitted),
                    java.util.Map.entry("heap_max_bytes", heapMax),
                    java.util.Map.entry("peak_non_heap_bytes", peakNonHeapUsed),
                    java.util.Map.entry("non_heap_committed_bytes", nonHeapCommitted),
                    java.util.Map.entry("heap_allocated_bytes", heapAllocated),
                    java.util.Map.entry("non_heap_allocated_bytes", nonHeapAllocated),
                    java.util.Map.entry("heap_utilization_percent", heapUtilization()),
                    java.util.Map.entry("allocation_rate_bytes_per_sec", allocationRateBytesPerSec()),
                    java.util.Map.entry("sample_count", sampleCount),
                    java.util.Map.entry("duration_ms", durationNanos / 1_000_000.0)
            );
        }

        private static String formatBytes(long bytes) {
            if (bytes < 0) return "N/A";
            if (bytes < 1024) return bytes + " B";
            if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
            if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
            return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
        }
    }
}
