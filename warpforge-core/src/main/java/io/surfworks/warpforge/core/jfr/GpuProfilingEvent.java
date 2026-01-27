package io.surfworks.warpforge.core.jfr;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Percentage;
import jdk.jfr.Timespan;

/**
 * JFR event for hardware-level GPU profiling metrics.
 *
 * <p>This event captures metrics obtained from hardware profiling APIs:
 * <ul>
 *   <li>NVIDIA: CUPTI (CUDA Profiling Tools Interface)</li>
 *   <li>AMD: roctracer / rocprofiler</li>
 * </ul>
 *
 * <p>Unlike {@link GpuKernelEvent} which captures theoretical occupancy estimates,
 * this event records actual achieved metrics from hardware counters. This enables:
 * <ul>
 *   <li>Identifying the real bottleneck (compute vs memory vs latency)</li>
 *   <li>Measuring actual vs theoretical occupancy</li>
 *   <li>Cache efficiency analysis</li>
 *   <li>Warp scheduling efficiency</li>
 * </ul>
 *
 * <p>Usage pattern:
 * <pre>{@code
 * // After kernel execution with profiling enabled
 * GpuProfilingEvent event = new GpuProfilingEvent();
 * event.kernelName = "gemm_nn_128x128";
 * event.correlationId = kernelEvent.correlationId;
 *
 * // From CUPTI/roctracer activity records
 * event.achievedOccupancyPercent = cuptiMetrics.getAchievedOccupancy();
 * event.smEfficiencyPercent = cuptiMetrics.getSmEfficiency();
 *
 * // Cache metrics
 * event.l1CacheHitRate = cuptiMetrics.getL1HitRate();
 * event.l2CacheHitRate = cuptiMetrics.getL2HitRate();
 *
 * event.commit();
 * }</pre>
 *
 * @see GpuKernelEvent
 * @see io.surfworks.warpforge.core.profiling.HardwareProfiler
 */
@Name("io.surfworks.warpforge.GpuProfiling")
@Label("GPU Hardware Profiling")
@Category({"WarpForge", "GPU", "Profiling"})
@Description("Hardware counter metrics from CUPTI/roctracer")
public class GpuProfilingEvent extends Event {

    // ==================== Identity ====================

    @Label("Kernel Name")
    @Description("Kernel function name from profiler")
    public String kernelName;

    @Label("Correlation ID")
    @Description("Correlation ID linking to GpuKernelEvent")
    public long correlationId;

    @Label("Device Index")
    @Description("GPU device index")
    public int deviceIndex;

    @Label("Stream ID")
    @Description("CUDA/HIP stream handle")
    public long streamId;

    // ==================== Achieved Occupancy ====================

    @Label("Achieved Occupancy %")
    @Description("Actual occupancy measured by hardware counters (0-100)")
    @Percentage
    public double achievedOccupancyPercent;

    @Label("Theoretical Occupancy %")
    @Description("Theoretical maximum occupancy for comparison")
    @Percentage
    public double theoreticalOccupancyPercent;

    @Label("Occupancy Efficiency %")
    @Description("Achieved / Theoretical occupancy ratio (0-100)")
    @Percentage
    public double occupancyEfficiencyPercent;

    // ==================== SM Efficiency ====================

    @Label("SM Efficiency %")
    @Description("Percentage of time at least one warp was active on an SM")
    @Percentage
    public double smEfficiencyPercent;

    @Label("Active Warps Per SM")
    @Description("Average active warps per SM during execution")
    public double activeWarpsPerSM;

    @Label("Eligible Warps Per SM")
    @Description("Average warps eligible to issue per cycle")
    public double eligibleWarpsPerSM;

    // ==================== Compute Utilization ====================

    @Label("Compute Throughput %")
    @Description("Achieved compute throughput as percentage of peak")
    @Percentage
    public double computeThroughputPercent;

    @Label("IPC (Instructions Per Cycle)")
    @Description("Average instructions executed per cycle per SM")
    public double instructionsPerCycle;

    @Label("Warp Execution Efficiency %")
    @Description("Average percentage of active threads per executed instruction")
    @Percentage
    public double warpExecutionEfficiencyPercent;

    // ==================== Memory Throughput ====================

    @Label("Memory Throughput %")
    @Description("Achieved memory throughput as percentage of peak")
    @Percentage
    public double memoryThroughputPercent;

    @Label("DRAM Throughput GB/s")
    @Description("Device memory (HBM/GDDR) throughput in GB/s")
    public double dramThroughputGBps;

    @Label("DRAM Read Throughput GB/s")
    @Description("Device memory read throughput in GB/s")
    public double dramReadThroughputGBps;

    @Label("DRAM Write Throughput GB/s")
    @Description("Device memory write throughput in GB/s")
    public double dramWriteThroughputGBps;

    // ==================== Cache Metrics ====================

    @Label("L1 Cache Hit Rate %")
    @Description("L1/TEX cache hit rate (0-100)")
    @Percentage
    public double l1CacheHitRatePercent;

    @Label("L2 Cache Hit Rate %")
    @Description("L2 cache hit rate (0-100)")
    @Percentage
    public double l2CacheHitRatePercent;

    @Label("L1 Cache Throughput GB/s")
    @Description("L1 cache throughput in GB/s")
    public double l1CacheThroughputGBps;

    @Label("L2 Cache Throughput GB/s")
    @Description("L2 cache throughput in GB/s")
    public double l2CacheThroughputGBps;

    @Label("Shared Memory Throughput GB/s")
    @Description("Shared memory (LDS) throughput in GB/s")
    public double sharedMemoryThroughputGBps;

    // ==================== Warp Stall Analysis ====================

    @Label("Stall - Memory Dependency %")
    @Description("Percentage of stalls due to memory dependencies")
    @Percentage
    public double stallMemoryDependencyPercent;

    @Label("Stall - Execution Dependency %")
    @Description("Percentage of stalls due to execution dependencies")
    @Percentage
    public double stallExecutionDependencyPercent;

    @Label("Stall - Synchronization %")
    @Description("Percentage of stalls due to __syncthreads or barriers")
    @Percentage
    public double stallSynchronizationPercent;

    @Label("Stall - Texture %")
    @Description("Percentage of stalls due to texture memory operations")
    @Percentage
    public double stallTexturePercent;

    @Label("Stall - Instruction Fetch %")
    @Description("Percentage of stalls due to instruction cache misses")
    @Percentage
    public double stallInstructionFetchPercent;

    @Label("Stall - Other %")
    @Description("Percentage of stalls due to other reasons")
    @Percentage
    public double stallOtherPercent;

    // ==================== Timing ====================

    @Label("Kernel Duration")
    @Description("Kernel execution time from hardware timestamps")
    @Timespan(Timespan.NANOSECONDS)
    public long kernelDurationNanos;

    @Label("GPU Start Timestamp")
    @Description("GPU clock timestamp when kernel started")
    public long gpuStartTimestamp;

    @Label("GPU End Timestamp")
    @Description("GPU clock timestamp when kernel completed")
    public long gpuEndTimestamp;

    // ==================== Profiler Metadata ====================

    @Label("Profiler Backend")
    @Description("Profiling backend used: CUPTI or roctracer")
    public String profilerBackend;

    @Label("Profiling Overhead Nanos")
    @Description("Estimated overhead introduced by profiling")
    @Timespan(Timespan.NANOSECONDS)
    public long profilingOverheadNanos;

    /**
     * Compute derived efficiency metrics.
     *
     * <p>Call this after setting achievedOccupancyPercent and theoreticalOccupancyPercent.
     */
    public void computeDerivedMetrics() {
        if (theoreticalOccupancyPercent > 0) {
            this.occupancyEfficiencyPercent =
                (achievedOccupancyPercent / theoreticalOccupancyPercent) * 100.0;
        }
    }

    /**
     * Determine the primary bottleneck based on stall analysis.
     *
     * @return the name of the primary bottleneck category
     */
    public String getPrimaryBottleneck() {
        double maxStall = 0;
        String bottleneck = "unknown";

        if (stallMemoryDependencyPercent > maxStall) {
            maxStall = stallMemoryDependencyPercent;
            bottleneck = "memory_dependency";
        }
        if (stallExecutionDependencyPercent > maxStall) {
            maxStall = stallExecutionDependencyPercent;
            bottleneck = "execution_dependency";
        }
        if (stallSynchronizationPercent > maxStall) {
            maxStall = stallSynchronizationPercent;
            bottleneck = "synchronization";
        }
        if (stallTexturePercent > maxStall) {
            maxStall = stallTexturePercent;
            bottleneck = "texture";
        }
        if (stallInstructionFetchPercent > maxStall) {
            maxStall = stallInstructionFetchPercent;
            bottleneck = "instruction_fetch";
        }

        // If compute throughput is high and stalls are low, it's compute-bound
        if (maxStall < 10 && computeThroughputPercent > 80) {
            return "compute_bound";
        }

        // If memory throughput is high, it's memory-bound
        if (memoryThroughputPercent > 80) {
            return "memory_bound";
        }

        return bottleneck;
    }

    /**
     * Check if the kernel is memory-bound based on metrics.
     *
     * @return true if the kernel appears to be memory-bound
     */
    public boolean isMemoryBound() {
        return memoryThroughputPercent > computeThroughputPercent
            && stallMemoryDependencyPercent > 20;
    }

    /**
     * Check if the kernel is compute-bound based on metrics.
     *
     * @return true if the kernel appears to be compute-bound
     */
    public boolean isComputeBound() {
        return computeThroughputPercent > memoryThroughputPercent
            && stallExecutionDependencyPercent > stallMemoryDependencyPercent;
    }
}
