package io.surfworks.warpforge.core.jfr;

import jdk.jfr.Category;
import jdk.jfr.DataAmount;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Timespan;

/**
 * JFR event recording a single GPU kernel execution.
 *
 * <p>This event captures timing and throughput metrics for both vendor library calls
 * (cuBLAS, rocBLAS) and generated kernels (PTX, HIP). It enables unified profiling
 * across all execution tiers.
 *
 * <p>Usage:
 * <pre>{@code
 * GpuKernelEvent event = new GpuKernelEvent();
 * event.operation = "GEMM";
 * event.shape = "4096x4096 * 4096x4096";
 * event.gpuTimeMicros = elapsedMicros;
 * event.teraflops = (2.0 * M * N * K) / (elapsedMs * 1e9);
 * event.backend = "cuBLAS";
 * event.tier = "PRODUCTION";
 * event.deviceIndex = 0;
 * event.commit();
 * }</pre>
 */
@Name("io.surfworks.warpforge.GpuKernel")
@Label("GPU Kernel Execution")
@Category({"WarpForge", "GPU"})
@Description("Records GPU kernel execution time and throughput")
public class GpuKernelEvent extends Event {

    @Label("Operation")
    @Description("The StableHLO operation name (e.g., GEMM, Add, Convolution)")
    public String operation;

    @Label("Shape")
    @Description("Tensor shape description (e.g., '4096x4096 * 4096x4096')")
    public String shape;

    @Label("GPU Time")
    @Description("Kernel execution time in microseconds (from CUDA/HIP Events)")
    @Timespan(Timespan.MICROSECONDS)
    public long gpuTimeMicros;

    @Label("Throughput (TFLOPS)")
    @Description("Throughput in teraflops (for compute-bound operations)")
    public double teraflops;

    @Label("Backend")
    @Description("Backend implementation (cuBLAS, PTX, rocBLAS, HIP, MIOpen)")
    public String backend;

    @Label("Tier")
    @Description("Execution tier (PRODUCTION, OPTIMIZED_OBSERVABLE, CORRECTNESS)")
    public String tier;

    @Label("Device Index")
    @Description("GPU device index")
    public int deviceIndex;

    @Label("Bytes Transferred")
    @Description("Total bytes read and written by the kernel")
    @DataAmount
    public long bytesTransferred;

    @Label("Memory Bandwidth (GB/s)")
    @Description("Effective memory bandwidth in GB/s")
    public double memoryBandwidthGBps;

    // ==================== Launch Configuration ====================
    // See architecture/CPU-GPU-VISIBILITY.md for the full vision

    @Label("Grid Dim X")
    @Description("Number of blocks in X dimension")
    public int gridDimX;

    @Label("Grid Dim Y")
    @Description("Number of blocks in Y dimension")
    public int gridDimY;

    @Label("Grid Dim Z")
    @Description("Number of blocks in Z dimension")
    public int gridDimZ;

    @Label("Block Dim X")
    @Description("Threads per block in X dimension")
    public int blockDimX;

    @Label("Block Dim Y")
    @Description("Threads per block in Y dimension")
    public int blockDimY;

    @Label("Block Dim Z")
    @Description("Threads per block in Z dimension")
    public int blockDimZ;

    @Label("Total Threads")
    @Description("Total GPU threads (gridDim * blockDim)")
    public long totalThreads;

    @Label("Total Warps")
    @Description("Total warps (totalThreads / warpSize)")
    public long totalWarps;

    @Label("Total Blocks")
    @Description("Total blocks (gridDimX * gridDimY * gridDimZ)")
    public int totalBlocks;

    // ==================== Stream Context ====================

    @Label("Stream ID")
    @Description("CUDA/HIP stream handle")
    public long streamId;

    // ==================== Java Context ====================

    @Label("Virtual Thread ID")
    @Description("Java virtual thread ID (Thread.currentThread().threadId())")
    public long virtualThreadId;

    @Label("Scope ID")
    @Description("GpuTaskScope identifier")
    public long scopeId;

    @Label("Scope Name")
    @Description("Human-readable scope name for profiling")
    public String scopeName;

    /**
     * Populates computed fields from the launch configuration.
     *
     * <p>Call this after setting gridDim and blockDim fields.
     *
     * @param warpSize 32 for NVIDIA, 64 for AMD RDNA, 32 for AMD CDNA
     */
    public void computeDerivedFields(int warpSize) {
        this.totalBlocks = gridDimX * gridDimY * gridDimZ;
        long threadsPerBlock = (long) blockDimX * blockDimY * blockDimZ;
        this.totalThreads = (long) totalBlocks * threadsPerBlock;
        this.totalWarps = (totalThreads + warpSize - 1) / warpSize;
    }

    /**
     * Captures the current Java thread context.
     *
     * <p>Call this from the virtual thread executing the GPU operation.
     */
    public void captureThreadContext() {
        Thread current = Thread.currentThread();
        this.virtualThreadId = current.threadId();
    }
}
