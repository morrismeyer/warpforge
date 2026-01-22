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
}
