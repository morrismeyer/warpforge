package io.surfworks.warpforge.core.jfr;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Timespan;

/**
 * JFR event recording GPU kernel compilation.
 *
 * <p>Captures compilation time and cache status for PTX/SPIR-V kernel compilation,
 * enabling analysis of JIT compilation overhead.
 *
 * <p>Usage:
 * <pre>{@code
 * GpuCompilationEvent event = new GpuCompilationEvent();
 * event.kernelName = "add_f32_salt0";
 * event.backend = "PTX";
 * event.compileTimeMicros = elapsedMicros;
 * event.wasCached = false;
 * event.deviceIndex = 0;
 * event.commit();
 * }</pre>
 */
@Name("io.surfworks.warpforge.GpuCompilation")
@Label("GPU Kernel Compilation")
@Category({"WarpForge", "GPU", "Compilation"})
@Description("Records GPU kernel compilation time and cache status")
public class GpuCompilationEvent extends Event {

    @Label("Kernel Name")
    @Description("Name of the compiled kernel")
    public String kernelName;

    @Label("Backend")
    @Description("Compilation backend: PTX, SPIR-V, HIP")
    public String backend;

    @Label("Compile Time")
    @Description("Compilation time in microseconds")
    @Timespan(Timespan.MICROSECONDS)
    public long compileTimeMicros;

    @Label("Cached")
    @Description("Whether the compiled kernel was retrieved from cache")
    public boolean wasCached;

    @Label("Device Index")
    @Description("Target GPU device index")
    public int deviceIndex;

    @Label("PTX Size")
    @Description("Size of the PTX source in bytes (0 if not applicable)")
    public long ptxSizeBytes;

    @Label("CUBIN Size")
    @Description("Size of the compiled CUBIN in bytes (0 if not applicable)")
    public long cubinSizeBytes;

    @Label("Salt Level")
    @Description("Instrumentation salt level (0=none, 1=timing, 2=trace)")
    public int saltLevel;
}
