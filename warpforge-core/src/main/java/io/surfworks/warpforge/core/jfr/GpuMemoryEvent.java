package io.surfworks.warpforge.core.jfr;

import jdk.jfr.Category;
import jdk.jfr.DataAmount;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.Timespan;

/**
 * JFR event recording GPU memory transfer operations.
 *
 * <p>Captures timing and bandwidth metrics for host-device memory transfers,
 * enabling identification of memory-bound bottlenecks.
 *
 * <p>Usage:
 * <pre>{@code
 * GpuMemoryEvent event = new GpuMemoryEvent();
 * event.direction = "H2D";
 * event.bytes = bufferSize;
 * event.timeMicros = elapsedMicros;
 * event.bandwidthGBps = (bytes / 1e9) / (timeMicros / 1e6);
 * event.deviceIndex = 0;
 * event.commit();
 * }</pre>
 */
@Name("io.surfworks.warpforge.GpuMemory")
@Label("GPU Memory Transfer")
@Category({"WarpForge", "GPU", "Memory"})
@Description("Records GPU memory transfer time and bandwidth")
public class GpuMemoryEvent extends Event {

    @Label("Direction")
    @Description("Transfer direction: H2D (host to device), D2H (device to host), D2D (device to device)")
    public String direction;

    @Label("Bytes")
    @Description("Number of bytes transferred")
    @DataAmount
    public long bytes;

    @Label("Transfer Time")
    @Description("Transfer time in microseconds")
    @Timespan(Timespan.MICROSECONDS)
    public long timeMicros;

    @Label("Bandwidth (GB/s)")
    @Description("Effective bandwidth in GB/s")
    public double bandwidthGBps;

    @Label("Device Index")
    @Description("GPU device index")
    public int deviceIndex;

    @Label("Async")
    @Description("Whether the transfer was asynchronous")
    public boolean async;

    @Label("Pinned Memory")
    @Description("Whether pinned (page-locked) host memory was used")
    public boolean pinnedMemory;
}
