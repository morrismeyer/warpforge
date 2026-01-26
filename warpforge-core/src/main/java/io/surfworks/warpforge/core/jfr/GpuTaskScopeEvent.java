package io.surfworks.warpforge.core.jfr;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;
import jdk.jfr.Timespan;

/**
 * JFR event recording GPU task scope lifecycle.
 *
 * <p>This event is emitted at the start and end of each {@code GpuTaskScope},
 * capturing metrics for:
 * <ul>
 *   <li>Scope duration and task counts</li>
 *   <li>Success/failure rates</li>
 *   <li>Stream resource utilization</li>
 *   <li>Device affinity</li>
 * </ul>
 *
 * <p>These events integrate with Java Flight Recorder for unified profiling
 * of GPU workloads alongside JVM metrics.
 *
 * <p>Example JFR recording analysis:
 * <pre>{@code
 * # Record for 60 seconds
 * jcmd <pid> JFR.start duration=60s filename=gpu-profile.jfr
 *
 * # Analyze with JMC or jfr tool
 * jfr print --events io.surfworks.warpforge.GpuTaskScope gpu-profile.jfr
 * }</pre>
 *
 * @see io.surfworks.warpforge.core.concurrency.GpuTaskScope
 */
@Name("io.surfworks.warpforge.GpuTaskScope")
@Label("GPU Task Scope Lifecycle")
@Category({"WarpForge", "GPU", "Concurrency"})
@Description("Records structured concurrency scope lifecycle for GPU operations")
@StackTrace(false) // Reduce overhead; enable selectively for debugging
public class GpuTaskScopeEvent extends Event {

    /**
     * Unique identifier for this scope instance.
     */
    @Label("Scope ID")
    @Description("Unique identifier for this scope instance")
    public long scopeId;

    /**
     * Human-readable name for the scope (e.g., "inference-batch", "training-step").
     */
    @Label("Scope Name")
    @Description("Human-readable name for the scope")
    public String scopeName;

    /**
     * Lifecycle phase: START, END, CANCELLED, or FAILED.
     */
    @Label("Phase")
    @Description("Lifecycle phase: START, END, CANCELLED, or FAILED")
    public String phase;

    /**
     * Total scope duration in microseconds (only set on END phase).
     */
    @Label("Duration")
    @Description("Total scope duration in microseconds (only set on END phase)")
    @Timespan(Timespan.MICROSECONDS)
    public long durationMicros;

    /**
     * Number of tasks forked within this scope.
     */
    @Label("Tasks Forked")
    @Description("Number of tasks forked within this scope")
    public int tasksForked;

    /**
     * Number of tasks that completed successfully.
     */
    @Label("Tasks Completed")
    @Description("Number of tasks that completed successfully")
    public int tasksCompleted;

    /**
     * Number of tasks that failed or were cancelled.
     */
    @Label("Tasks Failed")
    @Description("Number of tasks that failed or were cancelled")
    public int tasksFailed;

    /**
     * GPU device index used by this scope.
     */
    @Label("Device Index")
    @Description("GPU device index used by this scope")
    public int deviceIndex;

    /**
     * Number of CUDA/HIP streams acquired by this scope.
     */
    @Label("Streams Acquired")
    @Description("Number of CUDA/HIP streams acquired by this scope")
    public int streamsAcquired;
}
