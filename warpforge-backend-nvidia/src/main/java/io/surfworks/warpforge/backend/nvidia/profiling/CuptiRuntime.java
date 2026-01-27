package io.surfworks.warpforge.backend.nvidia.profiling;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * FFM bindings to CUPTI (CUDA Profiling Tools Interface).
 *
 * <p>CUPTI provides hardware-level profiling capabilities including:
 * <ul>
 *   <li>Achieved occupancy measurement</li>
 *   <li>SM efficiency metrics</li>
 *   <li>Cache hit rates</li>
 *   <li>Warp stall analysis</li>
 *   <li>Memory throughput</li>
 * </ul>
 *
 * <p>Note: CUPTI is a separate library from the CUDA driver and requires
 * the CUDA Toolkit to be installed (not just the driver).
 *
 * @see io.surfworks.warpforge.core.profiling.HardwareProfiler
 */
public final class CuptiRuntime {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup CUPTI;
    private static final boolean AVAILABLE;

    // CUPTI Activity API function handles
    private static final MethodHandle cuptiActivityEnable;
    private static final MethodHandle cuptiActivityDisable;
    private static final MethodHandle cuptiActivityFlushAll;
    private static final MethodHandle cuptiActivityGetNextRecord;
    private static final MethodHandle cuptiActivityRegisterCallbacks;

    // CUPTI Metric API function handles
    private static final MethodHandle cuptiMetricGetIdFromName;
    private static final MethodHandle cuptiMetricGetValue;
    private static final MethodHandle cuptiMetricEnumByDevice;

    // CUPTI general functions
    private static final MethodHandle cuptiGetResultString;
    private static final MethodHandle cuptiGetVersion;

    // Activity types
    public static final int CUPTI_ACTIVITY_KIND_KERNEL = 3;
    public static final int CUPTI_ACTIVITY_KIND_MEMCPY = 1;
    public static final int CUPTI_ACTIVITY_KIND_MEMSET = 2;
    public static final int CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10;
    public static final int CUPTI_ACTIVITY_KIND_DRIVER = 6;
    public static final int CUPTI_ACTIVITY_KIND_RUNTIME = 7;

    // CUPTI Result codes
    public static final int CUPTI_SUCCESS = 0;
    public static final int CUPTI_ERROR_INVALID_PARAMETER = 1;
    public static final int CUPTI_ERROR_OUT_OF_MEMORY = 4;
    public static final int CUPTI_ERROR_MAX_LIMIT_REACHED = 14;
    public static final int CUPTI_ERROR_NOT_INITIALIZED = 15;

    // Metric IDs for common metrics
    public static final String METRIC_ACHIEVED_OCCUPANCY = "achieved_occupancy";
    public static final String METRIC_SM_EFFICIENCY = "sm_efficiency";
    public static final String METRIC_DRAM_THROUGHPUT = "dram_throughput";
    public static final String METRIC_L1_HIT_RATE = "l1_cache_global_hit_rate";
    public static final String METRIC_L2_HIT_RATE = "l2_l1_read_hit_rate";
    public static final String METRIC_WARP_EXECUTION_EFFICIENCY = "warp_execution_efficiency";

    static {
        SymbolLookup lookup = null;
        boolean available = false;

        try {
            // CUPTI is typically in CUDA_HOME/extras/CUPTI/lib64
            // Try common library names
            String os = System.getProperty("os.name").toLowerCase();
            String libraryName;
            if (os.contains("win")) {
                libraryName = "cupti64_2024.3.2";  // Windows: version-specific DLL
            } else if (os.contains("mac")) {
                libraryName = "libcupti.dylib";
            } else {
                libraryName = "libcupti.so";
            }
            lookup = SymbolLookup.libraryLookup(libraryName, Arena.global());
            available = true;
        } catch (IllegalArgumentException e) {
            // Try with explicit path
            try {
                String cudaHome = System.getenv("CUDA_HOME");
                if (cudaHome == null) {
                    cudaHome = "/usr/local/cuda";
                }
                String libPath = cudaHome + "/extras/CUPTI/lib64/libcupti.so";
                lookup = SymbolLookup.libraryLookup(libPath, Arena.global());
                available = true;
            } catch (IllegalArgumentException e2) {
                // CUPTI not available
                lookup = SymbolLookup.loaderLookup();
            }
        }

        CUPTI = lookup;
        AVAILABLE = available;

        if (available) {
            // Activity API
            cuptiActivityEnable = downcall("cuptiActivityEnable",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            cuptiActivityDisable = downcall("cuptiActivityDisable",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            cuptiActivityFlushAll = downcall("cuptiActivityFlushAll",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            cuptiActivityGetNextRecord = downcall("cuptiActivityGetNextRecord",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            cuptiActivityRegisterCallbacks = downcall("cuptiActivityRegisterCallbacks",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

            // Metric API
            cuptiMetricGetIdFromName = downcall("cuptiMetricGetIdFromName",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            cuptiMetricGetValue = downcall("cuptiMetricGetValue",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            cuptiMetricEnumByDevice = downcall("cuptiMetricEnumByDevice",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

            // General
            cuptiGetResultString = downcall("cuptiGetResultString",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            cuptiGetVersion = downcall("cuptiGetVersion",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        } else {
            cuptiActivityEnable = null;
            cuptiActivityDisable = null;
            cuptiActivityFlushAll = null;
            cuptiActivityGetNextRecord = null;
            cuptiActivityRegisterCallbacks = null;
            cuptiMetricGetIdFromName = null;
            cuptiMetricGetValue = null;
            cuptiMetricEnumByDevice = null;
            cuptiGetResultString = null;
            cuptiGetVersion = null;
        }
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        return CUPTI.find(name)
            .map(symbol -> LINKER.downcallHandle(symbol, descriptor))
            .orElse(null);
    }

    /**
     * Check if CUPTI is available.
     *
     * @return true if CUPTI library was successfully loaded
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Enable collection of a specific activity kind.
     *
     * @param activityKind the activity kind (e.g., CUPTI_ACTIVITY_KIND_KERNEL)
     * @return CUPTI result code
     * @throws Throwable if the native call fails
     */
    public static int activityEnable(int activityKind) throws Throwable {
        checkAvailable();
        return (int) cuptiActivityEnable.invokeExact(activityKind);
    }

    /**
     * Disable collection of a specific activity kind.
     *
     * @param activityKind the activity kind
     * @return CUPTI result code
     * @throws Throwable if the native call fails
     */
    public static int activityDisable(int activityKind) throws Throwable {
        checkAvailable();
        return (int) cuptiActivityDisable.invokeExact(activityKind);
    }

    /**
     * Flush all activity buffers.
     *
     * @param flag flush flag (0 for blocking flush)
     * @return CUPTI result code
     * @throws Throwable if the native call fails
     */
    public static int activityFlushAll(int flag) throws Throwable {
        checkAvailable();
        return (int) cuptiActivityFlushAll.invokeExact(flag);
    }

    /**
     * Get the CUPTI version.
     *
     * @param arena memory arena for allocation
     * @return CUPTI version number
     * @throws Throwable if the native call fails
     */
    public static int getVersion(Arena arena) throws Throwable {
        checkAvailable();
        MemorySegment versionPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) cuptiGetVersion.invokeExact(versionPtr);
        if (result != CUPTI_SUCCESS) {
            throw new RuntimeException("cuptiGetVersion failed with error " + result);
        }
        return versionPtr.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Get the error string for a CUPTI result code.
     *
     * @param arena memory arena for allocation
     * @param result the CUPTI result code
     * @return human-readable error string
     * @throws Throwable if the native call fails
     */
    public static String getResultString(Arena arena, int result) throws Throwable {
        checkAvailable();
        MemorySegment strPtr = arena.allocate(ValueLayout.ADDRESS);
        @SuppressWarnings("unused")
        int ret = (int) cuptiGetResultString.invokeExact(result, strPtr);
        MemorySegment strSeg = strPtr.get(ValueLayout.ADDRESS, 0);
        if (strSeg.equals(MemorySegment.NULL)) {
            return "Unknown error " + result;
        }
        return strSeg.reinterpret(256).getString(0);
    }

    /**
     * Get a metric ID from its name.
     *
     * @param arena memory arena for allocation
     * @param device CUDA device index
     * @param metricName the metric name (e.g., "achieved_occupancy")
     * @return metric ID, or -1 if not found
     * @throws Throwable if the native call fails
     */
    public static int getMetricIdFromName(Arena arena, int device, String metricName) throws Throwable {
        checkAvailable();
        MemorySegment namePtr = arena.allocateFrom(metricName + "\0", java.nio.charset.StandardCharsets.UTF_8);
        MemorySegment idPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) cuptiMetricGetIdFromName.invokeExact(device, namePtr, idPtr);
        if (result != CUPTI_SUCCESS) {
            return -1;
        }
        return idPtr.get(ValueLayout.JAVA_INT, 0);
    }

    private static void checkAvailable() {
        if (!AVAILABLE) {
            throw new IllegalStateException("CUPTI is not available. Ensure CUDA Toolkit is installed.");
        }
    }

    /**
     * Kernel activity record structure.
     *
     * <p>This record represents the data from a CUPTI_ACTIVITY_KIND_KERNEL
     * activity record.
     */
    public record KernelActivityRecord(
        long correlationId,
        String name,
        long start,
        long end,
        int deviceId,
        int streamId,
        int gridX,
        int gridY,
        int gridZ,
        int blockX,
        int blockY,
        int blockZ,
        int dynamicSharedMemory,
        int staticSharedMemory,
        int registersPerThread
    ) {
        /**
         * Get the kernel duration in nanoseconds.
         *
         * @return duration in nanoseconds
         */
        public long durationNanos() {
            return end - start;
        }
    }

    // Layout for CUpti_ActivityKernel4 struct (CUPTI 10.0+)
    // Note: Actual struct layout is complex; this is simplified
    private static final MemoryLayout KERNEL_ACTIVITY_LAYOUT = MemoryLayout.structLayout(
        ValueLayout.JAVA_INT.withName("kind"),
        ValueLayout.JAVA_INT.withName("pad"),
        ValueLayout.JAVA_LONG.withName("correlationId"),
        ValueLayout.JAVA_LONG.withName("start"),
        ValueLayout.JAVA_LONG.withName("end"),
        ValueLayout.JAVA_INT.withName("deviceId"),
        ValueLayout.JAVA_INT.withName("contextId"),
        ValueLayout.JAVA_INT.withName("streamId"),
        ValueLayout.JAVA_INT.withName("gridX"),
        ValueLayout.JAVA_INT.withName("gridY"),
        ValueLayout.JAVA_INT.withName("gridZ"),
        ValueLayout.JAVA_INT.withName("blockX"),
        ValueLayout.JAVA_INT.withName("blockY"),
        ValueLayout.JAVA_INT.withName("blockZ"),
        ValueLayout.JAVA_INT.withName("dynamicSharedMemory"),
        ValueLayout.JAVA_INT.withName("staticSharedMemory"),
        ValueLayout.JAVA_INT.withName("registersPerThread"),
        ValueLayout.ADDRESS.withName("name")
    );

    private CuptiRuntime() {
        // Utility class
    }
}
