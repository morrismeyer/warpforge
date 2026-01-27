package io.surfworks.warpforge.backend.amd.profiling;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * FFM bindings to roctracer (AMD GPU profiling API).
 *
 * <p>roctracer provides hardware-level profiling capabilities for AMD GPUs
 * including:
 * <ul>
 *   <li>Achieved occupancy measurement</li>
 *   <li>CU (Compute Unit) efficiency metrics</li>
 *   <li>Cache hit rates</li>
 *   <li>Wavefront stall analysis</li>
 *   <li>Memory throughput</li>
 * </ul>
 *
 * <p>Note: roctracer is part of the ROCm software stack and requires
 * ROCm to be installed.
 *
 * @see io.surfworks.warpforge.core.profiling.HardwareProfiler
 */
public final class RoctracerRuntime {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup ROCTRACER;
    private static final boolean AVAILABLE;

    // roctracer API function handles
    private static final MethodHandle roctracer_enable_callback;
    private static final MethodHandle roctracer_disable_callback;
    private static final MethodHandle roctracer_enable_activity;
    private static final MethodHandle roctracer_disable_activity;
    private static final MethodHandle roctracer_flush_activity;
    private static final MethodHandle roctracer_activity_push_external_correlation_id;
    private static final MethodHandle roctracer_activity_pop_external_correlation_id;

    // rocprofiler API function handles (may be separate library)
    private static final MethodHandle rocprofiler_open;
    private static final MethodHandle rocprofiler_close;
    private static final MethodHandle rocprofiler_start;
    private static final MethodHandle rocprofiler_stop;
    private static final MethodHandle rocprofiler_read;
    private static final MethodHandle rocprofiler_get_metrics;

    // Activity domain types
    public static final int ACTIVITY_DOMAIN_HIP_API = 0;
    public static final int ACTIVITY_DOMAIN_HIP_OPS = 1;
    public static final int ACTIVITY_DOMAIN_HSA_API = 2;
    public static final int ACTIVITY_DOMAIN_HSA_OPS = 3;
    public static final int ACTIVITY_DOMAIN_ROCTX = 4;

    // Operation types for HIP_OPS domain
    public static final int HIP_OP_KERNEL_DISPATCH = 0;
    public static final int HIP_OP_COPY = 1;
    public static final int HIP_OP_BARRIER = 2;

    // Result codes
    public static final int ROCTRACER_STATUS_SUCCESS = 0;
    public static final int ROCTRACER_STATUS_ERROR = 1;
    public static final int ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT = 2;
    public static final int ROCTRACER_STATUS_ERROR_INVALID_OPERATION = 3;

    // Common metric names
    public static final String METRIC_OCCUPANCY = "GPU_OCCUPANCY";
    public static final String METRIC_CU_OCCUPANCY = "CU_OCCUPANCY";
    public static final String METRIC_WAVE_OCCUPANCY = "WAVE_OCCUPANCY";
    public static final String METRIC_LDS_BANK_CONFLICT = "LDS_BANK_CONFLICT";
    public static final String METRIC_FETCH_SIZE = "FETCH_SIZE";
    public static final String METRIC_WRITE_SIZE = "WRITE_SIZE";
    public static final String METRIC_L2_CACHE_HIT = "L2_CACHE_HIT";
    public static final String METRIC_L1_CACHE_HIT = "TCP_READ_TAGCONFLICT_STALL_CYCLES";

    static {
        SymbolLookup lookup = null;
        boolean available = false;

        try {
            // roctracer is typically in /opt/rocm/lib or /opt/rocm-<version>/lib
            String rocmPath = System.getenv("ROCM_PATH");
            if (rocmPath == null) {
                rocmPath = "/opt/rocm";
            }

            String libraryPath = rocmPath + "/lib/libroctracer64.so";
            lookup = SymbolLookup.libraryLookup(libraryPath, Arena.global());
            available = true;
        } catch (IllegalArgumentException e) {
            // Try without explicit path
            try {
                lookup = SymbolLookup.libraryLookup("libroctracer64.so", Arena.global());
                available = true;
            } catch (IllegalArgumentException e2) {
                // roctracer not available
                lookup = SymbolLookup.loaderLookup();
            }
        }

        ROCTRACER = lookup;
        AVAILABLE = available;

        if (available) {
            // roctracer callback/activity API
            roctracer_enable_callback = downcall("roctracer_enable_callback",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            roctracer_disable_callback = downcall("roctracer_disable_callback",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            roctracer_enable_activity = downcall("roctracer_enable_activity_expl",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            roctracer_disable_activity = downcall("roctracer_disable_activity",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            roctracer_flush_activity = downcall("roctracer_flush_activity_expl",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            roctracer_activity_push_external_correlation_id = downcall("roctracer_activity_push_external_correlation_id",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            roctracer_activity_pop_external_correlation_id = downcall("roctracer_activity_pop_external_correlation_id",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            // rocprofiler (for hardware counters)
            rocprofiler_open = downcallOptional("rocprofiler_open",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            rocprofiler_close = downcallOptional("rocprofiler_close",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            rocprofiler_start = downcallOptional("rocprofiler_start",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            rocprofiler_stop = downcallOptional("rocprofiler_stop",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            rocprofiler_read = downcallOptional("rocprofiler_read",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            rocprofiler_get_metrics = downcallOptional("rocprofiler_get_metrics",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        } else {
            roctracer_enable_callback = null;
            roctracer_disable_callback = null;
            roctracer_enable_activity = null;
            roctracer_disable_activity = null;
            roctracer_flush_activity = null;
            roctracer_activity_push_external_correlation_id = null;
            roctracer_activity_pop_external_correlation_id = null;
            rocprofiler_open = null;
            rocprofiler_close = null;
            rocprofiler_start = null;
            rocprofiler_stop = null;
            rocprofiler_read = null;
            rocprofiler_get_metrics = null;
        }
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        return ROCTRACER.find(name)
            .map(symbol -> LINKER.downcallHandle(symbol, descriptor))
            .orElse(null);
    }

    private static MethodHandle downcallOptional(String name, FunctionDescriptor descriptor) {
        // For rocprofiler functions that may be in a separate library
        return ROCTRACER.find(name)
            .map(symbol -> LINKER.downcallHandle(symbol, descriptor))
            .orElse(null);
    }

    /**
     * Check if roctracer is available.
     *
     * @return true if roctracer library was successfully loaded
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Check if rocprofiler (hardware counter) support is available.
     *
     * @return true if rocprofiler functions are available
     */
    public static boolean isRocprofilerAvailable() {
        return AVAILABLE && rocprofiler_open != null;
    }

    /**
     * Enable activity recording for a domain and operation.
     *
     * @param domain the activity domain (e.g., ACTIVITY_DOMAIN_HIP_OPS)
     * @param operation the operation type (e.g., HIP_OP_KERNEL_DISPATCH)
     * @param pool the activity pool (can be NULL for default)
     * @return roctracer status code
     * @throws Throwable if the native call fails
     */
    public static int enableActivity(int domain, int operation, MemorySegment pool) throws Throwable {
        checkAvailable();
        if (roctracer_enable_activity == null) {
            return ROCTRACER_STATUS_ERROR_INVALID_OPERATION;
        }
        return (int) roctracer_enable_activity.invokeExact(domain, operation, pool);
    }

    /**
     * Disable activity recording for a domain and operation.
     *
     * @param domain the activity domain
     * @param operation the operation type
     * @return roctracer status code
     * @throws Throwable if the native call fails
     */
    public static int disableActivity(int domain, int operation) throws Throwable {
        checkAvailable();
        if (roctracer_disable_activity == null) {
            return ROCTRACER_STATUS_ERROR_INVALID_OPERATION;
        }
        return (int) roctracer_disable_activity.invokeExact(domain, operation);
    }

    /**
     * Flush all activity buffers.
     *
     * @param pool the activity pool (can be NULL for default)
     * @return roctracer status code
     * @throws Throwable if the native call fails
     */
    public static int flushActivity(MemorySegment pool) throws Throwable {
        checkAvailable();
        if (roctracer_flush_activity == null) {
            return ROCTRACER_STATUS_ERROR_INVALID_OPERATION;
        }
        return (int) roctracer_flush_activity.invokeExact(pool);
    }

    /**
     * Push an external correlation ID for tracking.
     *
     * @param correlationId the correlation ID
     * @return roctracer status code
     * @throws Throwable if the native call fails
     */
    public static int pushExternalCorrelationId(long correlationId) throws Throwable {
        checkAvailable();
        if (roctracer_activity_push_external_correlation_id == null) {
            return ROCTRACER_STATUS_ERROR_INVALID_OPERATION;
        }
        return (int) roctracer_activity_push_external_correlation_id.invokeExact(correlationId);
    }

    /**
     * Pop the external correlation ID.
     *
     * @param arena memory arena for allocation
     * @return the popped correlation ID
     * @throws Throwable if the native call fails
     */
    public static long popExternalCorrelationId(Arena arena) throws Throwable {
        checkAvailable();
        if (roctracer_activity_pop_external_correlation_id == null) {
            return -1;
        }
        MemorySegment idPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) roctracer_activity_pop_external_correlation_id.invokeExact(idPtr);
        if (result != ROCTRACER_STATUS_SUCCESS) {
            return -1;
        }
        return idPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    private static void checkAvailable() {
        if (!AVAILABLE) {
            throw new IllegalStateException("roctracer is not available. Ensure ROCm is installed.");
        }
    }

    /**
     * Kernel activity record structure.
     *
     * <p>This record represents the data from a HIP kernel dispatch
     * activity record.
     */
    public record KernelActivityRecord(
        long correlationId,
        String name,
        long startNs,
        long endNs,
        int deviceId,
        int streamId,
        int gridDimX,
        int gridDimY,
        int gridDimZ,
        int workgroupDimX,
        int workgroupDimY,
        int workgroupDimZ,
        long groupSegmentSize,
        long privateSegmentSize,
        int workgroupGroupSegmentByteSize,
        int kernargSegmentSize
    ) {
        /**
         * Get the kernel duration in nanoseconds.
         *
         * @return duration in nanoseconds
         */
        public long durationNanos() {
            return endNs - startNs;
        }

        /**
         * Get the number of workitems (threads).
         *
         * @return total workitems
         */
        public long totalWorkitems() {
            return (long) gridDimX * gridDimY * gridDimZ *
                   workgroupDimX * workgroupDimY * workgroupDimZ;
        }

        /**
         * Get the number of wavefronts.
         *
         * @param waveSize wavefront size (32 for RDNA, 64 for CDNA)
         * @return total wavefronts
         */
        public long totalWavefronts(int waveSize) {
            long workitemsPerWorkgroup = (long) workgroupDimX * workgroupDimY * workgroupDimZ;
            long wavesPerWorkgroup = (workitemsPerWorkgroup + waveSize - 1) / waveSize;
            long numWorkgroups = (long) gridDimX * gridDimY * gridDimZ;
            return numWorkgroups * wavesPerWorkgroup;
        }
    }

    // Layout for roctracer_record_t struct (simplified)
    private static final MemoryLayout ACTIVITY_RECORD_LAYOUT = MemoryLayout.structLayout(
        ValueLayout.JAVA_INT.withName("domain"),
        ValueLayout.JAVA_INT.withName("op"),
        ValueLayout.JAVA_LONG.withName("correlation_id"),
        ValueLayout.JAVA_LONG.withName("begin_ns"),
        ValueLayout.JAVA_LONG.withName("end_ns"),
        ValueLayout.JAVA_INT.withName("device_id"),
        ValueLayout.JAVA_INT.withName("queue_id")
    );

    private RoctracerRuntime() {
        // Utility class
    }
}
