package io.surfworks.warpforge.backend.amd.smi;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * FFM bindings to AMD ROCm System Management Interface (SMI).
 *
 * <p>ROCm SMI provides GPU monitoring and management capabilities:
 * <ul>
 *   <li>GPU utilization (% of time kernels were running)</li>
 *   <li>Memory utilization and usage</li>
 *   <li>Temperature and power monitoring</li>
 *   <li>Device identification and properties</li>
 * </ul>
 *
 * <p><b>Important caveat:</b> Like NVML, ROCm SMI's "utilization" measures
 * the percentage of time over the sample period during which the GPU was busy.
 * It does NOT measure what percentage of compute capacity (CUs) is being used.
 *
 * <p>For Orion-style occupancy-based admission control, we use utilization
 * as a proxy for GPU busyness, combined with active stream count and memory pressure.
 *
 * @see <a href="https://rocm.docs.amd.com/projects/amdsmi/en/docs-6.1.0/">ROCm SMI API Reference</a>
 */
public final class RocmSmiRuntime {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup SMI;
    private static final boolean AVAILABLE;

    // ROCm SMI function handles
    private static final MethodHandle rsmiInit;
    private static final MethodHandle rsmiShutDown;
    private static final MethodHandle rsmiNumMonitorDevices;
    private static final MethodHandle rsmiDevBusyPercentGet;
    private static final MethodHandle rsmiDevMemoryBusyPercentGet;
    private static final MethodHandle rsmiDevTempMetricGet;
    private static final MethodHandle rsmiDevPowerAveGet;
    private static final MethodHandle rsmiDevMemoryTotalGet;
    private static final MethodHandle rsmiDevMemoryUsageGet;
    private static final MethodHandle rsmiStatusString;

    // ROCm SMI return codes
    public static final int RSMI_STATUS_SUCCESS = 0;
    public static final int RSMI_STATUS_INVALID_ARGS = 1;
    public static final int RSMI_STATUS_NOT_SUPPORTED = 2;
    public static final int RSMI_STATUS_FILE_ERROR = 3;
    public static final int RSMI_STATUS_PERMISSION = 4;
    public static final int RSMI_STATUS_OUT_OF_RESOURCES = 5;
    public static final int RSMI_STATUS_INTERNAL_EXCEPTION = 6;
    public static final int RSMI_STATUS_INPUT_OUT_OF_BOUNDS = 7;
    public static final int RSMI_STATUS_INIT_ERROR = 8;
    public static final int RSMI_STATUS_NOT_YET_IMPLEMENTED = 9;
    public static final int RSMI_STATUS_NOT_FOUND = 10;

    // Temperature sensor types
    public static final int RSMI_TEMP_TYPE_EDGE = 0;
    public static final int RSMI_TEMP_TYPE_JUNCTION = 1;
    public static final int RSMI_TEMP_TYPE_MEMORY = 2;

    // Temperature metric types
    public static final int RSMI_TEMP_CURRENT = 0;

    // Memory types
    public static final int RSMI_MEM_TYPE_VRAM = 0;
    public static final int RSMI_MEM_TYPE_VIS_VRAM = 1;
    public static final int RSMI_MEM_TYPE_GTT = 2;

    // Init flags
    public static final long RSMI_INIT_FLAG_ALL_GPUS = 0;

    static {
        SymbolLookup lookup = null;
        boolean available = false;

        try {
            // Try to load librocm_smi64.so (Linux only - ROCm is Linux-only)
            lookup = SymbolLookup.libraryLookup("librocm_smi64.so", Arena.global());
            available = true;
        } catch (IllegalArgumentException e) {
            // ROCm SMI not available
            lookup = SymbolLookup.loaderLookup();
        }

        SMI = lookup;
        AVAILABLE = available;

        if (available) {
            rsmiInit = downcall("rsmi_init",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

            rsmiShutDown = downcall("rsmi_shut_down",
                FunctionDescriptor.of(ValueLayout.JAVA_INT));

            rsmiNumMonitorDevices = downcall("rsmi_num_monitor_devices",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            rsmiDevBusyPercentGet = downcall("rsmi_dev_busy_percent_get",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            rsmiDevMemoryBusyPercentGet = downcall("rsmi_dev_memory_busy_percent_get",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            rsmiDevTempMetricGet = downcall("rsmi_dev_temp_metric_get",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            rsmiDevPowerAveGet = downcall("rsmi_dev_power_ave_get",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            rsmiDevMemoryTotalGet = downcall("rsmi_dev_memory_total_get",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            rsmiDevMemoryUsageGet = downcall("rsmi_dev_memory_usage_get",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            rsmiStatusString = downcall("rsmi_status_string",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        } else {
            rsmiInit = null;
            rsmiShutDown = null;
            rsmiNumMonitorDevices = null;
            rsmiDevBusyPercentGet = null;
            rsmiDevMemoryBusyPercentGet = null;
            rsmiDevTempMetricGet = null;
            rsmiDevPowerAveGet = null;
            rsmiDevMemoryTotalGet = null;
            rsmiDevMemoryUsageGet = null;
            rsmiStatusString = null;
        }
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        return SMI.find(name)
            .map(symbol -> LINKER.downcallHandle(symbol, descriptor))
            .orElse(null);
    }

    private RocmSmiRuntime() {}

    /**
     * Check if ROCm SMI is available on this system.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Ensure ROCm SMI is available, throwing if not.
     */
    public static void ensureAvailable() {
        if (!AVAILABLE) {
            throw new UnsupportedOperationException("ROCm SMI is not available on this system");
        }
    }

    // ==================== Initialization ====================

    /**
     * Initialize ROCm SMI. Must be called before any other SMI functions.
     *
     * @return RSMI_STATUS_SUCCESS on success
     */
    public static int init() throws Throwable {
        ensureAvailable();
        return (int) rsmiInit.invokeExact(RSMI_INIT_FLAG_ALL_GPUS);
    }

    /**
     * Shutdown ROCm SMI. Should be called when done using SMI.
     *
     * @return RSMI_STATUS_SUCCESS on success
     */
    public static int shutdown() throws Throwable {
        ensureAvailable();
        return (int) rsmiShutDown.invokeExact();
    }

    // ==================== Device Queries ====================

    /**
     * Get the number of ROCm SMI-visible devices.
     */
    public static int getDeviceCount(Arena arena) throws Throwable {
        ensureAvailable();
        MemorySegment countPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) rsmiNumMonitorDevices.invokeExact(countPtr);
        checkError(result, "rsmi_num_monitor_devices");
        return countPtr.get(ValueLayout.JAVA_INT, 0);
    }

    // ==================== Utilization ====================

    /**
     * Get GPU utilization (busy percent).
     *
     * <p><b>Important:</b> GPU utilization measures the percentage of time over the
     * past sample period during which the GPU was busy. It does NOT measure what
     * percentage of the GPU's compute capacity is being used.
     *
     * @param deviceIndex Device index (0-based)
     * @return GPU utilization percentage (0-100)
     */
    public static int getGpuUtilization(Arena arena, int deviceIndex) throws Throwable {
        ensureAvailable();
        MemorySegment utilPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) rsmiDevBusyPercentGet.invokeExact(deviceIndex, utilPtr);
        checkError(result, "rsmi_dev_busy_percent_get");
        return utilPtr.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Get memory bandwidth utilization.
     *
     * @param deviceIndex Device index (0-based)
     * @return Memory utilization percentage (0-100)
     */
    public static int getMemoryUtilization(Arena arena, int deviceIndex) throws Throwable {
        ensureAvailable();
        MemorySegment utilPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) rsmiDevMemoryBusyPercentGet.invokeExact(deviceIndex, utilPtr);
        checkError(result, "rsmi_dev_memory_busy_percent_get");
        return utilPtr.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Get GPU utilization rates (both GPU and memory).
     *
     * @param deviceIndex Device index (0-based)
     * @return Utilization with gpu (0-100) and memory (0-100) percentages
     */
    public static Utilization getUtilizationRates(Arena arena, int deviceIndex) throws Throwable {
        int gpu = getGpuUtilization(arena, deviceIndex);
        int memory = getMemoryUtilization(arena, deviceIndex);
        return new Utilization(gpu, memory);
    }

    /**
     * GPU utilization rates.
     *
     * @param gpu Percent of time over the past sample period during which GPU was busy (0-100)
     * @param memory Percent of time over the past sample period during which memory was being accessed (0-100)
     */
    public record Utilization(int gpu, int memory) {
        @Override
        public String toString() {
            return String.format("Utilization[gpu=%d%%, memory=%d%%]", gpu, memory);
        }
    }

    // ==================== Memory Info ====================

    /**
     * Get total VRAM for a device.
     *
     * @param deviceIndex Device index
     * @return Total VRAM in bytes
     */
    public static long getMemoryTotal(Arena arena, int deviceIndex) throws Throwable {
        ensureAvailable();
        MemorySegment memPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) rsmiDevMemoryTotalGet.invokeExact(deviceIndex, RSMI_MEM_TYPE_VRAM, memPtr);
        checkError(result, "rsmi_dev_memory_total_get");
        return memPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Get used VRAM for a device.
     *
     * @param deviceIndex Device index
     * @return Used VRAM in bytes
     */
    public static long getMemoryUsed(Arena arena, int deviceIndex) throws Throwable {
        ensureAvailable();
        MemorySegment memPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) rsmiDevMemoryUsageGet.invokeExact(deviceIndex, RSMI_MEM_TYPE_VRAM, memPtr);
        checkError(result, "rsmi_dev_memory_usage_get");
        return memPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Get memory information for a device.
     *
     * @param deviceIndex Device index
     * @return Memory info with total, free, and used bytes
     */
    public static MemoryInfo getMemoryInfo(Arena arena, int deviceIndex) throws Throwable {
        long total = getMemoryTotal(arena, deviceIndex);
        long used = getMemoryUsed(arena, deviceIndex);
        long free = total - used;
        return new MemoryInfo(total, free, used);
    }

    /**
     * GPU memory information.
     *
     * @param total Total installed GPU memory in bytes
     * @param free Unallocated GPU memory in bytes
     * @param used Allocated GPU memory in bytes
     */
    public record MemoryInfo(long total, long free, long used) {
        public double usedPercent() {
            return total > 0 ? (used * 100.0 / total) : 0;
        }

        @Override
        public String toString() {
            return String.format("MemoryInfo[total=%.1fGB, free=%.1fGB, used=%.1fGB (%.1f%%)]",
                total / 1e9, free / 1e9, used / 1e9, usedPercent());
        }
    }

    // ==================== Device Properties ====================

    /**
     * Get GPU temperature.
     *
     * @param deviceIndex Device index
     * @return Temperature in degrees Celsius (millidegrees from SMI, converted)
     */
    public static int getTemperature(Arena arena, int deviceIndex) throws Throwable {
        ensureAvailable();
        MemorySegment tempPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) rsmiDevTempMetricGet.invokeExact(deviceIndex, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, tempPtr);
        checkError(result, "rsmi_dev_temp_metric_get");
        // ROCm SMI returns millidegrees, convert to degrees
        long millidegrees = tempPtr.get(ValueLayout.JAVA_LONG, 0);
        return (int) (millidegrees / 1000);
    }

    /**
     * Get GPU power usage.
     *
     * @param deviceIndex Device index
     * @return Power usage in milliwatts (microwatts from SMI, converted)
     */
    public static int getPowerUsage(Arena arena, int deviceIndex) throws Throwable {
        ensureAvailable();
        MemorySegment powerPtr = arena.allocate(ValueLayout.JAVA_LONG);
        // sensor_ind = 0 for default power sensor
        int result = (int) rsmiDevPowerAveGet.invokeExact(deviceIndex, 0, powerPtr);
        checkError(result, "rsmi_dev_power_ave_get");
        // ROCm SMI returns microwatts, convert to milliwatts
        long microwatts = powerPtr.get(ValueLayout.JAVA_LONG, 0);
        return (int) (microwatts / 1000);
    }

    // ==================== Error Handling ====================

    /**
     * Get error string for a ROCm SMI error code.
     */
    public static String getErrorString(int errorCode) {
        if (!AVAILABLE || rsmiStatusString == null) {
            return "RSMI_STATUS_" + errorCode;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment strPtrPtr = arena.allocate(ValueLayout.ADDRESS);
            int result = (int) rsmiStatusString.invokeExact(errorCode, strPtrPtr);
            if (result == RSMI_STATUS_SUCCESS) {
                MemorySegment strPtr = strPtrPtr.get(ValueLayout.ADDRESS, 0);
                if (!strPtr.equals(MemorySegment.NULL)) {
                    return strPtr.reinterpret(256).getString(0);
                }
            }
        } catch (Throwable t) {
            // Ignore
        }
        return "RSMI_STATUS_" + errorCode;
    }

    /**
     * Check a ROCm SMI result and throw if not success.
     */
    public static void checkError(int result, String operation) {
        if (result != RSMI_STATUS_SUCCESS) {
            throw new RocmSmiException(operation + " failed: " + getErrorString(result) + " (" + result + ")");
        }
    }

    /**
     * Exception thrown when a ROCm SMI operation fails.
     */
    public static class RocmSmiException extends RuntimeException {
        public RocmSmiException(String message) {
            super(message);
        }

        public RocmSmiException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
