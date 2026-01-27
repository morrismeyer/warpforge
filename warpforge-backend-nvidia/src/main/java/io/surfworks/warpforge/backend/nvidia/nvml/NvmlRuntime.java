package io.surfworks.warpforge.backend.nvidia.nvml;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * FFM bindings to NVIDIA Management Library (NVML).
 *
 * <p>NVML provides GPU monitoring and management capabilities:
 * <ul>
 *   <li>GPU utilization (% of time kernels were running)</li>
 *   <li>Memory utilization and usage</li>
 *   <li>Temperature and power monitoring</li>
 *   <li>Device identification and properties</li>
 * </ul>
 *
 * <p><b>Important caveat from architecture docs:</b> NVML's "utilization" means
 * "% of time any kernel was running", not "% of compute capacity used".
 * A kernel using 10% of SMs still shows 100% utilization while running.
 *
 * <p>For Orion-style occupancy-based admission control, we use utilization
 * as a proxy for GPU busyness, combined with active stream count and memory pressure.
 *
 * @see <a href="https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html">NVML API Reference</a>
 */
public final class NvmlRuntime {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup NVML;
    private static final boolean AVAILABLE;

    // NVML function handles
    private static final MethodHandle nvmlInit;
    private static final MethodHandle nvmlShutdown;
    private static final MethodHandle nvmlDeviceGetCount;
    private static final MethodHandle nvmlDeviceGetHandleByIndex;
    private static final MethodHandle nvmlDeviceGetUtilizationRates;
    private static final MethodHandle nvmlDeviceGetMemoryInfo;
    private static final MethodHandle nvmlDeviceGetName;
    private static final MethodHandle nvmlDeviceGetTemperature;
    private static final MethodHandle nvmlDeviceGetPowerUsage;
    private static final MethodHandle nvmlErrorString;

    // NVML return codes
    public static final int NVML_SUCCESS = 0;
    public static final int NVML_ERROR_UNINITIALIZED = 1;
    public static final int NVML_ERROR_INVALID_ARGUMENT = 2;
    public static final int NVML_ERROR_NOT_SUPPORTED = 3;
    public static final int NVML_ERROR_NO_PERMISSION = 4;
    public static final int NVML_ERROR_NOT_FOUND = 6;
    public static final int NVML_ERROR_DRIVER_NOT_LOADED = 9;

    // Temperature sensor types
    public static final int NVML_TEMPERATURE_GPU = 0;

    // Memory layout for nvmlUtilization_t struct
    // struct nvmlUtilization_t { unsigned int gpu; unsigned int memory; }
    private static final MemoryLayout UTILIZATION_LAYOUT = MemoryLayout.structLayout(
        ValueLayout.JAVA_INT.withName("gpu"),
        ValueLayout.JAVA_INT.withName("memory")
    );

    // Memory layout for nvmlMemory_t struct (v1)
    // struct nvmlMemory_t { unsigned long long total; unsigned long long free; unsigned long long used; }
    private static final MemoryLayout MEMORY_LAYOUT = MemoryLayout.structLayout(
        ValueLayout.JAVA_LONG.withName("total"),
        ValueLayout.JAVA_LONG.withName("free"),
        ValueLayout.JAVA_LONG.withName("used")
    );

    static {
        SymbolLookup lookup = null;
        boolean available = false;

        try {
            // Try to load libnvidia-ml.so (Linux) or nvml.dll (Windows)
            String libraryName = System.getProperty("os.name").toLowerCase().contains("win")
                ? "nvml"
                : "libnvidia-ml.so.1";
            lookup = SymbolLookup.libraryLookup(libraryName, Arena.global());
            available = true;
        } catch (IllegalArgumentException e) {
            // NVML not available
            lookup = SymbolLookup.loaderLookup();
        }

        NVML = lookup;
        AVAILABLE = available;

        if (available) {
            nvmlInit = downcall("nvmlInit_v2",
                FunctionDescriptor.of(ValueLayout.JAVA_INT));

            nvmlShutdown = downcall("nvmlShutdown",
                FunctionDescriptor.of(ValueLayout.JAVA_INT));

            nvmlDeviceGetCount = downcall("nvmlDeviceGetCount_v2",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            nvmlDeviceGetHandleByIndex = downcall("nvmlDeviceGetHandleByIndex_v2",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            nvmlDeviceGetUtilizationRates = downcall("nvmlDeviceGetUtilizationRates",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

            nvmlDeviceGetMemoryInfo = downcall("nvmlDeviceGetMemoryInfo",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

            nvmlDeviceGetName = downcall("nvmlDeviceGetName",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

            nvmlDeviceGetTemperature = downcall("nvmlDeviceGetTemperature",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            nvmlDeviceGetPowerUsage = downcall("nvmlDeviceGetPowerUsage",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));

            nvmlErrorString = downcall("nvmlErrorString",
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        } else {
            nvmlInit = null;
            nvmlShutdown = null;
            nvmlDeviceGetCount = null;
            nvmlDeviceGetHandleByIndex = null;
            nvmlDeviceGetUtilizationRates = null;
            nvmlDeviceGetMemoryInfo = null;
            nvmlDeviceGetName = null;
            nvmlDeviceGetTemperature = null;
            nvmlDeviceGetPowerUsage = null;
            nvmlErrorString = null;
        }
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        return NVML.find(name)
            .map(symbol -> LINKER.downcallHandle(symbol, descriptor))
            .orElse(null);
    }

    private NvmlRuntime() {}

    /**
     * Check if NVML is available on this system.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Ensure NVML is available, throwing if not.
     */
    public static void ensureAvailable() {
        if (!AVAILABLE) {
            throw new UnsupportedOperationException("NVML is not available on this system");
        }
    }

    // ==================== Initialization ====================

    /**
     * Initialize NVML. Must be called before any other NVML functions.
     *
     * @return NVML_SUCCESS on success
     */
    public static int init() throws Throwable {
        ensureAvailable();
        return (int) nvmlInit.invokeExact();
    }

    /**
     * Shutdown NVML. Should be called when done using NVML.
     *
     * @return NVML_SUCCESS on success
     */
    public static int shutdown() throws Throwable {
        ensureAvailable();
        return (int) nvmlShutdown.invokeExact();
    }

    // ==================== Device Queries ====================

    /**
     * Get the number of NVML-visible devices.
     */
    public static int getDeviceCount(Arena arena) throws Throwable {
        ensureAvailable();
        MemorySegment countPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) nvmlDeviceGetCount.invokeExact(countPtr);
        checkError(result, "nvmlDeviceGetCount");
        return countPtr.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Get a device handle by index.
     *
     * @param index Device index (0-based)
     * @return Device handle (nvmlDevice_t)
     */
    public static long getDeviceHandle(Arena arena, int index) throws Throwable {
        ensureAvailable();
        MemorySegment handlePtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) nvmlDeviceGetHandleByIndex.invokeExact(index, handlePtr);
        checkError(result, "nvmlDeviceGetHandleByIndex");
        return handlePtr.get(ValueLayout.JAVA_LONG, 0);
    }

    // ==================== Utilization ====================

    /**
     * Get GPU utilization rates.
     *
     * <p><b>Important:</b> GPU utilization measures the percentage of time over the
     * past sample period during which one or more kernels was executing on the GPU.
     * It does NOT measure what percentage of the GPU's compute capacity is being used.
     *
     * <p>Sampling period varies by product: 166ms to 1 second.
     *
     * @param device Device handle from getDeviceHandle()
     * @return Utilization with gpu (0-100) and memory (0-100) percentages
     */
    public static Utilization getUtilizationRates(Arena arena, long device) throws Throwable {
        ensureAvailable();
        MemorySegment utilPtr = arena.allocate(UTILIZATION_LAYOUT);
        int result = (int) nvmlDeviceGetUtilizationRates.invokeExact(device, utilPtr);
        checkError(result, "nvmlDeviceGetUtilizationRates");

        int gpu = utilPtr.get(ValueLayout.JAVA_INT, 0);
        int memory = utilPtr.get(ValueLayout.JAVA_INT, 4);
        return new Utilization(gpu, memory);
    }

    /**
     * GPU utilization rates.
     *
     * @param gpu Percent of time over the past sample period during which kernels were executing (0-100)
     * @param memory Percent of time over the past sample period during which GPU memory was being read/written (0-100)
     */
    public record Utilization(int gpu, int memory) {
        @Override
        public String toString() {
            return String.format("Utilization[gpu=%d%%, memory=%d%%]", gpu, memory);
        }
    }

    // ==================== Memory Info ====================

    /**
     * Get memory information for a device.
     *
     * @param device Device handle
     * @return Memory info with total, free, and used bytes
     */
    public static MemoryInfo getMemoryInfo(Arena arena, long device) throws Throwable {
        ensureAvailable();
        MemorySegment memPtr = arena.allocate(MEMORY_LAYOUT);
        int result = (int) nvmlDeviceGetMemoryInfo.invokeExact(device, memPtr);
        checkError(result, "nvmlDeviceGetMemoryInfo");

        long total = memPtr.get(ValueLayout.JAVA_LONG, 0);
        long free = memPtr.get(ValueLayout.JAVA_LONG, 8);
        long used = memPtr.get(ValueLayout.JAVA_LONG, 16);
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
     * Get device name.
     *
     * @param device Device handle
     * @return Device name (e.g., "NVIDIA GeForce RTX 3090")
     */
    public static String getDeviceName(Arena arena, long device) throws Throwable {
        ensureAvailable();
        int bufferSize = 96; // NVML_DEVICE_NAME_BUFFER_SIZE
        MemorySegment namePtr = arena.allocate(bufferSize);
        int result = (int) nvmlDeviceGetName.invokeExact(device, namePtr, bufferSize);
        checkError(result, "nvmlDeviceGetName");
        return namePtr.getString(0);
    }

    /**
     * Get GPU temperature.
     *
     * @param device Device handle
     * @return Temperature in degrees Celsius
     */
    public static int getTemperature(Arena arena, long device) throws Throwable {
        ensureAvailable();
        MemorySegment tempPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) nvmlDeviceGetTemperature.invokeExact(device, NVML_TEMPERATURE_GPU, tempPtr);
        checkError(result, "nvmlDeviceGetTemperature");
        return tempPtr.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Get GPU power usage.
     *
     * @param device Device handle
     * @return Power usage in milliwatts
     */
    public static int getPowerUsage(Arena arena, long device) throws Throwable {
        ensureAvailable();
        MemorySegment powerPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) nvmlDeviceGetPowerUsage.invokeExact(device, powerPtr);
        checkError(result, "nvmlDeviceGetPowerUsage");
        return powerPtr.get(ValueLayout.JAVA_INT, 0);
    }

    // ==================== Error Handling ====================

    /**
     * Get error string for an NVML error code.
     */
    public static String getErrorString(int errorCode) {
        if (!AVAILABLE || nvmlErrorString == null) {
            return "NVML_ERROR_" + errorCode;
        }
        try {
            MemorySegment strPtr = (MemorySegment) nvmlErrorString.invokeExact(errorCode);
            if (!strPtr.equals(MemorySegment.NULL)) {
                return strPtr.reinterpret(256).getString(0);
            }
        } catch (Throwable t) {
            // Ignore
        }
        return "NVML_ERROR_" + errorCode;
    }

    /**
     * Check an NVML result and throw if not success.
     */
    public static void checkError(int result, String operation) {
        if (result != NVML_SUCCESS) {
            throw new NvmlException(operation + " failed: " + getErrorString(result) + " (" + result + ")");
        }
    }

    /**
     * Exception thrown when an NVML operation fails.
     */
    public static class NvmlException extends RuntimeException {
        public NvmlException(String message) {
            super(message);
        }

        public NvmlException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
