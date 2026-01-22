package io.surfworks.warpforge.backend.amd.hip;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;

/**
 * FFM bindings to HIP Runtime API.
 *
 * <p>This class provides low-level access to AMD HIP functionality via Java's
 * Foreign Function & Memory API. HIP (Heterogeneous-computing Interface for Portability)
 * provides a CUDA-like API for AMD GPUs.
 *
 * <p>The HIP API closely mirrors CUDA, making porting straightforward:
 * <ul>
 *   <li>hipMalloc ↔ cudaMalloc</li>
 *   <li>hipMemcpy ↔ cudaMemcpy</li>
 *   <li>hipLaunchKernelGGL ↔ cudaLaunchKernel</li>
 * </ul>
 *
 * <p>Error handling: All HIP functions return hipError_t. A value of 0
 * (hipSuccess) indicates success.
 *
 * @see <a href="https://rocm.docs.amd.com/projects/HIP/en/latest/">HIP Documentation</a>
 */
public final class HipRuntime {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup HIP;
    private static final boolean AVAILABLE;

    // HIP error codes
    public static final int HIP_SUCCESS = 0;
    public static final int HIP_ERROR_INVALID_VALUE = 1;
    public static final int HIP_ERROR_OUT_OF_MEMORY = 2;
    public static final int HIP_ERROR_NOT_INITIALIZED = 3;
    public static final int HIP_ERROR_INVALID_DEVICE = 101;

    // HIP memcpy kinds
    public static final int HIP_MEMCPY_HOST_TO_HOST = 0;
    public static final int HIP_MEMCPY_HOST_TO_DEVICE = 1;
    public static final int HIP_MEMCPY_DEVICE_TO_HOST = 2;
    public static final int HIP_MEMCPY_DEVICE_TO_DEVICE = 3;
    public static final int HIP_MEMCPY_DEFAULT = 4;

    // HIP Runtime API function handles
    private static final MethodHandle hipInit;
    private static final MethodHandle hipGetDeviceCount;
    private static final MethodHandle hipSetDevice;
    private static final MethodHandle hipGetDevice;
    private static final MethodHandle hipDeviceSynchronize;
    private static final MethodHandle hipMalloc;
    private static final MethodHandle hipFree;
    private static final MethodHandle hipMemcpy;
    private static final MethodHandle hipMemcpyAsync;
    private static final MethodHandle hipModuleLoad;
    private static final MethodHandle hipModuleLoadData;
    private static final MethodHandle hipModuleUnload;
    private static final MethodHandle hipModuleGetFunction;
    private static final MethodHandle hipModuleLaunchKernel;
    private static final MethodHandle hipStreamCreate;
    private static final MethodHandle hipStreamDestroy;
    private static final MethodHandle hipStreamSynchronize;
    private static final MethodHandle hipGetErrorString;
    private static final MethodHandle hipGetErrorName;
    private static final MethodHandle hipEventCreate;
    private static final MethodHandle hipEventDestroy;
    private static final MethodHandle hipEventRecord;
    private static final MethodHandle hipEventSynchronize;
    private static final MethodHandle hipEventElapsedTime;

    static {
        SymbolLookup lookup = null;
        boolean available = false;

        try {
            // Try to load libamdhip64.so (Linux)
            lookup = SymbolLookup.libraryLookup("amdhip64", Arena.global());
            available = true;
        } catch (IllegalArgumentException e) {
            // Try common ROCm installation path
            try {
                lookup = SymbolLookup.libraryLookup("/opt/rocm/lib/libamdhip64.so", Arena.global());
                available = true;
            } catch (IllegalArgumentException e2) {
                // HIP not available
                lookup = SymbolLookup.loaderLookup();
            }
        }

        HIP = lookup;
        AVAILABLE = available;

        if (available) {
            // Initialization
            hipInit = downcall("hipInit", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

            // Device management
            hipGetDeviceCount = downcall("hipGetDeviceCount", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            hipSetDevice = downcall("hipSetDevice", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            hipGetDevice = downcall("hipGetDevice", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            hipDeviceSynchronize = downcall("hipDeviceSynchronize", FunctionDescriptor.of(
                ValueLayout.JAVA_INT));

            // Memory management
            hipMalloc = downcall("hipMalloc", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            hipFree = downcall("hipFree", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            hipMemcpy = downcall("hipMemcpy", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // dst
                ValueLayout.ADDRESS,    // src (can be host or device)
                ValueLayout.JAVA_LONG,  // size
                ValueLayout.JAVA_INT    // kind
            ));
            hipMemcpyAsync = downcall("hipMemcpyAsync", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // dst
                ValueLayout.ADDRESS,    // src
                ValueLayout.JAVA_LONG,  // size
                ValueLayout.JAVA_INT,   // kind
                ValueLayout.JAVA_LONG   // stream
            ));

            // Module/Kernel management
            hipModuleLoad = downcall("hipModuleLoad", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            hipModuleLoadData = downcall("hipModuleLoadData", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            hipModuleUnload = downcall("hipModuleUnload", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            hipModuleGetFunction = downcall("hipModuleGetFunction", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            hipModuleLaunchKernel = downcall("hipModuleLaunchKernel", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // function
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,  // grid dims
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,  // block dims
                ValueLayout.JAVA_INT,   // shared mem
                ValueLayout.JAVA_LONG,  // stream
                ValueLayout.ADDRESS,    // kernel params
                ValueLayout.ADDRESS     // extra
            ));

            // Stream management
            hipStreamCreate = downcall("hipStreamCreate", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            hipStreamDestroy = downcall("hipStreamDestroy", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            hipStreamSynchronize = downcall("hipStreamSynchronize", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));

            // Error handling
            hipGetErrorString = downcall("hipGetErrorString", FunctionDescriptor.of(
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            hipGetErrorName = downcall("hipGetErrorName", FunctionDescriptor.of(
                ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

            // Event management (for timing)
            hipEventCreate = downcall("hipEventCreate", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            hipEventDestroy = downcall("hipEventDestroy", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            hipEventRecord = downcall("hipEventRecord", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            hipEventSynchronize = downcall("hipEventSynchronize", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            hipEventElapsedTime = downcall("hipEventElapsedTime", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
        } else {
            hipInit = null;
            hipGetDeviceCount = null;
            hipSetDevice = null;
            hipGetDevice = null;
            hipDeviceSynchronize = null;
            hipMalloc = null;
            hipFree = null;
            hipMemcpy = null;
            hipMemcpyAsync = null;
            hipModuleLoad = null;
            hipModuleLoadData = null;
            hipModuleUnload = null;
            hipModuleGetFunction = null;
            hipModuleLaunchKernel = null;
            hipStreamCreate = null;
            hipStreamDestroy = null;
            hipStreamSynchronize = null;
            hipGetErrorString = null;
            hipGetErrorName = null;
            hipEventCreate = null;
            hipEventDestroy = null;
            hipEventRecord = null;
            hipEventSynchronize = null;
            hipEventElapsedTime = null;
        }
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        return HIP.find(name)
            .map(symbol -> LINKER.downcallHandle(symbol, descriptor))
            .orElse(null);
    }

    private HipRuntime() {}

    /**
     * Check if HIP is available on this system.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Ensure HIP is available, throwing if not.
     */
    public static void ensureAvailable() {
        if (!AVAILABLE) {
            throw new UnsupportedOperationException("HIP/ROCm is not available on this system");
        }
    }

    // ==================== Initialization ====================

    /**
     * Initialize HIP. Must be called before any other HIP functions.
     * @return hipSuccess (0) on success
     */
    public static int init() throws Throwable {
        ensureAvailable();
        return (int) hipInit.invokeExact(0);
    }

    // ==================== Device Management ====================

    /**
     * Get the number of HIP devices.
     */
    public static int getDeviceCount(Arena arena) throws Throwable {
        ensureAvailable();
        MemorySegment countPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) hipGetDeviceCount.invokeExact(countPtr);
        checkError(result, "hipGetDeviceCount");
        return countPtr.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Set the current HIP device.
     */
    public static void setDevice(int deviceIndex) throws Throwable {
        ensureAvailable();
        int result = (int) hipSetDevice.invokeExact(deviceIndex);
        checkError(result, "hipSetDevice");
    }

    /**
     * Get the current HIP device.
     */
    public static int getDevice(Arena arena) throws Throwable {
        ensureAvailable();
        MemorySegment devicePtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) hipGetDevice.invokeExact(devicePtr);
        checkError(result, "hipGetDevice");
        return devicePtr.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Synchronize the current device (wait for all operations to complete).
     */
    public static void synchronize() throws Throwable {
        ensureAvailable();
        int result = (int) hipDeviceSynchronize.invokeExact();
        checkError(result, "hipDeviceSynchronize");
    }

    // ==================== Memory Management ====================

    /**
     * Allocate device memory.
     * @param byteSize Number of bytes to allocate
     * @return Device pointer
     */
    public static long memAlloc(Arena arena, long byteSize) throws Throwable {
        ensureAvailable();
        MemorySegment ptrPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) hipMalloc.invokeExact(ptrPtr, byteSize);
        checkError(result, "hipMalloc");
        return ptrPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Free device memory.
     */
    public static void memFree(long devicePtr) throws Throwable {
        ensureAvailable();
        int result = (int) hipFree.invokeExact(devicePtr);
        checkError(result, "hipFree");
    }

    /**
     * Copy data from host to device.
     */
    public static void memcpyHtoD(long devicePtr, MemorySegment hostData, long byteSize) throws Throwable {
        ensureAvailable();
        int result = (int) hipMemcpy.invokeExact(devicePtr, hostData, byteSize, HIP_MEMCPY_HOST_TO_DEVICE);
        checkError(result, "hipMemcpy H2D");
    }

    /**
     * Copy data from device to host.
     */
    public static void memcpyDtoH(MemorySegment hostData, long devicePtr, long byteSize) throws Throwable {
        ensureAvailable();
        // For D2H, we need to handle the signature differently
        // hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind)
        int result = (int) hipMemcpy.invokeExact(
            hostData.address(),
            MemorySegment.ofAddress(devicePtr),
            byteSize,
            HIP_MEMCPY_DEVICE_TO_HOST
        );
        checkError(result, "hipMemcpy D2H");
    }

    // ==================== Module/Kernel Management ====================

    /**
     * Load a module from compiled binary (HSACO or fatbin).
     * @param arena Arena for allocations
     * @param filePath Path to the compiled module file
     * @return Module handle
     */
    public static long loadModule(Arena arena, String filePath) throws Throwable {
        ensureAvailable();
        MemorySegment modulePtr = arena.allocate(ValueLayout.JAVA_LONG);
        byte[] pathBytes = (filePath + "\0").getBytes(StandardCharsets.UTF_8);
        MemorySegment pathSegment = arena.allocate(pathBytes.length);
        pathSegment.copyFrom(MemorySegment.ofArray(pathBytes));
        int result = (int) hipModuleLoad.invokeExact(modulePtr, pathSegment);
        checkError(result, "hipModuleLoad");
        return modulePtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Load a module from in-memory data (compiled HSACO).
     * @param arena Arena for allocations
     * @param moduleData Compiled module binary data
     * @return Module handle
     */
    public static long loadModuleData(Arena arena, byte[] moduleData) throws Throwable {
        ensureAvailable();
        MemorySegment modulePtr = arena.allocate(ValueLayout.JAVA_LONG);
        MemorySegment dataSegment = arena.allocate(moduleData.length);
        dataSegment.copyFrom(MemorySegment.ofArray(moduleData));
        int result = (int) hipModuleLoadData.invokeExact(modulePtr, dataSegment);
        checkError(result, "hipModuleLoadData");
        return modulePtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Unload a module.
     */
    public static void unloadModule(long module) throws Throwable {
        ensureAvailable();
        int result = (int) hipModuleUnload.invokeExact(module);
        checkError(result, "hipModuleUnload");
    }

    /**
     * Get a kernel function from a loaded module.
     */
    public static long getFunction(Arena arena, long module, String functionName) throws Throwable {
        ensureAvailable();
        MemorySegment funcPtr = arena.allocate(ValueLayout.JAVA_LONG);
        byte[] nameBytes = (functionName + "\0").getBytes(StandardCharsets.UTF_8);
        MemorySegment nameSegment = arena.allocate(nameBytes.length);
        nameSegment.copyFrom(MemorySegment.ofArray(nameBytes));
        int result = (int) hipModuleGetFunction.invokeExact(funcPtr, module, nameSegment);
        checkError(result, "hipModuleGetFunction");
        return funcPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Launch a kernel.
     */
    public static void launchKernel(
            long function,
            int gridDimX, int gridDimY, int gridDimZ,
            int blockDimX, int blockDimY, int blockDimZ,
            int sharedMemBytes,
            long stream,
            MemorySegment kernelParams) throws Throwable {
        ensureAvailable();
        int result = (int) hipModuleLaunchKernel.invokeExact(
            function,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes,
            stream,
            kernelParams,
            MemorySegment.NULL
        );
        checkError(result, "hipModuleLaunchKernel");
    }

    // ==================== Stream Management ====================

    /**
     * Create a HIP stream.
     */
    public static long createStream(Arena arena) throws Throwable {
        ensureAvailable();
        MemorySegment streamPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) hipStreamCreate.invokeExact(streamPtr);
        checkError(result, "hipStreamCreate");
        return streamPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Destroy a HIP stream.
     */
    public static void destroyStream(long stream) throws Throwable {
        ensureAvailable();
        int result = (int) hipStreamDestroy.invokeExact(stream);
        checkError(result, "hipStreamDestroy");
    }

    /**
     * Synchronize a HIP stream.
     */
    public static void synchronizeStream(long stream) throws Throwable {
        ensureAvailable();
        int result = (int) hipStreamSynchronize.invokeExact(stream);
        checkError(result, "hipStreamSynchronize");
    }

    // ==================== Event Management (for JFR timing) ====================

    /**
     * Create a HIP event.
     */
    public static long eventCreate(Arena arena) throws Throwable {
        ensureAvailable();
        MemorySegment eventPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) hipEventCreate.invokeExact(eventPtr);
        checkError(result, "hipEventCreate");
        return eventPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Destroy a HIP event.
     */
    public static void eventDestroy(long event) throws Throwable {
        ensureAvailable();
        int result = (int) hipEventDestroy.invokeExact(event);
        checkError(result, "hipEventDestroy");
    }

    /**
     * Record an event on a stream.
     */
    public static void eventRecord(long event, long stream) throws Throwable {
        ensureAvailable();
        int result = (int) hipEventRecord.invokeExact(event, stream);
        checkError(result, "hipEventRecord");
    }

    /**
     * Synchronize (wait for) an event.
     */
    public static void eventSynchronize(long event) throws Throwable {
        ensureAvailable();
        int result = (int) hipEventSynchronize.invokeExact(event);
        checkError(result, "hipEventSynchronize");
    }

    /**
     * Get elapsed time between two events in milliseconds.
     */
    public static float eventElapsedTime(Arena arena, long startEvent, long stopEvent) throws Throwable {
        ensureAvailable();
        MemorySegment msPtr = arena.allocate(ValueLayout.JAVA_FLOAT);
        int result = (int) hipEventElapsedTime.invokeExact(msPtr, startEvent, stopEvent);
        checkError(result, "hipEventElapsedTime");
        return msPtr.get(ValueLayout.JAVA_FLOAT, 0);
    }

    // ==================== Error Handling ====================

    /**
     * Get a human-readable error message for a HIP error code.
     */
    public static String getErrorString(int errorCode) {
        if (!AVAILABLE || hipGetErrorString == null) {
            return "HIP_ERROR_" + errorCode;
        }
        try {
            MemorySegment strPtr = (MemorySegment) hipGetErrorString.invokeExact(errorCode);
            if (!strPtr.equals(MemorySegment.NULL)) {
                return strPtr.reinterpret(256).getString(0);
            }
        } catch (Throwable t) {
            // Ignore
        }
        return "HIP_ERROR_" + errorCode;
    }

    /**
     * Get the name of a HIP error code.
     */
    public static String getErrorName(int errorCode) {
        if (!AVAILABLE || hipGetErrorName == null) {
            return "HIP_ERROR_" + errorCode;
        }
        try {
            MemorySegment strPtr = (MemorySegment) hipGetErrorName.invokeExact(errorCode);
            if (!strPtr.equals(MemorySegment.NULL)) {
                return strPtr.reinterpret(256).getString(0);
            }
        } catch (Throwable t) {
            // Ignore
        }
        return "HIP_ERROR_" + errorCode;
    }

    /**
     * Check a HIP result and throw if not success.
     */
    public static void checkError(int result, String operation) {
        if (result != HIP_SUCCESS) {
            throw new HipException(operation + " failed: " + getErrorName(result) +
                " - " + getErrorString(result) + " (" + result + ")");
        }
    }

    /**
     * Exception thrown when a HIP operation fails.
     */
    public static class HipException extends RuntimeException {
        public HipException(String message) {
            super(message);
        }

        public HipException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
