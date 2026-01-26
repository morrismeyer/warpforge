package io.surfworks.warpforge.backend.nvidia.cuda;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;

/**
 * FFM bindings to CUDA Driver API.
 *
 * <p>This class provides low-level access to CUDA functionality via Java's
 * Foreign Function & Memory API. We use the Driver API (cuXXX) rather than
 * the Runtime API (cudaXXX) because it provides explicit control over
 * module loading, which we need for loading PTX kernels.
 *
 * <p>Error handling: All CUDA functions return a CUresult status code.
 * A value of 0 (CUDA_SUCCESS) indicates success. Other values indicate
 * errors - use {@link #getErrorName(int)} to get a human-readable name.
 */
public final class CudaRuntime {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup CUDA;
    private static final boolean AVAILABLE;

    // CUDA Driver API function handles
    private static final MethodHandle cuInit;
    private static final MethodHandle cuDeviceGet;
    private static final MethodHandle cuDeviceGetCount;
    private static final MethodHandle cuCtxCreate;
    private static final MethodHandle cuCtxSetCurrent;
    private static final MethodHandle cuCtxDestroy;
    private static final MethodHandle cuMemAlloc;
    private static final MethodHandle cuMemFree;
    private static final MethodHandle cuMemcpyHtoD;
    private static final MethodHandle cuMemcpyDtoH;
    private static final MethodHandle cuModuleLoadData;
    private static final MethodHandle cuModuleUnload;
    private static final MethodHandle cuModuleGetFunction;
    private static final MethodHandle cuLaunchKernel;
    private static final MethodHandle cuCtxSynchronize;
    private static final MethodHandle cuGetErrorName;

    // CUDA Event API for GPU timing (JFR profiling)
    private static final MethodHandle cuEventCreate;
    private static final MethodHandle cuEventDestroy;
    private static final MethodHandle cuEventRecord;
    private static final MethodHandle cuEventSynchronize;
    private static final MethodHandle cuEventElapsedTime;

    static {
        SymbolLookup lookup = null;
        boolean available = false;

        try {
            // Try to load libcuda.so (Linux) or cuda.dll (Windows)
            // Note: Java FFM requires the full library name with extension
            String libraryName = System.getProperty("os.name").toLowerCase().contains("win")
                ? "nvcuda"  // Windows: nvcuda.dll
                : "libcuda.so";  // Linux/macOS: libcuda.so
            lookup = SymbolLookup.libraryLookup(libraryName, Arena.global());
            available = true;
        } catch (IllegalArgumentException e) {
            // CUDA not available
            lookup = SymbolLookup.loaderLookup(); // Fallback (won't have CUDA symbols)
        }

        CUDA = lookup;
        AVAILABLE = available;

        if (available) {
            cuInit = downcall("cuInit", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            cuDeviceGet = downcall("cuDeviceGet", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            cuDeviceGetCount = downcall("cuDeviceGetCount", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            cuCtxCreate = downcall("cuCtxCreate_v2", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            cuCtxSetCurrent = downcall("cuCtxSetCurrent", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            cuCtxDestroy = downcall("cuCtxDestroy_v2", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            cuMemAlloc = downcall("cuMemAlloc_v2", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            cuMemFree = downcall("cuMemFree_v2", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            cuMemcpyHtoD = downcall("cuMemcpyHtoD_v2", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            cuMemcpyDtoH = downcall("cuMemcpyDtoH_v2", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            cuModuleLoadData = downcall("cuModuleLoadData", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            cuModuleUnload = downcall("cuModuleUnload", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            cuModuleGetFunction = downcall("cuModuleGetFunction", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            cuLaunchKernel = downcall("cuLaunchKernel", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // function
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,  // grid dims
                ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,  // block dims
                ValueLayout.JAVA_INT,   // shared mem
                ValueLayout.JAVA_LONG,  // stream (0 = default)
                ValueLayout.ADDRESS,    // kernel params
                ValueLayout.ADDRESS     // extra (null)
            ));
            cuCtxSynchronize = downcall("cuCtxSynchronize", FunctionDescriptor.of(ValueLayout.JAVA_INT));
            cuGetErrorName = downcall("cuGetErrorName", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.ADDRESS));

            // Event API for GPU timing
            // CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags)
            cuEventCreate = downcall("cuEventCreate", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            // CUresult cuEventDestroy(CUevent hEvent)
            cuEventDestroy = downcall("cuEventDestroy_v2", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            // CUresult cuEventRecord(CUevent hEvent, CUstream hStream)
            cuEventRecord = downcall("cuEventRecord", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            // CUresult cuEventSynchronize(CUevent hEvent)
            cuEventSynchronize = downcall("cuEventSynchronize", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            // CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd)
            cuEventElapsedTime = downcall("cuEventElapsedTime", FunctionDescriptor.of(
                ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
        } else {
            cuInit = null;
            cuDeviceGet = null;
            cuDeviceGetCount = null;
            cuCtxCreate = null;
            cuCtxSetCurrent = null;
            cuCtxDestroy = null;
            cuMemAlloc = null;
            cuMemFree = null;
            cuMemcpyHtoD = null;
            cuMemcpyDtoH = null;
            cuModuleLoadData = null;
            cuModuleUnload = null;
            cuModuleGetFunction = null;
            cuLaunchKernel = null;
            cuCtxSynchronize = null;
            cuGetErrorName = null;
            cuEventCreate = null;
            cuEventDestroy = null;
            cuEventRecord = null;
            cuEventSynchronize = null;
            cuEventElapsedTime = null;
        }
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        return CUDA.find(name)
            .map(symbol -> LINKER.downcallHandle(symbol, descriptor))
            .orElse(null);
    }

    private CudaRuntime() {}

    /**
     * Check if CUDA is available on this system.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Ensure CUDA is available, throwing if not.
     */
    public static void ensureAvailable() {
        if (!AVAILABLE) {
            throw new UnsupportedOperationException("CUDA is not available on this system");
        }
    }

    // ==================== Initialization ====================

    /**
     * Initialize the CUDA driver. Must be called before any other CUDA functions.
     * @return CUDA_SUCCESS (0) on success
     */
    public static int init() throws Throwable {
        ensureAvailable();
        return (int) cuInit.invokeExact(0);
    }

    /**
     * Get the number of CUDA devices.
     */
    public static int getDeviceCount(Arena arena) throws Throwable {
        ensureAvailable();
        MemorySegment countPtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) cuDeviceGetCount.invokeExact(countPtr);
        checkError(result, "cuDeviceGetCount");
        return countPtr.get(ValueLayout.JAVA_INT, 0);
    }

    /**
     * Get a device handle.
     * @param ordinal Device index (0-based)
     * @return Device handle (CUdevice)
     */
    public static int getDevice(Arena arena, int ordinal) throws Throwable {
        ensureAvailable();
        MemorySegment devicePtr = arena.allocate(ValueLayout.JAVA_INT);
        int result = (int) cuDeviceGet.invokeExact(devicePtr, ordinal);
        checkError(result, "cuDeviceGet");
        return devicePtr.get(ValueLayout.JAVA_INT, 0);
    }

    // ==================== Context Management ====================

    /**
     * Create a CUDA context for the given device.
     * @param device Device handle from getDevice()
     * @param flags Context creation flags (0 for default)
     * @return Context handle (CUcontext)
     */
    public static long createContext(Arena arena, int device, int flags) throws Throwable {
        ensureAvailable();
        MemorySegment ctxPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) cuCtxCreate.invokeExact(ctxPtr, flags, device);
        checkError(result, "cuCtxCreate");
        return ctxPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Set the current context for this thread.
     */
    public static void setCurrentContext(long context) throws Throwable {
        ensureAvailable();
        int result = (int) cuCtxSetCurrent.invokeExact(context);
        checkError(result, "cuCtxSetCurrent");
    }

    /**
     * Destroy a CUDA context.
     */
    public static void destroyContext(long context) throws Throwable {
        ensureAvailable();
        int result = (int) cuCtxDestroy.invokeExact(context);
        checkError(result, "cuCtxDestroy");
    }

    /**
     * Synchronize the current context (wait for all operations to complete).
     */
    public static void synchronize() throws Throwable {
        ensureAvailable();
        int result = (int) cuCtxSynchronize.invokeExact();
        checkError(result, "cuCtxSynchronize");
    }

    // ==================== Memory Management ====================

    /**
     * Allocate device memory.
     * @param byteSize Number of bytes to allocate
     * @return Device pointer (CUdeviceptr)
     */
    public static long memAlloc(Arena arena, long byteSize) throws Throwable {
        ensureAvailable();
        MemorySegment ptrPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) cuMemAlloc.invokeExact(ptrPtr, byteSize);
        checkError(result, "cuMemAlloc");
        return ptrPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Free device memory.
     */
    public static void memFree(long devicePtr) throws Throwable {
        ensureAvailable();
        int result = (int) cuMemFree.invokeExact(devicePtr);
        checkError(result, "cuMemFree");
    }

    /**
     * Copy data from host to device.
     */
    public static void memcpyHtoD(long devicePtr, MemorySegment hostData, long byteSize) throws Throwable {
        ensureAvailable();
        int result = (int) cuMemcpyHtoD.invokeExact(devicePtr, hostData, byteSize);
        checkError(result, "cuMemcpyHtoD");
    }

    /**
     * Copy data from device to host.
     */
    public static void memcpyDtoH(MemorySegment hostData, long devicePtr, long byteSize) throws Throwable {
        ensureAvailable();
        int result = (int) cuMemcpyDtoH.invokeExact(hostData, devicePtr, byteSize);
        checkError(result, "cuMemcpyDtoH");
    }

    // ==================== Module/Kernel Management ====================

    /**
     * Load a module from PTX source.
     * @param ptxSource PTX source code as a string
     * @return Module handle (CUmodule)
     */
    public static long loadModule(Arena arena, String ptxSource) throws Throwable {
        ensureAvailable();
        MemorySegment modulePtr = arena.allocate(ValueLayout.JAVA_LONG);
        byte[] ptxBytes = (ptxSource + "\0").getBytes(StandardCharsets.UTF_8);
        MemorySegment ptxSegment = arena.allocate(ptxBytes.length);
        ptxSegment.copyFrom(MemorySegment.ofArray(ptxBytes));
        int result = (int) cuModuleLoadData.invokeExact(modulePtr, ptxSegment);
        checkError(result, "cuModuleLoadData");
        return modulePtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Unload a module.
     */
    public static void unloadModule(long module) throws Throwable {
        ensureAvailable();
        int result = (int) cuModuleUnload.invokeExact(module);
        checkError(result, "cuModuleUnload");
    }

    /**
     * Get a kernel function from a loaded module.
     * @param module Module handle
     * @param functionName Name of the kernel function
     * @return Function handle (CUfunction)
     */
    public static long getFunction(Arena arena, long module, String functionName) throws Throwable {
        ensureAvailable();
        MemorySegment funcPtr = arena.allocate(ValueLayout.JAVA_LONG);
        byte[] nameBytes = (functionName + "\0").getBytes(StandardCharsets.UTF_8);
        MemorySegment nameSegment = arena.allocate(nameBytes.length);
        nameSegment.copyFrom(MemorySegment.ofArray(nameBytes));
        int result = (int) cuModuleGetFunction.invokeExact(funcPtr, module, nameSegment);
        checkError(result, "cuModuleGetFunction");
        return funcPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Launch a kernel.
     *
     * @param function Kernel function handle
     * @param gridDimX Grid dimension X
     * @param gridDimY Grid dimension Y
     * @param gridDimZ Grid dimension Z
     * @param blockDimX Block dimension X
     * @param blockDimY Block dimension Y
     * @param blockDimZ Block dimension Z
     * @param sharedMemBytes Shared memory size per block
     * @param stream CUDA stream (0 for default)
     * @param kernelParams Array of kernel parameter pointers
     */
    public static void launchKernel(
            long function,
            int gridDimX, int gridDimY, int gridDimZ,
            int blockDimX, int blockDimY, int blockDimZ,
            int sharedMemBytes,
            long stream,
            MemorySegment kernelParams) throws Throwable {
        ensureAvailable();
        int result = (int) cuLaunchKernel.invokeExact(
            function,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes,
            stream,
            kernelParams,
            MemorySegment.NULL
        );
        checkError(result, "cuLaunchKernel");
    }

    // ==================== Event Management (GPU Timing) ====================

    /**
     * Event creation flags.
     */
    public static final int CU_EVENT_DEFAULT = 0;
    public static final int CU_EVENT_BLOCKING_SYNC = 1;
    public static final int CU_EVENT_DISABLE_TIMING = 2;

    /**
     * Create a CUDA event for GPU timing.
     *
     * @param arena Arena for memory allocation
     * @param flags Event creation flags (use CU_EVENT_DEFAULT for timing)
     * @return Event handle (CUevent)
     */
    public static long eventCreate(Arena arena, int flags) throws Throwable {
        ensureAvailable();
        MemorySegment eventPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) cuEventCreate.invokeExact(eventPtr, flags);
        checkError(result, "cuEventCreate");
        return eventPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Create a CUDA event with default flags (timing enabled).
     *
     * @param arena Arena for memory allocation
     * @return Event handle (CUevent)
     */
    public static long eventCreate(Arena arena) throws Throwable {
        return eventCreate(arena, CU_EVENT_DEFAULT);
    }

    /**
     * Destroy a CUDA event.
     *
     * @param event Event handle to destroy
     */
    public static void eventDestroy(long event) throws Throwable {
        ensureAvailable();
        int result = (int) cuEventDestroy.invokeExact(event);
        checkError(result, "cuEventDestroy");
    }

    /**
     * Record an event on a stream.
     *
     * @param event Event handle
     * @param stream Stream handle (0 for default stream)
     */
    public static void eventRecord(long event, long stream) throws Throwable {
        ensureAvailable();
        int result = (int) cuEventRecord.invokeExact(event, stream);
        checkError(result, "cuEventRecord");
    }

    /**
     * Record an event on the default stream.
     *
     * @param event Event handle
     */
    public static void eventRecord(long event) throws Throwable {
        eventRecord(event, 0L);
    }

    /**
     * Wait for an event to complete.
     *
     * @param event Event handle
     */
    public static void eventSynchronize(long event) throws Throwable {
        ensureAvailable();
        int result = (int) cuEventSynchronize.invokeExact(event);
        checkError(result, "cuEventSynchronize");
    }

    /**
     * Compute elapsed time between two events in milliseconds.
     *
     * <p>Both events must have been recorded and completed before calling this.
     *
     * @param arena Arena for memory allocation
     * @param start Start event handle
     * @param end End event handle
     * @return Elapsed time in milliseconds
     */
    public static float eventElapsedTime(Arena arena, long start, long end) throws Throwable {
        ensureAvailable();
        MemorySegment msPtr = arena.allocate(ValueLayout.JAVA_FLOAT);
        int result = (int) cuEventElapsedTime.invokeExact(msPtr, start, end);
        checkError(result, "cuEventElapsedTime");
        return msPtr.get(ValueLayout.JAVA_FLOAT, 0);
    }

    /**
     * Time a GPU operation using events.
     *
     * <p>This is a convenience method that creates events, records before/after
     * the operation, and returns the elapsed time. The events are destroyed
     * after use.
     *
     * <p>Example usage:
     * <pre>{@code
     * try (Arena arena = Arena.ofConfined()) {
     *     float ms = CudaRuntime.timeOperation(arena, () -> {
     *         CudaRuntime.launchKernel(...);
     *     });
     *     System.out.println("Kernel took " + ms + " ms");
     * }
     * }</pre>
     *
     * @param arena Arena for memory allocation
     * @param operation The GPU operation to time
     * @return Elapsed time in milliseconds
     */
    public static float timeOperation(Arena arena, ThrowingRunnable operation) throws Throwable {
        long start = eventCreate(arena);
        long end = eventCreate(arena);
        try {
            eventRecord(start);
            operation.run();
            eventRecord(end);
            eventSynchronize(end);
            return eventElapsedTime(arena, start, end);
        } finally {
            eventDestroy(start);
            eventDestroy(end);
        }
    }

    /**
     * Functional interface for operations that may throw.
     */
    @FunctionalInterface
    public interface ThrowingRunnable {
        void run() throws Throwable;
    }

    // ==================== Error Handling ====================

    /**
     * Get the name of a CUDA error code.
     */
    public static String getErrorName(int errorCode) {
        if (!AVAILABLE || cuGetErrorName == null) {
            return "CUDA_ERROR_" + errorCode;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment namePtr = arena.allocate(ValueLayout.ADDRESS);
            int result = (int) cuGetErrorName.invokeExact(errorCode, namePtr);
            if (result == 0) {
                MemorySegment nameSegment = namePtr.get(ValueLayout.ADDRESS, 0);
                return nameSegment.reinterpret(256).getString(0);
            }
        } catch (Throwable t) {
            // Ignore
        }
        return "CUDA_ERROR_" + errorCode;
    }

    /**
     * Check a CUDA result and throw if not success.
     */
    public static void checkError(int result, String operation) {
        if (result != 0) {
            throw new CudaException(operation + " failed: " + getErrorName(result) + " (" + result + ")");
        }
    }

    /**
     * Exception thrown when a CUDA operation fails.
     */
    public static class CudaException extends RuntimeException {
        public CudaException(String message) {
            super(message);
        }

        public CudaException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
