package io.surfworks.warpforge.backend.amd.hip;

import io.surfworks.warpforge.backend.amd.rocblas.RocblasRuntime;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Manages a HIP device context and provides convenient methods for device operations.
 *
 * <p>This class wraps the low-level HIP runtime API calls from {@link HipRuntime}
 * and provides a higher-level interface for kernel execution.
 *
 * <p>Unlike CUDA which has explicit context objects, HIP uses a simpler device-centric
 * model where you set the current device and operations apply to it.
 */
public final class HipContext implements AutoCloseable {

    private final int deviceIndex;
    private final Arena arena;
    private final ConcurrentHashMap<String, Long> moduleCache;
    private final ConcurrentHashMap<String, Long> functionCache;
    private final AtomicBoolean closed;
    private final AtomicLong rocblasHandle;  // Lazily initialized rocBLAS handle

    /**
     * Create a HIP context for the default device (device 0).
     */
    public static HipContext create() {
        return create(0);
    }

    /**
     * Create a HIP context for the specified device.
     */
    public static HipContext create(int deviceIndex) {
        HipRuntime.ensureAvailable();
        try {
            HipRuntime.init();
            HipRuntime.setDevice(deviceIndex);
            Arena arena = Arena.ofShared();
            return new HipContext(deviceIndex, arena);
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to create HIP context", t);
        }
    }

    private HipContext(int deviceIndex, Arena arena) {
        this.deviceIndex = deviceIndex;
        this.arena = arena;
        this.moduleCache = new ConcurrentHashMap<>();
        this.functionCache = new ConcurrentHashMap<>();
        this.closed = new AtomicBoolean(false);
        this.rocblasHandle = new AtomicLong(0);  // 0 = not initialized
    }

    /**
     * Get the device index this context is bound to.
     */
    public int deviceIndex() {
        return deviceIndex;
    }

    /**
     * Ensure this device is current.
     */
    public void makeCurrent() {
        checkNotClosed();
        try {
            HipRuntime.setDevice(deviceIndex);
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to set current device", t);
        }
    }

    /**
     * Synchronize (wait for all operations to complete).
     */
    public void synchronize() {
        checkNotClosed();
        try {
            HipRuntime.synchronize();
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to synchronize", t);
        }
    }

    // ==================== Memory Operations ====================

    /**
     * Allocate device memory.
     * @return Device pointer
     */
    public long allocate(long byteSize) {
        checkNotClosed();
        try {
            return HipRuntime.memAlloc(arena, byteSize);
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to allocate " + byteSize + " bytes", t);
        }
    }

    /**
     * Free device memory.
     */
    public void free(long devicePtr) {
        checkNotClosed();
        try {
            HipRuntime.memFree(devicePtr);
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to free device memory", t);
        }
    }

    /**
     * Copy data from host to device.
     */
    public void copyToDevice(long devicePtr, MemorySegment hostData) {
        checkNotClosed();
        try {
            HipRuntime.memcpyHtoD(devicePtr, hostData, hostData.byteSize());
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to copy to device", t);
        }
    }

    /**
     * Copy data from device to host.
     */
    public void copyToHost(MemorySegment hostData, long devicePtr, long byteSize) {
        checkNotClosed();
        try {
            HipRuntime.memcpyDtoH(hostData, devicePtr, byteSize);
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to copy to host", t);
        }
    }

    // ==================== Kernel Management ====================

    /**
     * Compile HIP C++ source code and load it as a module.
     *
     * <p>This method uses HIPRTC to compile the source at runtime, then loads
     * the resulting binary via hipModuleLoadData. Results are cached by name.
     *
     * @param name Unique name for caching (used as program name in HIPRTC)
     * @param source HIP C++ source code
     * @param options Compiler options (e.g., "-O3")
     * @return Module handle
     */
    public long compileAndLoadModule(String name, String source, String... options) {
        checkNotClosed();
        return moduleCache.computeIfAbsent(name, k -> {
            HiprtcRuntime.ensureAvailable();
            try {
                byte[] hsaco = HiprtcRuntime.compile(source, name, options);
                return HipRuntime.loadModuleData(arena, hsaco);
            } catch (HiprtcRuntime.HiprtcException e) {
                throw new HipRuntime.HipException("Failed to compile and load module: " + name, e);
            } catch (Throwable t) {
                throw new HipRuntime.HipException("Failed to load compiled module: " + name, t);
            }
        });
    }

    /**
     * Load a module from compiled binary and cache it by name.
     * @param name Unique name for caching
     * @param moduleData Compiled HSACO binary
     * @return Module handle
     */
    public long loadModule(String name, byte[] moduleData) {
        checkNotClosed();
        return moduleCache.computeIfAbsent(name, k -> {
            try {
                return HipRuntime.loadModuleData(arena, moduleData);
            } catch (Throwable t) {
                throw new HipRuntime.HipException("Failed to load module: " + name, t);
            }
        });
    }

    /**
     * Get a kernel function from a loaded module (cached).
     */
    public long getFunction(long module, String functionName) {
        checkNotClosed();
        String cacheKey = module + ":" + functionName;
        return functionCache.computeIfAbsent(cacheKey, k -> {
            try {
                return HipRuntime.getFunction(arena, module, functionName);
            } catch (Throwable t) {
                throw new HipRuntime.HipException("Failed to get function: " + functionName, t);
            }
        });
    }

    /**
     * Launch a kernel with the given configuration and parameters.
     */
    public void launchKernel(long function, int[] gridDim, int[] blockDim, int sharedMem, long... params) {
        checkNotClosed();
        try (Arena launchArena = Arena.ofConfined()) {
            MemorySegment paramsArray = launchArena.allocate(ValueLayout.ADDRESS, params.length);

            for (int i = 0; i < params.length; i++) {
                MemorySegment paramPtr = launchArena.allocate(ValueLayout.JAVA_LONG);
                paramPtr.set(ValueLayout.JAVA_LONG, 0, params[i]);
                paramsArray.setAtIndex(ValueLayout.ADDRESS, i, paramPtr);
            }

            HipRuntime.launchKernel(
                function,
                gridDim[0], gridDim.length > 1 ? gridDim[1] : 1, gridDim.length > 2 ? gridDim[2] : 1,
                blockDim[0], blockDim.length > 1 ? blockDim[1] : 1, blockDim.length > 2 ? blockDim[2] : 1,
                sharedMem,
                0, // default stream
                paramsArray
            );
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to launch kernel", t);
        }
    }

    /**
     * Launch a kernel with int parameters.
     */
    public void launchKernelWithIntParams(long function, int[] gridDim, int[] blockDim, int sharedMem,
                                          long[] devicePtrs, int... intParams) {
        checkNotClosed();
        try (Arena launchArena = Arena.ofConfined()) {
            int totalParams = devicePtrs.length + intParams.length;
            MemorySegment paramsArray = launchArena.allocate(ValueLayout.ADDRESS, totalParams);

            int idx = 0;
            for (long ptr : devicePtrs) {
                MemorySegment paramPtr = launchArena.allocate(ValueLayout.JAVA_LONG);
                paramPtr.set(ValueLayout.JAVA_LONG, 0, ptr);
                paramsArray.setAtIndex(ValueLayout.ADDRESS, idx++, paramPtr);
            }
            for (int val : intParams) {
                MemorySegment paramPtr = launchArena.allocate(ValueLayout.JAVA_INT);
                paramPtr.set(ValueLayout.JAVA_INT, 0, val);
                paramsArray.setAtIndex(ValueLayout.ADDRESS, idx++, paramPtr);
            }

            HipRuntime.launchKernel(
                function,
                gridDim[0], gridDim.length > 1 ? gridDim[1] : 1, gridDim.length > 2 ? gridDim[2] : 1,
                blockDim[0], blockDim.length > 1 ? blockDim[1] : 1, blockDim.length > 2 ? blockDim[2] : 1,
                sharedMem,
                0,
                paramsArray
            );
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to launch kernel", t);
        }
    }

    // ==================== rocBLAS Operations ====================

    /**
     * Get the rocBLAS handle for this context, creating it lazily if needed.
     */
    public long getRocblasHandle() {
        checkNotClosed();
        long handle = rocblasHandle.get();
        if (handle == 0) {
            synchronized (this) {
                handle = rocblasHandle.get();
                if (handle == 0) {
                    try {
                        handle = RocblasRuntime.create(arena);
                        rocblasHandle.set(handle);
                    } catch (Throwable t) {
                        throw new HipRuntime.HipException("Failed to create rocBLAS handle", t);
                    }
                }
            }
        }
        return handle;
    }

    /**
     * Check if rocBLAS is available.
     */
    public boolean isRocblasAvailable() {
        return RocblasRuntime.isAvailable();
    }

    /**
     * Perform single-precision matrix multiplication using rocBLAS.
     *
     * <p>Computes C = A * B where A is MxK and B is KxN, producing MxN result.
     * Input matrices are assumed to be in row-major order.
     */
    public void sgemm(long dA, long dB, long dC, int M, int N, int K) {
        sgemmWithAlphaBeta(dA, dB, dC, M, N, K, 1.0f, 0.0f);
    }

    /**
     * Perform single-precision matrix multiplication with alpha and beta scalars.
     *
     * <p>Computes C = alpha * A * B + beta * C
     */
    public void sgemmWithAlphaBeta(long dA, long dB, long dC, int M, int N, int K,
                                    float alpha, float beta) {
        checkNotClosed();
        long handle = getRocblasHandle();

        try (Arena gemmArena = Arena.ofConfined()) {
            // rocBLAS uses column-major, same trick as cuBLAS for row-major data
            RocblasRuntime.sgemm(gemmArena, handle,
                RocblasRuntime.ROCBLAS_OP_N, RocblasRuntime.ROCBLAS_OP_N,
                N, M, K,
                alpha,
                dB, N,
                dA, K,
                beta,
                dC, N
            );
        } catch (Throwable t) {
            throw new HipRuntime.HipException("rocBLAS sgemm failed", t);
        }
    }

    // ==================== Event-Based GPU Timing (JFR Profiling) ====================

    /**
     * Create a HIP event for GPU timing.
     *
     * @return Event handle
     */
    public long createEvent() {
        checkNotClosed();
        try {
            return HipRuntime.eventCreate(arena);
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to create HIP event", t);
        }
    }

    /**
     * Destroy a HIP event.
     *
     * @param event Event handle to destroy
     */
    public void destroyEvent(long event) {
        checkNotClosed();
        try {
            HipRuntime.eventDestroy(event);
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to destroy HIP event", t);
        }
    }

    /**
     * Record an event on the default stream.
     *
     * @param event Event handle
     */
    public void recordEvent(long event) {
        checkNotClosed();
        try {
            HipRuntime.eventRecord(event, 0); // 0 = default stream
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to record HIP event", t);
        }
    }

    /**
     * Wait for an event to complete.
     *
     * @param event Event handle
     */
    public void synchronizeEvent(long event) {
        checkNotClosed();
        try {
            HipRuntime.eventSynchronize(event);
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to synchronize HIP event", t);
        }
    }

    /**
     * Compute elapsed time between two events in milliseconds.
     *
     * @param start Start event handle
     * @param end End event handle
     * @return Elapsed time in milliseconds
     */
    public float elapsedTime(long start, long end) {
        checkNotClosed();
        try {
            return HipRuntime.eventElapsedTime(arena, start, end);
        } catch (Throwable t) {
            throw new HipRuntime.HipException("Failed to compute elapsed time", t);
        }
    }

    /**
     * Time a GPU operation and return elapsed time in milliseconds.
     *
     * <p>Example:
     * <pre>{@code
     * float ms = ctx.timeOperation(() -> {
     *     ctx.launchKernel(function, gridDim, blockDim, 0, params);
     * });
     * }</pre>
     *
     * @param operation The GPU operation to time
     * @return Elapsed time in milliseconds
     */
    public float timeOperation(Runnable operation) {
        checkNotClosed();
        long start = createEvent();
        long end = createEvent();
        try {
            recordEvent(start);
            operation.run();
            recordEvent(end);
            synchronizeEvent(end);
            return elapsedTime(start, end);
        } finally {
            destroyEvent(start);
            destroyEvent(end);
        }
    }

    // ==================== Lifecycle ====================

    private void checkNotClosed() {
        if (closed.get()) {
            throw new IllegalStateException("HipContext has been closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            // Destroy rocBLAS handle if it was created
            long handle = rocblasHandle.get();
            if (handle != 0) {
                try {
                    RocblasRuntime.destroy(handle);
                } catch (Throwable t) {
                    // Ignore errors during cleanup
                }
                rocblasHandle.set(0);
            }

            // Unload all cached modules
            for (Long module : moduleCache.values()) {
                try {
                    HipRuntime.unloadModule(module);
                } catch (Throwable t) {
                    // Ignore errors during cleanup
                }
            }
            moduleCache.clear();
            functionCache.clear();

            arena.close();
        }
    }
}
