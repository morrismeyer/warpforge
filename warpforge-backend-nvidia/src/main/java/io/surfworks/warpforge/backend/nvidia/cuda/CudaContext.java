package io.surfworks.warpforge.backend.nvidia.cuda;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Manages a CUDA context and provides convenient methods for device operations.
 *
 * <p>This class wraps the low-level CUDA driver API calls from {@link CudaRuntime}
 * and provides a higher-level interface for kernel execution.
 *
 * <p>Thread safety: A CudaContext should be created and used from a single thread,
 * or explicit synchronization should be used. CUDA contexts are bound to threads.
 */
public final class CudaContext implements AutoCloseable {

    private final int deviceIndex;
    private final long context;
    private final Arena arena;
    private final ConcurrentHashMap<String, Long> moduleCache;
    private final ConcurrentHashMap<String, Long> functionCache;
    private final AtomicBoolean closed;

    /**
     * Create a CUDA context for the default device (device 0).
     */
    public static CudaContext create() {
        return create(0);
    }

    /**
     * Create a CUDA context for the specified device.
     */
    public static CudaContext create(int deviceIndex) {
        CudaRuntime.ensureAvailable();
        try {
            CudaRuntime.init();
            Arena arena = Arena.ofShared();
            int device = CudaRuntime.getDevice(arena, deviceIndex);
            long context = CudaRuntime.createContext(arena, device, 0);
            return new CudaContext(deviceIndex, context, arena);
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to create CUDA context", t);
        }
    }

    private CudaContext(int deviceIndex, long context, Arena arena) {
        this.deviceIndex = deviceIndex;
        this.context = context;
        this.arena = arena;
        this.moduleCache = new ConcurrentHashMap<>();
        this.functionCache = new ConcurrentHashMap<>();
        this.closed = new AtomicBoolean(false);
    }

    /**
     * Get the device index this context is bound to.
     */
    public int deviceIndex() {
        return deviceIndex;
    }

    /**
     * Ensure this context is current on the calling thread.
     */
    public void makeCurrent() {
        checkNotClosed();
        try {
            CudaRuntime.setCurrentContext(context);
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to set current context", t);
        }
    }

    /**
     * Synchronize (wait for all operations to complete).
     */
    public void synchronize() {
        checkNotClosed();
        try {
            CudaRuntime.synchronize();
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to synchronize", t);
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
            return CudaRuntime.memAlloc(arena, byteSize);
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to allocate " + byteSize + " bytes", t);
        }
    }

    /**
     * Free device memory.
     */
    public void free(long devicePtr) {
        checkNotClosed();
        try {
            CudaRuntime.memFree(devicePtr);
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to free device memory", t);
        }
    }

    /**
     * Copy data from host to device.
     */
    public void copyToDevice(long devicePtr, MemorySegment hostData) {
        checkNotClosed();
        try {
            CudaRuntime.memcpyHtoD(devicePtr, hostData, hostData.byteSize());
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to copy to device", t);
        }
    }

    /**
     * Copy data from device to host.
     */
    public void copyToHost(MemorySegment hostData, long devicePtr, long byteSize) {
        checkNotClosed();
        try {
            CudaRuntime.memcpyDtoH(hostData, devicePtr, byteSize);
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to copy to host", t);
        }
    }

    // ==================== Kernel Management ====================

    /**
     * Load a PTX module and cache it by name.
     * @param name Unique name for caching
     * @param ptxSource PTX source code
     * @return Module handle
     */
    public long loadModule(String name, String ptxSource) {
        checkNotClosed();
        return moduleCache.computeIfAbsent(name, k -> {
            try {
                return CudaRuntime.loadModule(arena, ptxSource);
            } catch (Throwable t) {
                throw new CudaRuntime.CudaException("Failed to load module: " + name, t);
            }
        });
    }

    /**
     * Get a kernel function from a loaded module (cached).
     * @param module Module handle
     * @param functionName Kernel function name
     * @return Function handle
     */
    public long getFunction(long module, String functionName) {
        checkNotClosed();
        String cacheKey = module + ":" + functionName;
        return functionCache.computeIfAbsent(cacheKey, k -> {
            try {
                return CudaRuntime.getFunction(arena, module, functionName);
            } catch (Throwable t) {
                throw new CudaRuntime.CudaException("Failed to get function: " + functionName, t);
            }
        });
    }

    /**
     * Launch a kernel with the given configuration and parameters.
     *
     * @param function Kernel function handle
     * @param gridDim Grid dimensions [x, y, z]
     * @param blockDim Block dimensions [x, y, z]
     * @param sharedMem Shared memory size in bytes
     * @param params Kernel parameters (device pointers and scalars)
     */
    public void launchKernel(long function, int[] gridDim, int[] blockDim, int sharedMem, long... params) {
        checkNotClosed();
        try (Arena launchArena = Arena.ofConfined()) {
            // Build kernel parameter array (array of pointers to parameters)
            MemorySegment paramsArray = launchArena.allocate(
                ValueLayout.ADDRESS, params.length);
            MemorySegment[] paramPtrs = new MemorySegment[params.length];

            for (int i = 0; i < params.length; i++) {
                // Allocate space for this parameter and store its value
                MemorySegment paramPtr = launchArena.allocate(ValueLayout.JAVA_LONG);
                paramPtr.set(ValueLayout.JAVA_LONG, 0, params[i]);
                paramsArray.setAtIndex(ValueLayout.ADDRESS, i, paramPtr);
            }

            CudaRuntime.launchKernel(
                function,
                gridDim[0], gridDim.length > 1 ? gridDim[1] : 1, gridDim.length > 2 ? gridDim[2] : 1,
                blockDim[0], blockDim.length > 1 ? blockDim[1] : 1, blockDim.length > 2 ? blockDim[2] : 1,
                sharedMem,
                0, // default stream
                paramsArray
            );
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to launch kernel", t);
        }
    }

    /**
     * Launch a kernel with int parameters (for element count, etc.).
     */
    public void launchKernelWithIntParams(long function, int[] gridDim, int[] blockDim, int sharedMem,
                                          long[] devicePtrs, int... intParams) {
        checkNotClosed();
        try (Arena launchArena = Arena.ofConfined()) {
            int totalParams = devicePtrs.length + intParams.length;
            MemorySegment paramsArray = launchArena.allocate(ValueLayout.ADDRESS, totalParams);

            int idx = 0;
            // Add device pointer parameters
            for (long ptr : devicePtrs) {
                MemorySegment paramPtr = launchArena.allocate(ValueLayout.JAVA_LONG);
                paramPtr.set(ValueLayout.JAVA_LONG, 0, ptr);
                paramsArray.setAtIndex(ValueLayout.ADDRESS, idx++, paramPtr);
            }
            // Add int parameters
            for (int val : intParams) {
                MemorySegment paramPtr = launchArena.allocate(ValueLayout.JAVA_INT);
                paramPtr.set(ValueLayout.JAVA_INT, 0, val);
                paramsArray.setAtIndex(ValueLayout.ADDRESS, idx++, paramPtr);
            }

            CudaRuntime.launchKernel(
                function,
                gridDim[0], gridDim.length > 1 ? gridDim[1] : 1, gridDim.length > 2 ? gridDim[2] : 1,
                blockDim[0], blockDim.length > 1 ? blockDim[1] : 1, blockDim.length > 2 ? blockDim[2] : 1,
                sharedMem,
                0,
                paramsArray
            );
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to launch kernel", t);
        }
    }

    // ==================== Lifecycle ====================

    private void checkNotClosed() {
        if (closed.get()) {
            throw new IllegalStateException("CudaContext has been closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            // Unload all cached modules
            for (Long module : moduleCache.values()) {
                try {
                    CudaRuntime.unloadModule(module);
                } catch (Throwable t) {
                    // Ignore errors during cleanup
                }
            }
            moduleCache.clear();
            functionCache.clear();

            // Destroy context
            try {
                CudaRuntime.destroyContext(context);
            } catch (Throwable t) {
                // Ignore
            }

            arena.close();
        }
    }
}
