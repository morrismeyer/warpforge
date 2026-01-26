package io.surfworks.warpforge.backend.nvidia.cuda;

import io.surfworks.warpforge.backend.nvidia.cublas.CublasRuntime;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

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
    private final AtomicLong cublasHandle;  // Lazily initialized cuBLAS handle

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
        this.cublasHandle = new AtomicLong(0);  // 0 = not initialized
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

    /**
     * Launch a kernel with int and float parameters.
     *
     * <p>Parameters are passed in order: device pointers, then ints, then floats.
     */
    public void launchKernelWithMixedParams(long function, int[] gridDim, int[] blockDim, int sharedMem,
                                            long[] devicePtrs, int[] intParams, float[] floatParams) {
        launchKernelWithMixedParams(function, gridDim, blockDim, sharedMem, devicePtrs, intParams, floatParams, null);
    }

    /**
     * Launch a kernel with int, float, and optional trailing pointer parameters.
     *
     * <p>Parameters are passed in order: device pointers, then ints, then floats, then trailing pointers.
     * This supports PTX kernels where timing pointers come at the end of the parameter list.
     */
    public void launchKernelWithMixedParams(long function, int[] gridDim, int[] blockDim, int sharedMem,
                                            long[] devicePtrs, int[] intParams, float[] floatParams,
                                            long[] trailingPtrs) {
        checkNotClosed();
        try (Arena launchArena = Arena.ofConfined()) {
            int totalParams = devicePtrs.length + intParams.length + floatParams.length
                + (trailingPtrs != null ? trailingPtrs.length : 0);
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
            // Add float parameters
            for (float val : floatParams) {
                MemorySegment paramPtr = launchArena.allocate(ValueLayout.JAVA_FLOAT);
                paramPtr.set(ValueLayout.JAVA_FLOAT, 0, val);
                paramsArray.setAtIndex(ValueLayout.ADDRESS, idx++, paramPtr);
            }
            // Add trailing pointer parameters (e.g., timing_ptr)
            if (trailingPtrs != null) {
                for (long ptr : trailingPtrs) {
                    MemorySegment paramPtr = launchArena.allocate(ValueLayout.JAVA_LONG);
                    paramPtr.set(ValueLayout.JAVA_LONG, 0, ptr);
                    paramsArray.setAtIndex(ValueLayout.ADDRESS, idx++, paramPtr);
                }
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

    // ==================== cuBLAS Operations ====================

    /**
     * Get the cuBLAS handle for this context, creating it lazily if needed.
     *
     * <p>The cuBLAS handle is created on first use and destroyed when the
     * context is closed.
     *
     * @return cuBLAS handle
     */
    public long getCublasHandle() {
        checkNotClosed();
        long handle = cublasHandle.get();
        if (handle == 0) {
            synchronized (this) {
                handle = cublasHandle.get();
                if (handle == 0) {
                    try {
                        handle = CublasRuntime.create(arena);
                        cublasHandle.set(handle);
                    } catch (Throwable t) {
                        throw new CudaRuntime.CudaException("Failed to create cuBLAS handle", t);
                    }
                }
            }
        }
        return handle;
    }

    /**
     * Check if cuBLAS is available.
     */
    public boolean isCublasAvailable() {
        return CublasRuntime.isAvailable();
    }

    /**
     * Perform single-precision matrix multiplication using cuBLAS.
     *
     * <p>Computes C = A * B where A is MxK and B is KxN, producing MxN result.
     *
     * <p>This method handles the row-major to column-major conversion automatically.
     * Input matrices are assumed to be in row-major order (C-style), and the output
     * will also be in row-major order.
     *
     * @param dA Device pointer to matrix A (MxK, row-major)
     * @param dB Device pointer to matrix B (KxN, row-major)
     * @param dC Device pointer to result matrix C (MxN, row-major)
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A and rows in B
     */
    public void sgemm(long dA, long dB, long dC, int M, int N, int K) {
        sgemmWithAlphaBeta(dA, dB, dC, M, N, K, 1.0f, 0.0f);
    }

    /**
     * Perform single-precision matrix multiplication with alpha and beta scalars.
     *
     * <p>Computes C = alpha * A * B + beta * C
     *
     * @param dA Device pointer to matrix A (MxK, row-major)
     * @param dB Device pointer to matrix B (KxN, row-major)
     * @param dC Device pointer to result matrix C (MxN, row-major)
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A and rows in B
     * @param alpha Scalar multiplier for A*B
     * @param beta Scalar multiplier for C
     */
    public void sgemmWithAlphaBeta(long dA, long dB, long dC, int M, int N, int K,
                                    float alpha, float beta) {
        checkNotClosed();
        long handle = getCublasHandle();

        try (Arena gemmArena = Arena.ofConfined()) {
            // cuBLAS uses column-major. For row-major C = A * B:
            // We compute C^T = B^T * A^T using cuBLAS, which gives us C in row-major.
            // In cuBLAS terms: C(col-major) = B * A with dimensions (N,M) = (N,K) * (K,M)
            // But since our data is row-major, we can treat:
            //   A_row[M,K] as A_col[K,M] (transposed view)
            //   B_row[K,N] as B_col[N,K] (transposed view)
            // So: C_col[N,M] = B_col[N,K] * A_col[K,M]
            // Which means: C_row[M,N] = A_row[M,K] * B_row[K,N] (what we want!)
            //
            // Call: sgemm(OP_N, OP_N, N, M, K, alpha, B, N, A, K, beta, C, N)
            CublasRuntime.sgemm(gemmArena, handle,
                CublasRuntime.CUBLAS_OP_N, CublasRuntime.CUBLAS_OP_N,
                N, M, K,
                alpha,
                dB, N,   // B is KxN row-major, treat as NxK col-major
                dA, K,   // A is MxK row-major, treat as KxM col-major
                beta,
                dC, N    // C is MxN row-major, treat as NxM col-major
            );
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("cuBLAS sgemm failed", t);
        }
    }

    /**
     * Perform double-precision matrix multiplication using cuBLAS.
     *
     * @see #sgemm(long, long, long, int, int, int)
     */
    public void dgemm(long dA, long dB, long dC, int M, int N, int K) {
        dgemmWithAlphaBeta(dA, dB, dC, M, N, K, 1.0, 0.0);
    }

    /**
     * Perform double-precision matrix multiplication with alpha and beta scalars.
     *
     * @see #sgemmWithAlphaBeta(long, long, long, int, int, int, float, float)
     */
    public void dgemmWithAlphaBeta(long dA, long dB, long dC, int M, int N, int K,
                                    double alpha, double beta) {
        checkNotClosed();
        long handle = getCublasHandle();

        try (Arena gemmArena = Arena.ofConfined()) {
            // Same row-major to column-major trick as sgemm
            CublasRuntime.dgemm(gemmArena, handle,
                CublasRuntime.CUBLAS_OP_N, CublasRuntime.CUBLAS_OP_N,
                N, M, K,
                alpha,
                dB, N,
                dA, K,
                beta,
                dC, N
            );
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("cuBLAS dgemm failed", t);
        }
    }

    // ==================== Event-Based GPU Timing (JFR Profiling) ====================

    /**
     * Create a CUDA event for GPU timing.
     *
     * @return Event handle
     */
    public long createEvent() {
        checkNotClosed();
        try {
            return CudaRuntime.eventCreate(arena);
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to create CUDA event", t);
        }
    }

    /**
     * Destroy a CUDA event.
     *
     * @param event Event handle to destroy
     */
    public void destroyEvent(long event) {
        checkNotClosed();
        try {
            CudaRuntime.eventDestroy(event);
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to destroy CUDA event", t);
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
            CudaRuntime.eventRecord(event);
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to record CUDA event", t);
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
            CudaRuntime.eventSynchronize(event);
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to synchronize CUDA event", t);
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
            return CudaRuntime.eventElapsedTime(arena, start, end);
        } catch (Throwable t) {
            throw new CudaRuntime.CudaException("Failed to compute elapsed time", t);
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
            throw new IllegalStateException("CudaContext has been closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            // Destroy cuBLAS handle if it was created
            long handle = cublasHandle.get();
            if (handle != 0) {
                try {
                    CublasRuntime.destroy(handle);
                } catch (Throwable t) {
                    // Ignore errors during cleanup
                }
                cublasHandle.set(0);
            }

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
