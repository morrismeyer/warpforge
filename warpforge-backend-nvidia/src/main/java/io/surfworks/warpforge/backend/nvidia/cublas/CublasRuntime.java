package io.surfworks.warpforge.backend.nvidia.cublas;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * FFM bindings to cuBLAS library.
 *
 * <p>This class provides access to NVIDIA's cuBLAS library for high-performance
 * linear algebra operations. cuBLAS provides optimized BLAS routines that run
 * on NVIDIA GPUs.
 *
 * <p>cuBLAS uses column-major storage by default (Fortran-style). For row-major
 * data (C-style), transpose flags can be used or matrices can be swapped:
 * C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major.
 *
 * <p>Error handling: cuBLAS functions return cublasStatus_t. A value of 0
 * (CUBLAS_STATUS_SUCCESS) indicates success.
 *
 * @see <a href="https://docs.nvidia.com/cuda/cublas/">cuBLAS Documentation</a>
 */
public final class CublasRuntime {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup CUBLAS;
    private static final boolean AVAILABLE;

    // cuBLAS status codes
    public static final int CUBLAS_STATUS_SUCCESS = 0;
    public static final int CUBLAS_STATUS_NOT_INITIALIZED = 1;
    public static final int CUBLAS_STATUS_ALLOC_FAILED = 3;
    public static final int CUBLAS_STATUS_INVALID_VALUE = 7;
    public static final int CUBLAS_STATUS_ARCH_MISMATCH = 8;
    public static final int CUBLAS_STATUS_MAPPING_ERROR = 11;
    public static final int CUBLAS_STATUS_EXECUTION_FAILED = 13;
    public static final int CUBLAS_STATUS_INTERNAL_ERROR = 14;
    public static final int CUBLAS_STATUS_NOT_SUPPORTED = 15;

    // cuBLAS operation types
    public static final int CUBLAS_OP_N = 0;  // No transpose
    public static final int CUBLAS_OP_T = 1;  // Transpose
    public static final int CUBLAS_OP_C = 2;  // Conjugate transpose

    // cuBLAS function handles
    private static final MethodHandle cublasCreate;
    private static final MethodHandle cublasDestroy;
    private static final MethodHandle cublasSetStream;
    private static final MethodHandle cublasGetStream;
    private static final MethodHandle cublasSgemm;
    private static final MethodHandle cublasDgemm;
    private static final MethodHandle cublasGetStatusString;

    static {
        SymbolLookup lookup = null;
        boolean available = false;

        try {
            // Try to load libcublas.so (Linux) or cublas64_*.dll (Windows)
            lookup = SymbolLookup.libraryLookup("cublas", Arena.global());
            available = true;
        } catch (IllegalArgumentException e) {
            // cuBLAS not available - try with full path on Linux
            try {
                lookup = SymbolLookup.libraryLookup("/usr/local/cuda/lib64/libcublas.so", Arena.global());
                available = true;
            } catch (IllegalArgumentException e2) {
                // Still not available
                lookup = SymbolLookup.loaderLookup();
            }
        }

        CUBLAS = lookup;
        AVAILABLE = available;

        if (available) {
            // cublasStatus_t cublasCreate_v2(cublasHandle_t *handle)
            cublasCreate = downcall("cublasCreate_v2", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS  // cublasHandle_t*
            ));

            // cublasStatus_t cublasDestroy_v2(cublasHandle_t handle)
            cublasDestroy = downcall("cublasDestroy_v2", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG  // cublasHandle_t
            ));

            // cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId)
            cublasSetStream = downcall("cublasSetStream_v2", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // cublasHandle_t
                ValueLayout.JAVA_LONG   // cudaStream_t
            ));

            // cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId)
            cublasGetStream = downcall("cublasGetStream_v2", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // cublasHandle_t
                ValueLayout.ADDRESS     // cudaStream_t*
            ));

            // cublasStatus_t cublasSgemm_v2(
            //   cublasHandle_t handle,
            //   cublasOperation_t transa, cublasOperation_t transb,
            //   int m, int n, int k,
            //   const float *alpha,
            //   const float *A, int lda,
            //   const float *B, int ldb,
            //   const float *beta,
            //   float *C, int ldc)
            cublasSgemm = downcall("cublasSgemm_v2", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // handle
                ValueLayout.JAVA_INT,   // transa
                ValueLayout.JAVA_INT,   // transb
                ValueLayout.JAVA_INT,   // m
                ValueLayout.JAVA_INT,   // n
                ValueLayout.JAVA_INT,   // k
                ValueLayout.ADDRESS,    // alpha
                ValueLayout.JAVA_LONG,  // A (device ptr)
                ValueLayout.JAVA_INT,   // lda
                ValueLayout.JAVA_LONG,  // B (device ptr)
                ValueLayout.JAVA_INT,   // ldb
                ValueLayout.ADDRESS,    // beta
                ValueLayout.JAVA_LONG,  // C (device ptr)
                ValueLayout.JAVA_INT    // ldc
            ));

            // cublasStatus_t cublasDgemm_v2 (same signature with double)
            cublasDgemm = downcall("cublasDgemm_v2", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // handle
                ValueLayout.JAVA_INT,   // transa
                ValueLayout.JAVA_INT,   // transb
                ValueLayout.JAVA_INT,   // m
                ValueLayout.JAVA_INT,   // n
                ValueLayout.JAVA_INT,   // k
                ValueLayout.ADDRESS,    // alpha
                ValueLayout.JAVA_LONG,  // A (device ptr)
                ValueLayout.JAVA_INT,   // lda
                ValueLayout.JAVA_LONG,  // B (device ptr)
                ValueLayout.JAVA_INT,   // ldb
                ValueLayout.ADDRESS,    // beta
                ValueLayout.JAVA_LONG,  // C (device ptr)
                ValueLayout.JAVA_INT    // ldc
            ));

            // const char* cublasGetStatusString(cublasStatus_t status)
            // Note: Returns pointer to static string, no need to free
            cublasGetStatusString = downcall("cublasGetStatusString", FunctionDescriptor.of(
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT
            ));
        } else {
            cublasCreate = null;
            cublasDestroy = null;
            cublasSetStream = null;
            cublasGetStream = null;
            cublasSgemm = null;
            cublasDgemm = null;
            cublasGetStatusString = null;
        }
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        return CUBLAS.find(name)
            .map(symbol -> LINKER.downcallHandle(symbol, descriptor))
            .orElse(null);
    }

    private CublasRuntime() {}

    /**
     * Check if cuBLAS is available on this system.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Ensure cuBLAS is available, throwing if not.
     */
    public static void ensureAvailable() {
        if (!AVAILABLE) {
            throw new UnsupportedOperationException("cuBLAS is not available on this system");
        }
    }

    // ==================== Handle Management ====================

    /**
     * Create a cuBLAS handle.
     *
     * @param arena Arena for allocating the handle pointer
     * @return cuBLAS handle
     */
    public static long create(Arena arena) throws Throwable {
        ensureAvailable();
        MemorySegment handlePtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) cublasCreate.invokeExact(handlePtr);
        checkError(result, "cublasCreate");
        return handlePtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Destroy a cuBLAS handle.
     */
    public static void destroy(long handle) throws Throwable {
        ensureAvailable();
        int result = (int) cublasDestroy.invokeExact(handle);
        checkError(result, "cublasDestroy");
    }

    /**
     * Set the CUDA stream for a cuBLAS handle.
     *
     * @param handle cuBLAS handle
     * @param stream CUDA stream (0 for default stream)
     */
    public static void setStream(long handle, long stream) throws Throwable {
        ensureAvailable();
        int result = (int) cublasSetStream.invokeExact(handle, stream);
        checkError(result, "cublasSetStream");
    }

    /**
     * Get the CUDA stream associated with a cuBLAS handle.
     */
    public static long getStream(Arena arena, long handle) throws Throwable {
        ensureAvailable();
        MemorySegment streamPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) cublasGetStream.invokeExact(handle, streamPtr);
        checkError(result, "cublasGetStream");
        return streamPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    // ==================== GEMM Operations ====================

    /**
     * Single-precision general matrix multiplication.
     *
     * <p>Computes C = alpha * op(A) * op(B) + beta * C
     *
     * <p>where op(X) is one of:
     * <ul>
     *   <li>op(X) = X if transX == CUBLAS_OP_N</li>
     *   <li>op(X) = X^T if transX == CUBLAS_OP_T</li>
     * </ul>
     *
     * <p><b>Important:</b> cuBLAS uses column-major storage. For row-major matrices,
     * use the transpose trick: to compute C = A * B with row-major data, call
     * sgemm(N, N, n, m, k, 1.0f, B, n, A, k, 0.0f, C, n) - i.e., swap A and B,
     * swap m and n.
     *
     * @param handle cuBLAS handle
     * @param transa Operation on A (CUBLAS_OP_N, CUBLAS_OP_T, or CUBLAS_OP_C)
     * @param transb Operation on B
     * @param m Number of rows of op(A) and C
     * @param n Number of columns of op(B) and C
     * @param k Number of columns of op(A) and rows of op(B)
     * @param alpha Scalar multiplier for A*B
     * @param dA Device pointer to matrix A
     * @param lda Leading dimension of A
     * @param dB Device pointer to matrix B
     * @param ldb Leading dimension of B
     * @param beta Scalar multiplier for C
     * @param dC Device pointer to matrix C
     * @param ldc Leading dimension of C
     */
    public static void sgemm(Arena arena, long handle,
                             int transa, int transb,
                             int m, int n, int k,
                             float alpha,
                             long dA, int lda,
                             long dB, int ldb,
                             float beta,
                             long dC, int ldc) throws Throwable {
        ensureAvailable();

        // Allocate alpha and beta on host (cuBLAS reads them from host memory by default)
        MemorySegment alphaPtr = arena.allocate(ValueLayout.JAVA_FLOAT);
        alphaPtr.set(ValueLayout.JAVA_FLOAT, 0, alpha);

        MemorySegment betaPtr = arena.allocate(ValueLayout.JAVA_FLOAT);
        betaPtr.set(ValueLayout.JAVA_FLOAT, 0, beta);

        int result = (int) cublasSgemm.invokeExact(
            handle,
            transa, transb,
            m, n, k,
            alphaPtr,
            dA, lda,
            dB, ldb,
            betaPtr,
            dC, ldc
        );
        checkError(result, "cublasSgemm");
    }

    /**
     * Double-precision general matrix multiplication.
     *
     * <p>Same semantics as {@link #sgemm} but with double precision.
     */
    public static void dgemm(Arena arena, long handle,
                             int transa, int transb,
                             int m, int n, int k,
                             double alpha,
                             long dA, int lda,
                             long dB, int ldb,
                             double beta,
                             long dC, int ldc) throws Throwable {
        ensureAvailable();

        MemorySegment alphaPtr = arena.allocate(ValueLayout.JAVA_DOUBLE);
        alphaPtr.set(ValueLayout.JAVA_DOUBLE, 0, alpha);

        MemorySegment betaPtr = arena.allocate(ValueLayout.JAVA_DOUBLE);
        betaPtr.set(ValueLayout.JAVA_DOUBLE, 0, beta);

        int result = (int) cublasDgemm.invokeExact(
            handle,
            transa, transb,
            m, n, k,
            alphaPtr,
            dA, lda,
            dB, ldb,
            betaPtr,
            dC, ldc
        );
        checkError(result, "cublasDgemm");
    }

    // ==================== Error Handling ====================

    /**
     * Get a human-readable error message for a cuBLAS status code.
     */
    public static String getStatusString(int status) {
        if (!AVAILABLE || cublasGetStatusString == null) {
            return "CUBLAS_STATUS_" + status;
        }
        try {
            MemorySegment strPtr = (MemorySegment) cublasGetStatusString.invokeExact(status);
            if (!strPtr.equals(MemorySegment.NULL)) {
                return strPtr.reinterpret(256).getString(0);
            }
        } catch (Throwable t) {
            // Ignore
        }
        return "CUBLAS_STATUS_" + status;
    }

    /**
     * Check a cuBLAS result and throw if not success.
     */
    public static void checkError(int result, String operation) {
        if (result != CUBLAS_STATUS_SUCCESS) {
            throw new CublasException(operation + " failed: " + getStatusString(result) + " (" + result + ")");
        }
    }

    /**
     * Exception thrown when a cuBLAS operation fails.
     */
    public static class CublasException extends RuntimeException {
        public CublasException(String message) {
            super(message);
        }

        public CublasException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
