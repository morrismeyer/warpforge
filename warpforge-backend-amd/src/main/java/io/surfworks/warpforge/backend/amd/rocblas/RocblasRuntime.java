package io.surfworks.warpforge.backend.amd.rocblas;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * FFM bindings to rocBLAS library.
 *
 * <p>This class provides access to AMD's rocBLAS library for high-performance
 * linear algebra operations on AMD GPUs. rocBLAS is the AMD equivalent of cuBLAS.
 *
 * <p>rocBLAS uses column-major storage by default (Fortran-style), same as cuBLAS.
 *
 * @see <a href="https://rocm.docs.amd.com/projects/rocBLAS/en/latest/">rocBLAS Documentation</a>
 */
public final class RocblasRuntime {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup ROCBLAS;
    private static final boolean AVAILABLE;

    // rocBLAS status codes
    public static final int ROCBLAS_STATUS_SUCCESS = 0;
    public static final int ROCBLAS_STATUS_INVALID_HANDLE = 1;
    public static final int ROCBLAS_STATUS_NOT_IMPLEMENTED = 2;
    public static final int ROCBLAS_STATUS_INVALID_POINTER = 3;
    public static final int ROCBLAS_STATUS_INVALID_SIZE = 4;
    public static final int ROCBLAS_STATUS_MEMORY_ERROR = 5;
    public static final int ROCBLAS_STATUS_INTERNAL_ERROR = 6;
    public static final int ROCBLAS_STATUS_PERF_DEGRADED = 7;
    public static final int ROCBLAS_STATUS_SIZE_QUERY_MISMATCH = 8;
    public static final int ROCBLAS_STATUS_SIZE_INCREASED = 9;
    public static final int ROCBLAS_STATUS_SIZE_UNCHANGED = 10;
    public static final int ROCBLAS_STATUS_INVALID_VALUE = 11;
    public static final int ROCBLAS_STATUS_CONTINUE = 12;
    public static final int ROCBLAS_STATUS_CHECK_NUMERICS_FAIL = 13;

    // rocBLAS operation types
    public static final int ROCBLAS_OP_N = 111;  // No transpose (rocblas_operation_none)
    public static final int ROCBLAS_OP_T = 112;  // Transpose (rocblas_operation_transpose)
    public static final int ROCBLAS_OP_C = 113;  // Conjugate transpose

    // rocBLAS function handles
    private static final MethodHandle rocblasCreate;
    private static final MethodHandle rocblasDestroy;
    private static final MethodHandle rocblasSetStream;
    private static final MethodHandle rocblasGetStream;
    private static final MethodHandle rocblasSgemm;
    private static final MethodHandle rocblasDgemm;
    private static final MethodHandle rocblasStatusToString;

    static {
        SymbolLookup lookup = null;
        boolean available = false;

        try {
            // Try to load librocblas.so
            lookup = SymbolLookup.libraryLookup("rocblas", Arena.global());
            available = true;
        } catch (IllegalArgumentException e) {
            // Try common ROCm installation path
            try {
                lookup = SymbolLookup.libraryLookup("/opt/rocm/lib/librocblas.so", Arena.global());
                available = true;
            } catch (IllegalArgumentException e2) {
                // rocBLAS not available
                lookup = SymbolLookup.loaderLookup();
            }
        }

        ROCBLAS = lookup;
        AVAILABLE = available;

        if (available) {
            // rocblas_status rocblas_create_handle(rocblas_handle *handle)
            rocblasCreate = downcall("rocblas_create_handle", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS  // rocblas_handle*
            ));

            // rocblas_status rocblas_destroy_handle(rocblas_handle handle)
            rocblasDestroy = downcall("rocblas_destroy_handle", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG  // rocblas_handle
            ));

            // rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream)
            rocblasSetStream = downcall("rocblas_set_stream", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // rocblas_handle
                ValueLayout.JAVA_LONG   // hipStream_t
            ));

            // rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t *stream)
            rocblasGetStream = downcall("rocblas_get_stream", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // rocblas_handle
                ValueLayout.ADDRESS     // hipStream_t*
            ));

            // rocblas_status rocblas_sgemm(
            //   rocblas_handle handle,
            //   rocblas_operation transA, rocblas_operation transB,
            //   rocblas_int m, rocblas_int n, rocblas_int k,
            //   const float *alpha,
            //   const float *A, rocblas_int lda,
            //   const float *B, rocblas_int ldb,
            //   const float *beta,
            //   float *C, rocblas_int ldc)
            rocblasSgemm = downcall("rocblas_sgemm", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // handle
                ValueLayout.JAVA_INT,   // transA
                ValueLayout.JAVA_INT,   // transB
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

            // rocblas_status rocblas_dgemm (same signature with double)
            rocblasDgemm = downcall("rocblas_dgemm", FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,  // handle
                ValueLayout.JAVA_INT,   // transA
                ValueLayout.JAVA_INT,   // transB
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

            // const char* rocblas_status_to_string(rocblas_status status)
            rocblasStatusToString = downcall("rocblas_status_to_string", FunctionDescriptor.of(
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_INT
            ));
        } else {
            rocblasCreate = null;
            rocblasDestroy = null;
            rocblasSetStream = null;
            rocblasGetStream = null;
            rocblasSgemm = null;
            rocblasDgemm = null;
            rocblasStatusToString = null;
        }
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        return ROCBLAS.find(name)
            .map(symbol -> LINKER.downcallHandle(symbol, descriptor))
            .orElse(null);
    }

    private RocblasRuntime() {}

    /**
     * Check if rocBLAS is available on this system.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Ensure rocBLAS is available, throwing if not.
     */
    public static void ensureAvailable() {
        if (!AVAILABLE) {
            throw new UnsupportedOperationException("rocBLAS is not available on this system");
        }
    }

    // ==================== Handle Management ====================

    /**
     * Create a rocBLAS handle.
     */
    public static long create(Arena arena) throws Throwable {
        ensureAvailable();
        MemorySegment handlePtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) rocblasCreate.invokeExact(handlePtr);
        checkError(result, "rocblas_create_handle");
        return handlePtr.get(ValueLayout.JAVA_LONG, 0);
    }

    /**
     * Destroy a rocBLAS handle.
     */
    public static void destroy(long handle) throws Throwable {
        ensureAvailable();
        int result = (int) rocblasDestroy.invokeExact(handle);
        checkError(result, "rocblas_destroy_handle");
    }

    /**
     * Set the HIP stream for a rocBLAS handle.
     */
    public static void setStream(long handle, long stream) throws Throwable {
        ensureAvailable();
        int result = (int) rocblasSetStream.invokeExact(handle, stream);
        checkError(result, "rocblas_set_stream");
    }

    /**
     * Get the HIP stream associated with a rocBLAS handle.
     */
    public static long getStream(Arena arena, long handle) throws Throwable {
        ensureAvailable();
        MemorySegment streamPtr = arena.allocate(ValueLayout.JAVA_LONG);
        int result = (int) rocblasGetStream.invokeExact(handle, streamPtr);
        checkError(result, "rocblas_get_stream");
        return streamPtr.get(ValueLayout.JAVA_LONG, 0);
    }

    // ==================== GEMM Operations ====================

    /**
     * Single-precision general matrix multiplication.
     *
     * <p>Computes C = alpha * op(A) * op(B) + beta * C
     *
     * <p><b>Important:</b> rocBLAS uses column-major storage. For row-major matrices,
     * use the transpose trick (same as cuBLAS).
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

        MemorySegment alphaPtr = arena.allocate(ValueLayout.JAVA_FLOAT);
        alphaPtr.set(ValueLayout.JAVA_FLOAT, 0, alpha);

        MemorySegment betaPtr = arena.allocate(ValueLayout.JAVA_FLOAT);
        betaPtr.set(ValueLayout.JAVA_FLOAT, 0, beta);

        int result = (int) rocblasSgemm.invokeExact(
            handle,
            transa, transb,
            m, n, k,
            alphaPtr,
            dA, lda,
            dB, ldb,
            betaPtr,
            dC, ldc
        );
        checkError(result, "rocblas_sgemm");
    }

    /**
     * Double-precision general matrix multiplication.
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

        int result = (int) rocblasDgemm.invokeExact(
            handle,
            transa, transb,
            m, n, k,
            alphaPtr,
            dA, lda,
            dB, ldb,
            betaPtr,
            dC, ldc
        );
        checkError(result, "rocblas_dgemm");
    }

    // ==================== Error Handling ====================

    /**
     * Get a human-readable error message for a rocBLAS status code.
     */
    public static String getStatusString(int status) {
        if (!AVAILABLE || rocblasStatusToString == null) {
            return "ROCBLAS_STATUS_" + status;
        }
        try {
            MemorySegment strPtr = (MemorySegment) rocblasStatusToString.invokeExact(status);
            if (!strPtr.equals(MemorySegment.NULL)) {
                return strPtr.reinterpret(256).getString(0);
            }
        } catch (Throwable t) {
            // Ignore
        }
        return "ROCBLAS_STATUS_" + status;
    }

    /**
     * Check a rocBLAS result and throw if not success.
     */
    public static void checkError(int result, String operation) {
        if (result != ROCBLAS_STATUS_SUCCESS) {
            throw new RocblasException(operation + " failed: " + getStatusString(result) + " (" + result + ")");
        }
    }

    /**
     * Exception thrown when a rocBLAS operation fails.
     */
    public static class RocblasException extends RuntimeException {
        public RocblasException(String message) {
            super(message);
        }

        public RocblasException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
