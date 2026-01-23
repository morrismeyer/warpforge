package io.surfworks.warpforge.io.collective.impl;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.io.ffi.ucc.Ucc;

/**
 * UCC constants and type mapping utilities.
 *
 * <p>This class centralizes UCC constants and provides mapping utilities
 * for converting between WarpForge types and UCC FFM types.
 *
 * <p>All constants are accessed via the FFM-generated {@link Ucc} class,
 * but this class provides convenient type-safe wrappers and conversions.
 */
public final class UccConstants {

    private UccConstants() {
        // Utility class
    }

    // ========================================================================
    // Status Codes
    // ========================================================================

    /** Operation completed successfully */
    public static final int OK = 0;

    /** Operation is still in progress */
    public static final int INPROGRESS = 1;

    /** Operation initialized but not yet posted */
    public static final int OPERATION_INITIALIZED = 2;

    // Error codes (negative values)
    public static final int ERR_NOT_SUPPORTED = -1;
    public static final int ERR_NOT_IMPLEMENTED = -2;
    public static final int ERR_INVALID_PARAM = -3;
    public static final int ERR_NO_MEMORY = -4;
    public static final int ERR_NO_RESOURCE = -5;
    public static final int ERR_NO_MESSAGE = -6;
    public static final int ERR_NOT_FOUND = -7;
    public static final int ERR_TIMED_OUT = -8;
    public static final int ERR_IO_ERROR = -9;

    // ========================================================================
    // Collective Types
    // ========================================================================

    public static final int COLL_TYPE_ALLGATHER = 1;
    public static final int COLL_TYPE_ALLGATHERV = 2;
    public static final int COLL_TYPE_ALLREDUCE = 4;
    public static final int COLL_TYPE_ALLTOALL = 8;
    public static final int COLL_TYPE_ALLTOALLV = 16;
    public static final int COLL_TYPE_BARRIER = 32;
    public static final int COLL_TYPE_BCAST = 64;
    public static final int COLL_TYPE_FANIN = 128;
    public static final int COLL_TYPE_FANOUT = 256;
    public static final int COLL_TYPE_GATHER = 512;
    public static final int COLL_TYPE_GATHERV = 1024;
    public static final int COLL_TYPE_REDUCE = 2048;
    public static final int COLL_TYPE_REDUCE_SCATTER = 4096;
    public static final int COLL_TYPE_REDUCE_SCATTERV = 8192;
    public static final int COLL_TYPE_SCATTER = 16384;
    public static final int COLL_TYPE_SCATTERV = 32768;

    // ========================================================================
    // Reduction Operations
    // ========================================================================

    public static final int OP_SUM = 0;
    public static final int OP_PROD = 1;
    public static final int OP_MAX = 2;
    public static final int OP_MIN = 3;
    public static final int OP_LAND = 4;
    public static final int OP_LOR = 5;
    public static final int OP_LXOR = 6;
    public static final int OP_BAND = 7;
    public static final int OP_BOR = 8;
    public static final int OP_BXOR = 9;
    public static final int OP_MAXLOC = 10;
    public static final int OP_MINLOC = 11;
    public static final int OP_AVG = 12;

    // ========================================================================
    // Data Types
    // ========================================================================

    public static final long DT_INT8 = 0L;
    public static final long DT_INT16 = 8L;
    public static final long DT_INT32 = 16L;
    public static final long DT_INT64 = 24L;
    public static final long DT_INT128 = 32L;
    public static final long DT_UINT8 = 40L;
    public static final long DT_UINT16 = 48L;
    public static final long DT_UINT32 = 56L;
    public static final long DT_UINT64 = 64L;
    public static final long DT_UINT128 = 72L;
    public static final long DT_FLOAT16 = 80L;
    public static final long DT_FLOAT32 = 88L;
    public static final long DT_FLOAT64 = 96L;
    public static final long DT_BFLOAT16 = 104L;
    public static final long DT_FLOAT128 = 112L;

    // ========================================================================
    // Memory Types
    // ========================================================================

    public static final int MEMORY_TYPE_HOST = 0;
    public static final int MEMORY_TYPE_CUDA = 1;
    public static final int MEMORY_TYPE_CUDA_MANAGED = 2;
    public static final int MEMORY_TYPE_ROCM = 3;
    public static final int MEMORY_TYPE_ROCM_MANAGED = 4;

    // ========================================================================
    // Collective Flags
    // ========================================================================

    public static final long COLL_ARGS_FLAG_IN_PLACE = 1L;

    // ========================================================================
    // Endpoint Range Types
    // ========================================================================

    public static final int EP_RANGE_CONTIG = 0;
    public static final int EP_RANGE_NONCONTIG = 1;

    // ========================================================================
    // Type Mapping Utilities
    // ========================================================================

    /**
     * Convert a WarpForge ScalarType to UCC datatype constant.
     *
     * @param type the WarpForge scalar type
     * @return the corresponding UCC datatype constant
     * @throws IllegalArgumentException if the type is not supported
     */
    public static long scalarTypeToUccDatatype(ScalarType type) {
        return switch (type) {
            case F16 -> DT_FLOAT16;
            case F32 -> DT_FLOAT32;
            case F64 -> DT_FLOAT64;
            case BF16 -> DT_BFLOAT16;
            case I8 -> DT_INT8;
            case I16 -> DT_INT16;
            case I32 -> DT_INT32;
            case I64 -> DT_INT64;
            case I1, BOOL -> DT_INT8; // Treat boolean as byte
            // 8-bit float formats can be transferred as raw bytes
            case F8_E5M2, F8_E4M3, F8_E4M3FN, F8_E8M0 -> DT_UINT8;
            // Sub-byte formats are not directly supported by UCC
            case F4_E2M1, F4_E1M2, F6_E3M2, F6_E2M3 ->
                throw new IllegalArgumentException("Sub-byte format " + type + " not supported by UCC");
        };
    }

    /**
     * Get the byte size of a UCC datatype.
     *
     * @param uccDatatype the UCC datatype constant
     * @return the byte size of one element
     */
    public static int uccDatatypeByteSize(long uccDatatype) {
        // UCC datatypes encode size in bits 0-2 of the type code
        // INT8=0, INT16=8, INT32=16, INT64=24, etc.
        // The pattern is: type_id * 8 where type_id encodes size
        if (uccDatatype == DT_INT8 || uccDatatype == DT_UINT8) return 1;
        if (uccDatatype == DT_INT16 || uccDatatype == DT_UINT16 || uccDatatype == DT_FLOAT16 || uccDatatype == DT_BFLOAT16) return 2;
        if (uccDatatype == DT_INT32 || uccDatatype == DT_UINT32 || uccDatatype == DT_FLOAT32) return 4;
        if (uccDatatype == DT_INT64 || uccDatatype == DT_UINT64 || uccDatatype == DT_FLOAT64) return 8;
        if (uccDatatype == DT_INT128 || uccDatatype == DT_UINT128 || uccDatatype == DT_FLOAT128) return 16;
        throw new IllegalArgumentException("Unknown UCC datatype: " + uccDatatype);
    }

    /**
     * Convert a UCC status code to a human-readable string.
     *
     * @param status the UCC status code
     * @return a human-readable description
     */
    public static String statusToString(int status) {
        return switch (status) {
            case OK -> "UCC_OK";
            case INPROGRESS -> "UCC_INPROGRESS";
            case OPERATION_INITIALIZED -> "UCC_OPERATION_INITIALIZED";
            case ERR_NOT_SUPPORTED -> "UCC_ERR_NOT_SUPPORTED";
            case ERR_NOT_IMPLEMENTED -> "UCC_ERR_NOT_IMPLEMENTED";
            case ERR_INVALID_PARAM -> "UCC_ERR_INVALID_PARAM";
            case ERR_NO_MEMORY -> "UCC_ERR_NO_MEMORY";
            case ERR_NO_RESOURCE -> "UCC_ERR_NO_RESOURCE";
            case ERR_NO_MESSAGE -> "UCC_ERR_NO_MESSAGE";
            case ERR_NOT_FOUND -> "UCC_ERR_NOT_FOUND";
            case ERR_TIMED_OUT -> "UCC_ERR_TIMED_OUT";
            case ERR_IO_ERROR -> "UCC_ERR_IO_ERROR";
            default -> "UCC_UNKNOWN(" + status + ")";
        };
    }

    /**
     * Convert a collective type to a human-readable string.
     *
     * @param collType the collective type constant
     * @return a human-readable description
     */
    public static String collTypeToString(int collType) {
        return switch (collType) {
            case COLL_TYPE_ALLGATHER -> "ALLGATHER";
            case COLL_TYPE_ALLGATHERV -> "ALLGATHERV";
            case COLL_TYPE_ALLREDUCE -> "ALLREDUCE";
            case COLL_TYPE_ALLTOALL -> "ALLTOALL";
            case COLL_TYPE_ALLTOALLV -> "ALLTOALLV";
            case COLL_TYPE_BARRIER -> "BARRIER";
            case COLL_TYPE_BCAST -> "BCAST";
            case COLL_TYPE_FANIN -> "FANIN";
            case COLL_TYPE_FANOUT -> "FANOUT";
            case COLL_TYPE_GATHER -> "GATHER";
            case COLL_TYPE_GATHERV -> "GATHERV";
            case COLL_TYPE_REDUCE -> "REDUCE";
            case COLL_TYPE_REDUCE_SCATTER -> "REDUCE_SCATTER";
            case COLL_TYPE_REDUCE_SCATTERV -> "REDUCE_SCATTERV";
            case COLL_TYPE_SCATTER -> "SCATTER";
            case COLL_TYPE_SCATTERV -> "SCATTERV";
            default -> "UNKNOWN(" + collType + ")";
        };
    }

    /**
     * Check if a status indicates an error.
     *
     * @param status the UCC status code
     * @return true if the status is an error (negative value)
     */
    public static boolean isError(int status) {
        return status < 0;
    }

    /**
     * Check if a status indicates the operation is still in progress.
     *
     * @param status the UCC status code
     * @return true if the operation is in progress
     */
    public static boolean isInProgress(int status) {
        return status == INPROGRESS;
    }

    /**
     * Check if a status indicates success.
     *
     * @param status the UCC status code
     * @return true if the operation completed successfully
     */
    public static boolean isOk(int status) {
        return status == OK;
    }
}
