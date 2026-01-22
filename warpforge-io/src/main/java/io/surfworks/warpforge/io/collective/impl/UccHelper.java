package io.surfworks.warpforge.io.collective.impl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.io.collective.CollectiveException;
import io.surfworks.warpforge.io.ffi.ucc.Ucc;
import io.surfworks.warpforge.io.ffi.ucc.ucc_coll_args;
import io.surfworks.warpforge.io.ffi.ucc.ucc_coll_buffer_info;

/**
 * Helper utilities for UCC FFM operations.
 *
 * <p>This class provides convenience methods for common FFM patterns when
 * working with UCC collective operations, including status checking,
 * buffer setup, and completion polling.
 */
public final class UccHelper {

    private UccHelper() {
        // Utility class
    }

    // ========================================================================
    // Status Checking
    // ========================================================================

    /**
     * Check a UCC status code and throw an exception if it indicates an error.
     *
     * @param status the UCC status code
     * @param operation description of the operation for error messages
     * @throws CollectiveException if status indicates an error
     */
    public static void checkStatus(int status, String operation) {
        if (UccConstants.isError(status)) {
            throw new CollectiveException(
                operation + " failed: " + UccConstants.statusToString(status),
                CollectiveException.ErrorCode.COMMUNICATION_ERROR,
                status
            );
        }
    }

    /**
     * Check a UCC status code, allowing INPROGRESS as success.
     *
     * @param status the UCC status code
     * @param operation description of the operation for error messages
     * @throws CollectiveException if status indicates an error (not INPROGRESS)
     */
    public static void checkStatusAllowInProgress(int status, String operation) {
        if (UccConstants.isError(status)) {
            throw new CollectiveException(
                operation + " failed: " + UccConstants.statusToString(status),
                CollectiveException.ErrorCode.COMMUNICATION_ERROR,
                status
            );
        }
    }

    // ========================================================================
    // Completion Polling
    // ========================================================================

    /**
     * Wait for a UCC collective operation to complete.
     *
     * <p>This method polls {@code ucc_collective_finalize} until the operation
     * completes or an error occurs.
     *
     * @param request the collective request handle
     * @throws CollectiveException if the operation fails
     */
    public static void waitForCompletion(MemorySegment request) {
        while (true) {
            int status = Ucc.ucc_collective_finalize(request);
            if (status == UccConstants.OK) {
                return;
            }
            if (status != UccConstants.INPROGRESS) {
                throw new CollectiveException(
                    "Collective operation failed: " + UccConstants.statusToString(status),
                    CollectiveException.ErrorCode.COMMUNICATION_ERROR,
                    status
                );
            }
            // Yield to allow other work while waiting
            Thread.onSpinWait();
        }
    }

    /**
     * Wait for a UCC collective operation to complete with timeout.
     *
     * @param request the collective request handle
     * @param timeoutMs maximum time to wait in milliseconds
     * @throws CollectiveException if the operation fails or times out
     */
    public static void waitForCompletionWithTimeout(MemorySegment request, long timeoutMs) {
        long startTime = System.currentTimeMillis();
        while (true) {
            int status = Ucc.ucc_collective_finalize(request);
            if (status == UccConstants.OK) {
                return;
            }
            if (status != UccConstants.INPROGRESS) {
                throw new CollectiveException(
                    "Collective operation failed: " + UccConstants.statusToString(status),
                    CollectiveException.ErrorCode.COMMUNICATION_ERROR,
                    status
                );
            }
            if (System.currentTimeMillis() - startTime > timeoutMs) {
                throw new CollectiveException(
                    "Collective operation timed out after " + timeoutMs + "ms",
                    CollectiveException.ErrorCode.TIMEOUT
                );
            }
            Thread.onSpinWait();
        }
    }

    // ========================================================================
    // Buffer Setup
    // ========================================================================

    /**
     * Set up a UCC buffer info structure from a Tensor.
     *
     * @param bufferInfo the pre-allocated ucc_coll_buffer_info segment
     * @param tensor the tensor to describe
     */
    public static void setupBufferInfo(MemorySegment bufferInfo, Tensor tensor) {
        ucc_coll_buffer_info.buffer(bufferInfo, tensor.data());
        ucc_coll_buffer_info.count(bufferInfo, tensor.spec().elementCount());
        ucc_coll_buffer_info.datatype(bufferInfo, UccConstants.scalarTypeToUccDatatype(tensor.dtype()));
        ucc_coll_buffer_info.mem_type(bufferInfo, UccConstants.MEMORY_TYPE_HOST);
    }

    /**
     * Set up a UCC buffer info structure from raw memory.
     *
     * @param bufferInfo the pre-allocated ucc_coll_buffer_info segment
     * @param buffer the memory segment containing the data
     * @param count the number of elements
     * @param uccDatatype the UCC datatype constant
     * @param memType the memory type (HOST, CUDA, etc.)
     */
    public static void setupBufferInfo(MemorySegment bufferInfo, MemorySegment buffer,
                                       long count, long uccDatatype, int memType) {
        ucc_coll_buffer_info.buffer(bufferInfo, buffer);
        ucc_coll_buffer_info.count(bufferInfo, count);
        ucc_coll_buffer_info.datatype(bufferInfo, uccDatatype);
        ucc_coll_buffer_info.mem_type(bufferInfo, memType);
    }

    // ========================================================================
    // Collective Args Setup
    // ========================================================================

    /**
     * Get the source buffer info from collective args.
     *
     * @param args the ucc_coll_args segment
     * @return the source buffer info segment
     */
    public static MemorySegment getSrcBufferInfo(MemorySegment args) {
        MemorySegment src = ucc_coll_args.src(args);
        return ucc_coll_args.src.info(src);
    }

    /**
     * Get the destination buffer info from collective args.
     *
     * @param args the ucc_coll_args segment
     * @return the destination buffer info segment
     */
    public static MemorySegment getDstBufferInfo(MemorySegment args) {
        MemorySegment dst = ucc_coll_args.dst(args);
        return ucc_coll_args.dst.info(dst);
    }

    /**
     * Set up collective args for a basic collective operation.
     *
     * @param args the pre-allocated ucc_coll_args segment
     * @param collType the collective type constant
     */
    public static void setupCollectiveArgs(MemorySegment args, int collType) {
        ucc_coll_args.mask(args, 0L);
        ucc_coll_args.coll_type(args, collType);
    }

    /**
     * Set up collective args for an operation with reduction.
     *
     * @param args the pre-allocated ucc_coll_args segment
     * @param collType the collective type constant
     * @param reductionOp the reduction operation constant
     */
    public static void setupCollectiveArgsWithOp(MemorySegment args, int collType, int reductionOp) {
        ucc_coll_args.mask(args, 0L);
        ucc_coll_args.coll_type(args, collType);
        ucc_coll_args.op(args, reductionOp);
    }

    /**
     * Set up collective args for an operation with root rank.
     *
     * @param args the pre-allocated ucc_coll_args segment
     * @param collType the collective type constant
     * @param root the root rank
     */
    public static void setupCollectiveArgsWithRoot(MemorySegment args, int collType, int root) {
        ucc_coll_args.mask(args, 0L);
        ucc_coll_args.coll_type(args, collType);
        ucc_coll_args.root(args, root);
    }

    /**
     * Set up collective args for an operation with both reduction and root.
     *
     * @param args the pre-allocated ucc_coll_args segment
     * @param collType the collective type constant
     * @param reductionOp the reduction operation constant
     * @param root the root rank
     */
    public static void setupCollectiveArgsWithOpAndRoot(MemorySegment args, int collType,
                                                        int reductionOp, int root) {
        ucc_coll_args.mask(args, 0L);
        ucc_coll_args.coll_type(args, collType);
        ucc_coll_args.op(args, reductionOp);
        ucc_coll_args.root(args, root);
    }

    // ========================================================================
    // Memory Allocation
    // ========================================================================

    /**
     * Allocate a pointer-sized memory segment for receiving a handle.
     *
     * @param arena the arena to allocate from
     * @return a segment of ADDRESS size
     */
    public static MemorySegment allocatePointer(Arena arena) {
        return arena.allocate(ValueLayout.ADDRESS);
    }

    /**
     * Dereference a pointer to get the contained address.
     *
     * @param pointerSegment a segment containing a pointer
     * @return the address contained in the pointer
     */
    public static MemorySegment dereferencePointer(MemorySegment pointerSegment) {
        return pointerSegment.get(ValueLayout.ADDRESS, 0);
    }

    /**
     * Check if a memory segment is null (all zeros or NULL constant).
     *
     * @param segment the segment to check
     * @return true if the segment represents a null pointer
     */
    public static boolean isNull(MemorySegment segment) {
        return segment == null || segment.equals(MemorySegment.NULL) || segment.address() == 0;
    }
}
