package io.surfworks.warpforge.io.rdma;

import java.lang.foreign.MemorySegment;

/**
 * RDMA-registered memory buffer.
 *
 * <p>Wraps a {@link MemorySegment} with UCX memory registration metadata,
 * enabling zero-copy RDMA operations. The memory is pinned (page-locked)
 * and registered with the RDMA device for direct hardware access.
 *
 * <p>Buffers must be explicitly closed when no longer needed to unregister
 * the memory and release resources. Using a buffer after close results in
 * undefined behavior.
 *
 * <h2>Zero-Copy Guarantee</h2>
 * <p>The underlying {@link MemorySegment} is the actual memory used for
 * RDMA operations. No copies are made during send/receive/read/write.
 * The segment can be passed directly to WarpForge GPU backends that
 * support MemorySegment.
 *
 * <h2>Thread Safety</h2>
 * <p>RdmaBuffer instances are not thread-safe. Concurrent access from
 * multiple threads requires external synchronization. However, different
 * buffers can be used concurrently from different threads.
 */
public interface RdmaBuffer extends AutoCloseable {

    /**
     * Returns the underlying MemorySegment.
     *
     * <p>This is the actual memory buffer used for RDMA operations.
     * Modifications to this segment are visible to remote peers after
     * RDMA operations complete.
     *
     * @return the underlying memory segment
     * @throws IllegalStateException if the buffer has been closed
     */
    MemorySegment segment();

    /**
     * Returns the remote key for RDMA operations.
     *
     * <p>The remote key (rkey) is used by remote peers to access this
     * buffer via one-sided RDMA read/write operations.
     *
     * @return opaque remote key value
     * @throws IllegalStateException if the buffer has been closed
     */
    long remoteKey();

    /**
     * Returns the local key for RDMA operations.
     *
     * <p>The local key (lkey) is used by the local process for RDMA
     * operations on this buffer.
     *
     * @return opaque local key value
     * @throws IllegalStateException if the buffer has been closed
     */
    long localKey();

    /**
     * Returns the size of this buffer in bytes.
     *
     * @return buffer size in bytes
     */
    long byteSize();

    /**
     * Returns the base address of this buffer for RDMA operations.
     *
     * <p>This address is used in RDMA read/write operations to specify
     * the target location within this buffer.
     *
     * @return virtual address of the buffer start
     * @throws IllegalStateException if the buffer has been closed
     */
    long address();

    /**
     * Returns whether this buffer is still valid (not closed).
     *
     * @return true if the buffer can be used for operations
     */
    boolean isValid();

    /**
     * Closes this buffer, unregistering the memory from RDMA.
     *
     * <p>After close, the underlying MemorySegment may still be valid
     * (depending on Arena ownership), but RDMA operations on this buffer
     * will fail.
     */
    @Override
    void close();
}
