package io.surfworks.warpforge.io.rdma.impl;

import io.surfworks.warpforge.io.rdma.RdmaBuffer;
import io.surfworks.warpforge.io.rdma.RdmaException;

import java.lang.foreign.MemorySegment;

/**
 * UCX implementation of RdmaBuffer.
 *
 * <p>Wraps a MemorySegment registered with UCX for RDMA operations.
 */
final class UcxRdmaBuffer implements RdmaBuffer {

    private final long id;
    private final MemorySegment segment;
    private final int flags;
    private final UcxRdmaImpl parent;
    private final MemorySegment ucpMemHandle; // UCX memory handle

    private volatile boolean valid = true;

    UcxRdmaBuffer(long id, MemorySegment segment, int flags, UcxRdmaImpl parent, MemorySegment ucpMemHandle) {
        this.id = id;
        this.segment = segment;
        this.flags = flags;
        this.parent = parent;
        this.ucpMemHandle = ucpMemHandle;
    }

    long id() {
        return id;
    }

    int flags() {
        return flags;
    }

    MemorySegment ucpMemHandle() {
        return ucpMemHandle;
    }

    @Override
    public MemorySegment segment() {
        checkValid();
        return segment;
    }

    @Override
    public long remoteKey() {
        checkValid();
        // TODO: Get rkey from UCX using ucp_rkey_pack()
        return id; // Placeholder
    }

    @Override
    public long localKey() {
        checkValid();
        return id; // Placeholder
    }

    @Override
    public long byteSize() {
        return segment.byteSize();
    }

    @Override
    public long address() {
        checkValid();
        return segment.address();
    }

    @Override
    public boolean isValid() {
        return valid;
    }

    @Override
    public void close() {
        if (valid) {
            valid = false;
            parent.unregisterMemory(this);
        }
    }

    void invalidate() {
        valid = false;
    }

    private void checkValid() {
        if (!valid) {
            throw new RdmaException("Buffer has been closed", RdmaException.ErrorCode.INVALID_STATE);
        }
    }
}
