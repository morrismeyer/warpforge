package io.surfworks.warpforge.io.rdma.mock;

import io.surfworks.warpforge.io.rdma.RdmaBuffer;
import io.surfworks.warpforge.io.rdma.RdmaException;

import java.lang.foreign.MemorySegment;

/**
 * Mock implementation of RdmaBuffer for testing.
 */
final class MockRdmaBuffer implements RdmaBuffer {

    private final long id;
    private final MemorySegment segment;
    private final int flags;
    private final RdmaMock parent;
    private volatile boolean valid = true;

    MockRdmaBuffer(long id, MemorySegment segment, int flags, RdmaMock parent) {
        this.id = id;
        this.segment = segment;
        this.flags = flags;
        this.parent = parent;
    }

    long id() {
        return id;
    }

    int flags() {
        return flags;
    }

    @Override
    public MemorySegment segment() {
        checkValid();
        return segment;
    }

    @Override
    public long remoteKey() {
        checkValid();
        // Mock remote key is just the buffer ID
        return id;
    }

    @Override
    public long localKey() {
        checkValid();
        return id;
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
