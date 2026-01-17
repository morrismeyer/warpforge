package io.surfworks.warpforge.io.rdma.mock;

import io.surfworks.warpforge.io.rdma.RdmaBuffer;
import io.surfworks.warpforge.io.rdma.RdmaEndpoint;
import io.surfworks.warpforge.io.rdma.RdmaException;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Mock implementation of RdmaEndpoint for testing.
 *
 * <p>Operations are simulated locally. For send/receive pairs within the
 * same process, data is copied via shared buffers. This allows testing
 * the RDMA API semantics without actual hardware.
 */
final class MockRdmaEndpoint implements RdmaEndpoint {

    private final String remoteAddress;
    private final int port;
    private final RdmaMock parent;

    private volatile EndpointState state = EndpointState.CONNECTED;
    private final AtomicLong bytesSent = new AtomicLong();
    private final AtomicLong bytesReceived = new AtomicLong();
    private final AtomicLong sendOps = new AtomicLong();
    private final AtomicLong recvOps = new AtomicLong();
    private final AtomicLong writeOps = new AtomicLong();
    private final AtomicLong readOps = new AtomicLong();
    private final AtomicLong atomicOps = new AtomicLong();
    private final AtomicLong errors = new AtomicLong();

    MockRdmaEndpoint(String remoteAddress, int port, RdmaMock parent) {
        this.remoteAddress = remoteAddress;
        this.port = port;
        this.parent = parent;
    }

    @Override
    public String remoteAddress() {
        return remoteAddress + ":" + port;
    }

    @Override
    public EndpointState state() {
        return state;
    }

    @Override
    public CompletableFuture<Void> send(RdmaBuffer buffer) {
        return send(buffer, 0, buffer.byteSize());
    }

    @Override
    public CompletableFuture<Void> send(RdmaBuffer buffer, long offset, long length) {
        checkConnected();
        return CompletableFuture.supplyAsync(() -> {
            // Simulate send - in mock, this is just recording the operation
            bytesSent.addAndGet(length);
            sendOps.incrementAndGet();
            parent.recordSend(length);
            return null;
        });
    }

    @Override
    public CompletableFuture<Long> receive(RdmaBuffer buffer) {
        checkConnected();
        return CompletableFuture.supplyAsync(() -> {
            // Simulate receive - return buffer size as if fully filled
            long received = buffer.byteSize();
            bytesReceived.addAndGet(received);
            recvOps.incrementAndGet();
            parent.recordReceive(received);
            return received;
        });
    }

    @Override
    public CompletableFuture<Void> write(RdmaBuffer localBuffer, long remoteAddress, long remoteKey) {
        return write(localBuffer, 0, localBuffer.byteSize(), remoteAddress, remoteKey);
    }

    @Override
    public CompletableFuture<Void> write(RdmaBuffer localBuffer, long localOffset, long length,
                                          long remoteAddress, long remoteKey) {
        checkConnected();
        return CompletableFuture.supplyAsync(() -> {
            // In mock mode, look up the remote buffer by key (which is the buffer ID)
            MockRdmaBuffer remoteBuffer = parent.getBuffer(remoteKey);
            if (remoteBuffer != null) {
                // Perform actual copy if remote buffer exists (loopback scenario)
                MemorySegment src = localBuffer.segment().asSlice(localOffset, length);
                MemorySegment dst = remoteBuffer.segment().asSlice(
                        remoteAddress - remoteBuffer.address(), length);
                MemorySegment.copy(src, 0, dst, 0, length);
            }

            bytesSent.addAndGet(length);
            writeOps.incrementAndGet();
            parent.recordSend(length);
            return null;
        });
    }

    @Override
    public CompletableFuture<Void> writeImmediate(RdmaBuffer localBuffer, long remoteAddress,
                                                   long remoteKey, int immediate) {
        // Immediate data is just metadata in mock - same as regular write
        return write(localBuffer, remoteAddress, remoteKey);
    }

    @Override
    public CompletableFuture<Void> read(RdmaBuffer localBuffer, long remoteAddress, long remoteKey) {
        return read(localBuffer, 0, localBuffer.byteSize(), remoteAddress, remoteKey);
    }

    @Override
    public CompletableFuture<Void> read(RdmaBuffer localBuffer, long localOffset, long length,
                                         long remoteAddress, long remoteKey) {
        checkConnected();
        return CompletableFuture.supplyAsync(() -> {
            // In mock mode, look up the remote buffer by key
            MockRdmaBuffer remoteBuffer = parent.getBuffer(remoteKey);
            if (remoteBuffer != null) {
                // Perform actual copy if remote buffer exists (loopback scenario)
                MemorySegment src = remoteBuffer.segment().asSlice(
                        remoteAddress - remoteBuffer.address(), length);
                MemorySegment dst = localBuffer.segment().asSlice(localOffset, length);
                MemorySegment.copy(src, 0, dst, 0, length);
            }

            bytesReceived.addAndGet(length);
            readOps.incrementAndGet();
            parent.recordReceive(length);
            return null;
        });
    }

    @Override
    public CompletableFuture<Void> atomicCompareSwap(RdmaBuffer localBuffer, long remoteAddress,
                                                      long remoteKey, long expected, long desired) {
        checkConnected();
        return CompletableFuture.supplyAsync(() -> {
            MockRdmaBuffer remoteBuffer = parent.getBuffer(remoteKey);
            if (remoteBuffer != null) {
                MemorySegment remoteSeg = remoteBuffer.segment();
                long remoteOffset = remoteAddress - remoteBuffer.address();

                // Read current value
                long current = remoteSeg.get(java.lang.foreign.ValueLayout.JAVA_LONG, remoteOffset);

                // Write current to local buffer (return value)
                localBuffer.segment().set(java.lang.foreign.ValueLayout.JAVA_LONG, 0, current);

                // Conditionally update remote
                if (current == expected) {
                    remoteSeg.set(java.lang.foreign.ValueLayout.JAVA_LONG, remoteOffset, desired);
                }
            }

            atomicOps.incrementAndGet();
            parent.recordOperation();
            return null;
        });
    }

    @Override
    public CompletableFuture<Void> atomicFetchAdd(RdmaBuffer localBuffer, long remoteAddress,
                                                   long remoteKey, long delta) {
        checkConnected();
        return CompletableFuture.supplyAsync(() -> {
            MockRdmaBuffer remoteBuffer = parent.getBuffer(remoteKey);
            if (remoteBuffer != null) {
                MemorySegment remoteSeg = remoteBuffer.segment();
                long remoteOffset = remoteAddress - remoteBuffer.address();

                // Read current value
                long current = remoteSeg.get(java.lang.foreign.ValueLayout.JAVA_LONG, remoteOffset);

                // Write current to local buffer (return value)
                localBuffer.segment().set(java.lang.foreign.ValueLayout.JAVA_LONG, 0, current);

                // Update remote with sum
                remoteSeg.set(java.lang.foreign.ValueLayout.JAVA_LONG, remoteOffset, current + delta);
            }

            atomicOps.incrementAndGet();
            parent.recordOperation();
            return null;
        });
    }

    @Override
    public void flush() {
        checkConnected();
        // No-op in mock - all operations complete synchronously
    }

    @Override
    public boolean awaitCompletion(long timeoutMillis) {
        // In mock, all operations complete immediately
        return true;
    }

    @Override
    public EndpointStats stats() {
        return new EndpointStats(
                bytesSent.get(),
                bytesReceived.get(),
                sendOps.get(),
                recvOps.get(),
                writeOps.get(),
                readOps.get(),
                atomicOps.get(),
                errors.get()
        );
    }

    @Override
    public void close() {
        if (state != EndpointState.DISCONNECTED) {
            state = EndpointState.DISCONNECTED;
            parent.endpointClosed();
        }
    }

    private void checkConnected() {
        if (state != EndpointState.CONNECTED) {
            errors.incrementAndGet();
            throw new RdmaException("Endpoint is not connected: " + state,
                    RdmaException.ErrorCode.INVALID_STATE);
        }
    }
}
