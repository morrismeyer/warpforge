package io.surfworks.warpforge.io.rdma.impl;

import io.surfworks.warpforge.io.VirtualThreads;
import io.surfworks.warpforge.io.rdma.RdmaBuffer;
import io.surfworks.warpforge.io.rdma.RdmaEndpoint;
import io.surfworks.warpforge.io.rdma.RdmaException;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * UCX implementation of RdmaEndpoint.
 *
 * <p>Wraps a UCX endpoint (ucp_ep_h) for RDMA operations.
 * Async operations use virtual threads for better scalability.
 */
final class UcxRdmaEndpoint implements RdmaEndpoint {

    private final String remoteAddress;
    private final int port;
    private final UcxRdmaImpl parent;
    private final MemorySegment ucpEndpoint; // UCX endpoint handle

    private volatile EndpointState state = EndpointState.CONNECTED;
    private final AtomicLong bytesSent = new AtomicLong();
    private final AtomicLong bytesReceived = new AtomicLong();
    private final AtomicLong sendOps = new AtomicLong();
    private final AtomicLong recvOps = new AtomicLong();
    private final AtomicLong writeOps = new AtomicLong();
    private final AtomicLong readOps = new AtomicLong();
    private final AtomicLong atomicOps = new AtomicLong();
    private final AtomicLong errors = new AtomicLong();

    UcxRdmaEndpoint(String remoteAddress, int port, UcxRdmaImpl parent, MemorySegment ucpEndpoint) {
        this.remoteAddress = remoteAddress;
        this.port = port;
        this.parent = parent;
        this.ucpEndpoint = ucpEndpoint;
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
        return VirtualThreads.supplyAsync(() -> {
            // TODO: Use ucp_tag_send_nb() or ucp_tag_send_sync_nb()
            bytesSent.addAndGet(length);
            sendOps.incrementAndGet();
            parent.recordSend(length);
            return null;
        });
    }

    @Override
    public CompletableFuture<Long> receive(RdmaBuffer buffer) {
        checkConnected();
        return VirtualThreads.supplyAsync(() -> {
            // TODO: Use ucp_tag_recv_nb()
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
        return VirtualThreads.supplyAsync(() -> {
            // TODO: Use ucp_put_nb() for RDMA write
            bytesSent.addAndGet(length);
            writeOps.incrementAndGet();
            parent.recordSend(length);
            return null;
        });
    }

    @Override
    public CompletableFuture<Void> writeImmediate(RdmaBuffer localBuffer, long remoteAddress,
                                                   long remoteKey, int immediate) {
        // UCX doesn't have direct write-with-immediate, use regular write
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
        return VirtualThreads.supplyAsync(() -> {
            // TODO: Use ucp_get_nb() for RDMA read
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
        return VirtualThreads.supplyAsync(() -> {
            // TODO: Use ucp_atomic_cswap64()
            atomicOps.incrementAndGet();
            return null;
        });
    }

    @Override
    public CompletableFuture<Void> atomicFetchAdd(RdmaBuffer localBuffer, long remoteAddress,
                                                   long remoteKey, long delta) {
        checkConnected();
        return VirtualThreads.supplyAsync(() -> {
            // TODO: Use ucp_atomic_fadd64()
            atomicOps.incrementAndGet();
            return null;
        });
    }

    @Override
    public void flush() {
        checkConnected();
        // TODO: Use ucp_ep_flush_nb() or ucp_worker_flush()
    }

    @Override
    public boolean awaitCompletion(long timeoutMillis) {
        // TODO: Progress UCX worker until all operations complete
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
            // TODO: Use ucp_ep_close_nb() to close endpoint
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
