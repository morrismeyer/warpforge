package io.surfworks.warpforge.io.rdma.impl;

import io.surfworks.warpforge.io.VirtualThreads;
import io.surfworks.warpforge.io.rdma.RdmaEndpoint;
import io.surfworks.warpforge.io.rdma.RdmaException;
import io.surfworks.warpforge.io.rdma.RdmaListener;

import java.util.concurrent.CompletableFuture;

/**
 * UCX implementation of RdmaListener.
 * Async operations use virtual threads for better scalability.
 */
final class UcxRdmaListener implements RdmaListener {

    private final int port;
    private final UcxRdmaImpl parent;
    private volatile boolean active = true;

    UcxRdmaListener(int port, UcxRdmaImpl parent) {
        this.port = port;
        this.parent = parent;
        // TODO: Set up UCX listener using ucp_listener_create()
    }

    @Override
    public int port() {
        return port;
    }

    @Override
    public String localAddress() {
        return "0.0.0.0:" + port;
    }

    @Override
    public RdmaEndpoint accept() {
        checkActive();
        // TODO: Accept incoming connection via UCX connection handler
        throw new RdmaException("UCX listener not yet implemented",
                RdmaException.ErrorCode.NOT_SUPPORTED);
    }

    @Override
    public RdmaEndpoint accept(long timeoutMillis) {
        checkActive();
        // TODO: Accept with timeout
        throw new RdmaException("UCX listener not yet implemented",
                RdmaException.ErrorCode.NOT_SUPPORTED);
    }

    @Override
    public CompletableFuture<RdmaEndpoint> acceptAsync() {
        return VirtualThreads.supplyAsync(this::accept);
    }

    @Override
    public boolean isActive() {
        return active;
    }

    @Override
    public void close() {
        active = false;
        // TODO: Use ucp_listener_destroy()
    }

    private void checkActive() {
        if (!active) {
            throw new RdmaException("Listener has been closed", RdmaException.ErrorCode.INVALID_STATE);
        }
    }
}
