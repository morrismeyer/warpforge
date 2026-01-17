package io.surfworks.warpforge.io.rdma.mock;

import io.surfworks.warpforge.io.rdma.RdmaEndpoint;
import io.surfworks.warpforge.io.rdma.RdmaException;
import io.surfworks.warpforge.io.rdma.RdmaListener;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Mock implementation of RdmaListener for testing.
 */
final class MockRdmaListener implements RdmaListener {

    private final int port;
    private final RdmaMock parent;
    private final LinkedBlockingQueue<MockRdmaEndpoint> pendingConnections = new LinkedBlockingQueue<>();
    private volatile boolean active = true;

    MockRdmaListener(int port, RdmaMock parent) {
        this.port = port;
        this.parent = parent;
    }

    @Override
    public int port() {
        return port;
    }

    @Override
    public String localAddress() {
        return "127.0.0.1:" + port;
    }

    @Override
    public RdmaEndpoint accept() {
        checkActive();
        try {
            MockRdmaEndpoint endpoint = pendingConnections.take();
            parent.recordOperation();
            return endpoint;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RdmaException("Interrupted while accepting connection",
                    RdmaException.ErrorCode.CANCELLED);
        }
    }

    @Override
    public RdmaEndpoint accept(long timeoutMillis) {
        checkActive();
        try {
            MockRdmaEndpoint endpoint = pendingConnections.poll(timeoutMillis, TimeUnit.MILLISECONDS);
            if (endpoint != null) {
                parent.recordOperation();
            }
            return endpoint;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RdmaException("Interrupted while accepting connection",
                    RdmaException.ErrorCode.CANCELLED);
        }
    }

    @Override
    public CompletableFuture<RdmaEndpoint> acceptAsync() {
        return CompletableFuture.supplyAsync(this::accept);
    }

    @Override
    public boolean isActive() {
        return active;
    }

    @Override
    public void close() {
        active = false;
        // Wake up any blocked accept() calls
        pendingConnections.offer(null);
    }

    /**
     * Simulates an incoming connection (for testing).
     */
    void simulateIncomingConnection(String remoteAddress, int remotePort) {
        if (active) {
            pendingConnections.offer(new MockRdmaEndpoint(remoteAddress, remotePort, parent));
        }
    }

    private void checkActive() {
        if (!active) {
            throw new RdmaException("Listener has been closed", RdmaException.ErrorCode.INVALID_STATE);
        }
    }
}
