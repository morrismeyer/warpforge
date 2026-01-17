package io.surfworks.warpforge.io.rdma;

import java.util.concurrent.CompletableFuture;

/**
 * Listener for incoming RDMA connections.
 *
 * <p>An RdmaListener binds to a local port and accepts incoming
 * connections from remote peers.
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * try (RdmaApi rdma = Rdma.load();
 *      RdmaListener listener = rdma.listen(12345)) {
 *
 *     // Accept connections in a loop
 *     while (running) {
 *         try (RdmaEndpoint endpoint = listener.accept()) {
 *             // Handle connection...
 *         }
 *     }
 * }
 * }</pre>
 */
public interface RdmaListener extends AutoCloseable {

    /**
     * Returns the local port this listener is bound to.
     *
     * @return local port number
     */
    int port();

    /**
     * Returns the local address this listener is bound to.
     *
     * @return local address string
     */
    String localAddress();

    /**
     * Accepts an incoming connection (blocking).
     *
     * <p>Blocks until a remote peer connects.
     *
     * @return endpoint for the accepted connection
     * @throws RdmaException if accept fails
     */
    RdmaEndpoint accept();

    /**
     * Accepts an incoming connection with timeout.
     *
     * @param timeoutMillis maximum time to wait in milliseconds
     * @return endpoint for the accepted connection, or null if timeout
     * @throws RdmaException if accept fails
     */
    RdmaEndpoint accept(long timeoutMillis);

    /**
     * Accepts an incoming connection asynchronously.
     *
     * @return future that completes with the accepted endpoint
     */
    CompletableFuture<RdmaEndpoint> acceptAsync();

    /**
     * Returns whether this listener is still active.
     *
     * @return true if the listener can accept connections
     */
    boolean isActive();

    /**
     * Closes this listener, stopping it from accepting new connections.
     */
    @Override
    void close();
}
