package io.surfworks.warpforge.io.rdma;

import java.util.concurrent.CompletableFuture;

/**
 * RDMA endpoint representing a connection to a remote peer.
 *
 * <p>Endpoints support both two-sided (send/receive) and one-sided (read/write)
 * RDMA operations. All operations are asynchronous and return
 * {@link CompletableFuture} for completion notification.
 *
 * <h2>Operation Types</h2>
 * <ul>
 *   <li><b>Send/Receive</b>: Two-sided operations requiring both peers to
 *       participate. The receiver must post a receive buffer before the
 *       sender transmits.</li>
 *   <li><b>Read/Write</b>: One-sided operations where only the initiator
 *       participates. The remote peer's CPU is not involved (zero-copy,
 *       kernel-bypass). Requires remote address and key.</li>
 * </ul>
 *
 * <h2>ibverbs Operations Exposed</h2>
 * <p>This interface maps to the following ibverbs operations:
 * <ul>
 *   <li>{@link #send} - IBV_WR_SEND</li>
 *   <li>{@link #receive} - ibv_post_recv</li>
 *   <li>{@link #write} - IBV_WR_RDMA_WRITE</li>
 *   <li>{@link #writeImmediate} - IBV_WR_RDMA_WRITE_WITH_IMM</li>
 *   <li>{@link #read} - IBV_WR_RDMA_READ</li>
 *   <li>{@link #atomicCompareSwap} - IBV_WR_ATOMIC_CMP_AND_SWP</li>
 *   <li>{@link #atomicFetchAdd} - IBV_WR_ATOMIC_FETCH_AND_ADD</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <p>Endpoint operations are thread-safe. Multiple threads can submit
 * operations concurrently. Completion ordering follows RDMA semantics.
 */
public interface RdmaEndpoint extends AutoCloseable {

    /**
     * Returns the remote peer's address.
     *
     * @return remote address string (host:port or GID format)
     */
    String remoteAddress();

    /**
     * Returns the current connection state.
     *
     * @return current endpoint state
     */
    EndpointState state();

    // ===== Two-Sided Operations (Send/Receive) =====

    /**
     * Sends data to the remote peer.
     *
     * <p>The remote peer must have posted a receive buffer before this
     * operation completes. This is an RDMA send operation (IBV_WR_SEND).
     *
     * @param buffer the buffer containing data to send
     * @return future that completes when the send is acknowledged
     * @throws IllegalStateException if the endpoint is not connected
     */
    CompletableFuture<Void> send(RdmaBuffer buffer);

    /**
     * Sends data to the remote peer with offset and length.
     *
     * @param buffer the buffer containing data to send
     * @param offset offset within the buffer
     * @param length number of bytes to send
     * @return future that completes when the send is acknowledged
     */
    CompletableFuture<Void> send(RdmaBuffer buffer, long offset, long length);

    /**
     * Posts a receive buffer to accept incoming data.
     *
     * <p>The buffer will be filled when a remote peer sends data.
     * Returns the number of bytes actually received.
     *
     * @param buffer the buffer to receive data into
     * @return future that completes with the number of bytes received
     */
    CompletableFuture<Long> receive(RdmaBuffer buffer);

    // ===== One-Sided Operations (Read/Write) =====

    /**
     * Writes data to a remote buffer (RDMA write).
     *
     * <p>This is a one-sided operation - the remote CPU is not involved.
     * The remote peer's memory is directly modified via DMA.
     *
     * @param localBuffer local buffer containing data to write
     * @param remoteAddress virtual address in the remote peer's memory
     * @param remoteKey remote memory key for access authorization
     * @return future that completes when the write is complete
     */
    CompletableFuture<Void> write(RdmaBuffer localBuffer, long remoteAddress, long remoteKey);

    /**
     * Writes data to a remote buffer with offset and length.
     *
     * @param localBuffer local buffer containing data to write
     * @param localOffset offset within the local buffer
     * @param length number of bytes to write
     * @param remoteAddress virtual address in the remote peer's memory
     * @param remoteKey remote memory key for access authorization
     * @return future that completes when the write is complete
     */
    CompletableFuture<Void> write(RdmaBuffer localBuffer, long localOffset, long length,
                                   long remoteAddress, long remoteKey);

    /**
     * Writes data to a remote buffer with immediate data.
     *
     * <p>Like {@link #write}, but also delivers a 32-bit immediate value
     * to the remote peer. The remote peer receives this via a receive
     * completion with the immediate flag set.
     *
     * @param localBuffer local buffer containing data to write
     * @param remoteAddress virtual address in the remote peer's memory
     * @param remoteKey remote memory key for access authorization
     * @param immediate 32-bit immediate value delivered to remote
     * @return future that completes when the write is complete
     */
    CompletableFuture<Void> writeImmediate(RdmaBuffer localBuffer, long remoteAddress,
                                            long remoteKey, int immediate);

    /**
     * Reads data from a remote buffer (RDMA read).
     *
     * <p>This is a one-sided operation - the remote CPU is not involved.
     * Data is read directly from the remote peer's memory via DMA.
     *
     * @param localBuffer local buffer to receive the read data
     * @param remoteAddress virtual address in the remote peer's memory
     * @param remoteKey remote memory key for access authorization
     * @return future that completes when the read data is available
     */
    CompletableFuture<Void> read(RdmaBuffer localBuffer, long remoteAddress, long remoteKey);

    /**
     * Reads data from a remote buffer with offset and length.
     *
     * @param localBuffer local buffer to receive the read data
     * @param localOffset offset within the local buffer
     * @param length number of bytes to read
     * @param remoteAddress virtual address in the remote peer's memory
     * @param remoteKey remote memory key for access authorization
     * @return future that completes when the read data is available
     */
    CompletableFuture<Void> read(RdmaBuffer localBuffer, long localOffset, long length,
                                  long remoteAddress, long remoteKey);

    // ===== Atomic Operations =====

    /**
     * Performs an atomic compare-and-swap on a remote 64-bit value.
     *
     * <p>Atomically compares the value at the remote address with
     * {@code expected}, and if equal, replaces it with {@code desired}.
     * The original value is returned regardless of success.
     *
     * @param localBuffer local buffer to receive the original value (8 bytes)
     * @param remoteAddress virtual address of the 64-bit value (must be 8-byte aligned)
     * @param remoteKey remote memory key for access authorization
     * @param expected value expected at the remote address
     * @param desired value to write if comparison succeeds
     * @return future that completes when the operation finishes
     */
    CompletableFuture<Void> atomicCompareSwap(RdmaBuffer localBuffer, long remoteAddress,
                                               long remoteKey, long expected, long desired);

    /**
     * Performs an atomic fetch-and-add on a remote 64-bit value.
     *
     * <p>Atomically adds {@code delta} to the value at the remote address.
     * The original value (before addition) is returned.
     *
     * @param localBuffer local buffer to receive the original value (8 bytes)
     * @param remoteAddress virtual address of the 64-bit value (must be 8-byte aligned)
     * @param remoteKey remote memory key for access authorization
     * @param delta value to add
     * @return future that completes when the operation finishes
     */
    CompletableFuture<Void> atomicFetchAdd(RdmaBuffer localBuffer, long remoteAddress,
                                            long remoteKey, long delta);

    // ===== Control Operations =====

    /**
     * Flushes all pending operations.
     *
     * <p>Ensures all previously submitted operations are visible to
     * the remote peer. This is a fence operation.
     */
    void flush();

    /**
     * Waits for all pending operations to complete.
     *
     * <p>Blocks until all previously submitted operations have completed.
     * This includes both local completion and remote acknowledgment where
     * applicable.
     *
     * @param timeoutMillis maximum time to wait in milliseconds
     * @return true if all operations completed, false if timeout occurred
     */
    boolean awaitCompletion(long timeoutMillis);

    /**
     * Returns statistics for this endpoint.
     *
     * @return endpoint statistics
     */
    EndpointStats stats();

    /**
     * Closes this endpoint, disconnecting from the remote peer.
     */
    @Override
    void close();

    /**
     * Endpoint connection states.
     */
    enum EndpointState {
        /** Initial state, not yet connected */
        CREATED,
        /** Connection in progress */
        CONNECTING,
        /** Fully connected and operational */
        CONNECTED,
        /** Disconnect in progress */
        DISCONNECTING,
        /** Disconnected (gracefully or due to error) */
        DISCONNECTED,
        /** Error state, endpoint is unusable */
        ERROR
    }

    /**
     * Statistics for an endpoint.
     */
    record EndpointStats(
            long bytesSent,
            long bytesReceived,
            long sendOperations,
            long receiveOperations,
            long writeOperations,
            long readOperations,
            long atomicOperations,
            long errors
    ) {
        public static EndpointStats zero() {
            return new EndpointStats(0, 0, 0, 0, 0, 0, 0, 0);
        }
    }
}
