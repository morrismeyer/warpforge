package io.surfworks.warpforge.io.rdma;

import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * RDMA abstraction for WarpForge high-performance networking.
 *
 * <p>This interface provides zero-copy data transfer between nodes using
 * RDMA (Remote Direct Memory Access) over InfiniBand or RoCE (RDMA over
 * Converged Ethernet) networks. All operations bypass the kernel for
 * minimum latency and maximum throughput.
 *
 * <h2>Implementations</h2>
 * <ul>
 *   <li><b>UcxRdmaImpl</b>: Production implementation using UCX (Unified
 *       Communication X) library with jextract-generated FFM bindings.
 *       Requires Linux with RDMA-capable hardware (e.g., Mellanox ConnectX).</li>
 *   <li><b>RdmaMock</b>: Mock implementation for development and testing
 *       on systems without RDMA hardware. All operations are local no-ops
 *       or simulated.</li>
 * </ul>
 *
 * <h2>Zero-Copy Guarantee</h2>
 * <p>Memory registered via {@link #registerMemory(MemorySegment)} is used
 * directly for RDMA operations without any copies. The MemorySegment can
 * be passed to WarpForge GPU backends that support FFM MemorySegment.
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * try (RdmaApi rdma = Rdma.load();
 *      Arena arena = Arena.ofConfined()) {
 *
 *     // Allocate and register memory
 *     MemorySegment segment = arena.allocate(1024 * 1024); // 1MB
 *     try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
 *
 *         // Connect to remote peer
 *         try (RdmaEndpoint endpoint = rdma.connect("192.168.1.100", 12345)) {
 *
 *             // Send data (zero-copy)
 *             endpoint.send(buffer).join();
 *         }
 *     }
 * }
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 * <p>RdmaApi instances are thread-safe for device queries and memory
 * registration. Individual endpoints have their own thread-safety
 * guarantees documented in {@link RdmaEndpoint}.
 *
 * @see Rdma#load()
 * @see RdmaBuffer
 * @see RdmaEndpoint
 */
public interface RdmaApi extends AutoCloseable {

    /**
     * Returns the backend name identifying this implementation.
     *
     * @return backend identifier (e.g., "ucx", "mock")
     */
    String backendName();

    /**
     * Returns the configuration used by this instance.
     *
     * @return RDMA configuration
     */
    RdmaConfig config();

    // ===== Device Management =====

    /**
     * Lists all available RDMA devices on this system.
     *
     * @return list of RDMA-capable devices
     */
    List<RdmaDevice> devices();

    /**
     * Returns the currently selected device.
     *
     * @return the active device, or null if none selected
     */
    RdmaDevice currentDevice();

    /**
     * Selects a specific device for RDMA operations.
     *
     * @param deviceName name of the device to use
     * @throws IllegalArgumentException if the device is not found
     */
    void selectDevice(String deviceName);

    // ===== Memory Registration =====

    /**
     * Registers a memory segment for RDMA operations.
     *
     * <p>The memory is pinned (page-locked) and registered with the RDMA
     * device. This is a relatively expensive operation (~100us), so it's
     * recommended to register large buffers and reuse them.
     *
     * <p>The returned {@link RdmaBuffer} wraps the original segment.
     * Zero-copy is guaranteed - the same memory is used for RDMA operations.
     *
     * @param segment the memory segment to register
     * @return RDMA-registered buffer wrapping the segment
     * @throws RdmaException if registration fails
     */
    RdmaBuffer registerMemory(MemorySegment segment);

    /**
     * Registers a memory segment with specific access flags.
     *
     * @param segment the memory segment to register
     * @param flags registration flags (see {@link MemoryFlags})
     * @return RDMA-registered buffer wrapping the segment
     * @throws RdmaException if registration fails
     */
    RdmaBuffer registerMemory(MemorySegment segment, int flags);

    /**
     * Unregisters a previously registered buffer.
     *
     * <p>This is equivalent to calling {@link RdmaBuffer#close()}.
     *
     * @param buffer the buffer to unregister
     */
    void unregisterMemory(RdmaBuffer buffer);

    // ===== Connection Management =====

    /**
     * Connects to a remote RDMA peer.
     *
     * <p>This establishes a reliable connection (RC in IB terms) to the
     * specified address and port. The connection is bidirectional.
     *
     * @param remoteAddress IP address or hostname of the remote peer
     * @param port port number on the remote peer
     * @return connected endpoint
     * @throws RdmaException if connection fails
     */
    RdmaEndpoint connect(String remoteAddress, int port);

    /**
     * Connects to a remote RDMA peer with timeout.
     *
     * @param remoteAddress IP address or hostname of the remote peer
     * @param port port number on the remote peer
     * @param timeoutMillis connection timeout in milliseconds
     * @return connected endpoint
     * @throws RdmaException if connection fails or times out
     */
    RdmaEndpoint connect(String remoteAddress, int port, long timeoutMillis);

    /**
     * Creates a listener for incoming RDMA connections.
     *
     * @param port local port to listen on
     * @return listener for accepting connections
     * @throws RdmaException if listener creation fails
     */
    RdmaListener listen(int port);

    // ===== Capability Queries =====

    /**
     * Returns whether GPU Direct RDMA is supported.
     *
     * <p>If true, GPU memory (allocated via CUDA/ROCm) can be registered
     * for RDMA operations without copying to host memory first.
     *
     * @return true if GPU Direct RDMA is available
     */
    boolean supportsGpuDirect();

    /**
     * Returns whether atomic operations are supported.
     *
     * @return true if atomic compare-swap and fetch-add are available
     */
    boolean supportsAtomics();

    /**
     * Returns the maximum message size for inline sends.
     *
     * <p>Messages smaller than this size can be sent without DMA,
     * reducing latency.
     *
     * @return maximum inline message size in bytes
     */
    int maxInlineSize();

    /**
     * Returns the maximum memory region size that can be registered.
     *
     * @return maximum registrable memory in bytes
     */
    long maxMemoryRegionSize();

    // ===== Statistics =====

    /**
     * Returns aggregate statistics for this RDMA context.
     *
     * @return context statistics
     */
    RdmaStats stats();

    /**
     * Closes this RDMA context and releases all resources.
     *
     * <p>All registered buffers and endpoints associated with this context
     * become invalid.
     */
    @Override
    void close();

    /**
     * Memory registration flags.
     */
    interface MemoryFlags {
        /** Memory can be read by remote peers */
        int REMOTE_READ = 1;
        /** Memory can be written by remote peers */
        int REMOTE_WRITE = 2;
        /** Memory can be used for local send operations */
        int LOCAL_WRITE = 4;
        /** Memory can be used for atomic operations */
        int REMOTE_ATOMIC = 8;
        /** Default flags: all permissions */
        int DEFAULT = REMOTE_READ | REMOTE_WRITE | LOCAL_WRITE | REMOTE_ATOMIC;
    }

    /**
     * Aggregate statistics for an RDMA context.
     */
    record RdmaStats(
            long totalBytesSent,
            long totalBytesReceived,
            long totalOperations,
            long activeEndpoints,
            long registeredMemoryBytes,
            long registrationCount
    ) {
        public static RdmaStats zero() {
            return new RdmaStats(0, 0, 0, 0, 0, 0);
        }
    }
}
