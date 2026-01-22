package io.surfworks.warpforge.io.collective.impl;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import java.util.logging.Logger;

import io.surfworks.warpforge.io.collective.CollectiveConfig;
import io.surfworks.warpforge.io.collective.CollectiveException;
import io.surfworks.warpforge.io.ffi.ucc.ucc_oob_coll;

/**
 * TCP-based out-of-band coordination for UCC team formation.
 *
 * <p>UCC requires an out-of-band (OOB) allgather mechanism during team creation
 * to exchange endpoint information between all participating ranks. This class
 * implements that mechanism using TCP sockets.
 *
 * <h2>Protocol</h2>
 * <p>Rank 0 acts as the coordinator (server), all other ranks connect as clients.
 * The allgather protocol is:
 * <ol>
 *   <li>Each rank sends its local data to rank 0</li>
 *   <li>Rank 0 gathers all data and broadcasts the combined result</li>
 *   <li>All ranks receive the same combined buffer</li>
 * </ol>
 *
 * <h2>Threading</h2>
 * <p>OOB callbacks are invoked synchronously by UCC during team creation.
 * The coordinator uses blocking I/O within these callbacks.
 */
public class OobCoordinator implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(OobCoordinator.class.getName());
    private static final int CONNECT_TIMEOUT_MS = 30000;
    private static final int READ_TIMEOUT_MS = 60000;

    private final CollectiveConfig config;
    private final Arena arena;

    // Network state
    private ServerSocket serverSocket;  // Only on rank 0
    private Socket[] peerSockets;       // Rank 0: connections from all ranks; Others: connection to rank 0
    private DataInputStream[] peerInputs;
    private DataOutputStream[] peerOutputs;

    // Request tracking
    private final ConcurrentHashMap<Long, OobRequest> pendingRequests = new ConcurrentHashMap<>();
    private final AtomicLong requestIdGenerator = new AtomicLong(1);

    // Upcall stubs (allocated in arena, valid for arena lifetime)
    private MemorySegment allgatherStub;
    private MemorySegment reqTestStub;
    private MemorySegment reqFreeStub;

    // The allocated OOB structure
    private MemorySegment oobStruct;

    private volatile boolean closed = false;

    /**
     * Create a new OOB coordinator.
     *
     * @param config the collective configuration
     * @param arena the memory arena for FFM allocations
     */
    public OobCoordinator(CollectiveConfig config, Arena arena) {
        this.config = config;
        this.arena = arena;

        try {
            initializeNetwork();
            createUpcallStubs();
            createOobStruct();
        } catch (IOException e) {
            throw new CollectiveException(
                "Failed to initialize OOB coordinator: " + e.getMessage(),
                CollectiveException.ErrorCode.COMMUNICATION_ERROR,
                e
            );
        }
    }

    /**
     * Get the allocated OOB structure for use in UCC team params.
     *
     * @return the ucc_oob_coll structure
     */
    public MemorySegment getOobStruct() {
        return oobStruct;
    }

    // ========================================================================
    // Network Initialization
    // ========================================================================

    private void initializeNetwork() throws IOException {
        int worldSize = config.worldSize();
        int rank = config.rank();

        peerSockets = new Socket[worldSize];
        peerInputs = new DataInputStream[worldSize];
        peerOutputs = new DataOutputStream[worldSize];

        if (rank == 0) {
            // Rank 0 is the server - accept connections from all other ranks
            initializeServer();
        } else {
            // Other ranks connect to rank 0
            initializeClient();
        }

        LOG.info("OOB coordinator initialized: rank=" + rank + ", worldSize=" + worldSize);
    }

    private void initializeServer() throws IOException {
        int worldSize = config.worldSize();

        serverSocket = new ServerSocket();
        serverSocket.setReuseAddress(true);
        serverSocket.bind(new InetSocketAddress(config.masterPort()));

        LOG.info("OOB server listening on port " + config.masterPort());

        // Accept connections from all other ranks
        for (int i = 1; i < worldSize; i++) {
            Socket socket = serverSocket.accept();
            socket.setSoTimeout(READ_TIMEOUT_MS);
            socket.setTcpNoDelay(true);

            DataInputStream in = new DataInputStream(socket.getInputStream());
            DataOutputStream out = new DataOutputStream(socket.getOutputStream());

            // Read the connecting rank's ID
            int peerRank = in.readInt();
            if (peerRank < 1 || peerRank >= worldSize) {
                socket.close();
                throw new IOException("Invalid peer rank: " + peerRank);
            }

            peerSockets[peerRank] = socket;
            peerInputs[peerRank] = in;
            peerOutputs[peerRank] = out;

            LOG.fine("Accepted connection from rank " + peerRank);
        }
    }

    private void initializeClient() throws IOException {
        int rank = config.rank();

        Socket socket = new Socket();
        socket.setTcpNoDelay(true);
        socket.connect(
            new InetSocketAddress(config.masterAddress(), config.masterPort()),
            CONNECT_TIMEOUT_MS
        );
        socket.setSoTimeout(READ_TIMEOUT_MS);

        DataInputStream in = new DataInputStream(socket.getInputStream());
        DataOutputStream out = new DataOutputStream(socket.getOutputStream());

        // Send our rank to the server
        out.writeInt(rank);
        out.flush();

        peerSockets[0] = socket;
        peerInputs[0] = in;
        peerOutputs[0] = out;

        LOG.fine("Connected to master at " + config.masterAddress() + ":" + config.masterPort());
    }

    // ========================================================================
    // Upcall Stubs
    // ========================================================================

    private void createUpcallStubs() {
        // Create upcall stubs that UCC will call during team formation
        allgatherStub = ucc_oob_coll.allgather.allocate(this::allgatherCallback, arena);
        reqTestStub = ucc_oob_coll.req_test.allocate(this::reqTestCallback, arena);
        reqFreeStub = ucc_oob_coll.req_free.allocate(this::reqFreeCallback, arena);
    }

    private void createOobStruct() {
        oobStruct = ucc_oob_coll.allocate(arena);

        ucc_oob_coll.allgather(oobStruct, allgatherStub);
        ucc_oob_coll.req_test(oobStruct, reqTestStub);
        ucc_oob_coll.req_free(oobStruct, reqFreeStub);
        ucc_oob_coll.coll_info(oobStruct, MemorySegment.NULL); // Not used
        ucc_oob_coll.n_oob_eps(oobStruct, config.worldSize());
        ucc_oob_coll.oob_ep(oobStruct, config.rank());
    }

    // ========================================================================
    // OOB Callbacks
    // ========================================================================

    /**
     * Allgather callback invoked by UCC during team formation.
     *
     * <p>Parameters:
     * <ul>
     *   <li>sbuf - Send buffer (this rank's data)</li>
     *   <li>rbuf - Receive buffer (combined data from all ranks)</li>
     *   <li>size - Size of each rank's contribution</li>
     *   <li>collInfo - User context (unused)</li>
     *   <li>reqPtr - Pointer to store request handle</li>
     * </ul>
     *
     * @return UCC status code
     */
    private int allgatherCallback(MemorySegment sbuf, MemorySegment rbuf, long size,
                                   MemorySegment collInfo, MemorySegment reqPtr) {
        if (closed) {
            return UccConstants.ERR_INVALID_PARAM;
        }

        try {
            // Create a request to track this operation
            long reqId = requestIdGenerator.getAndIncrement();
            OobRequest request = new OobRequest(reqId, rbuf, size);
            pendingRequests.put(reqId, request);

            // Store the request ID in the pointer
            MemorySegment reqIdSegment = arena.allocate(ValueLayout.JAVA_LONG);
            reqIdSegment.set(ValueLayout.JAVA_LONG, 0, reqId);
            reqPtr.set(ValueLayout.ADDRESS, 0, reqIdSegment);

            // Perform the allgather synchronously
            performAllgather(sbuf, rbuf, size);

            // Mark as complete
            request.complete();

            return UccConstants.OK;
        } catch (Exception e) {
            LOG.log(Level.SEVERE, "OOB allgather failed", e);
            return UccConstants.ERR_IO_ERROR;
        }
    }

    /**
     * Test if a request is complete.
     *
     * @param req the request handle
     * @return UCC_OK if complete, UCC_INPROGRESS if pending
     */
    private int reqTestCallback(MemorySegment req) {
        if (UccHelper.isNull(req)) {
            return UccConstants.ERR_INVALID_PARAM;
        }

        try {
            long reqId = req.get(ValueLayout.JAVA_LONG, 0);
            OobRequest request = pendingRequests.get(reqId);

            if (request == null) {
                // Request not found - assume it was already completed and freed
                return UccConstants.OK;
            }

            return request.isComplete() ? UccConstants.OK : UccConstants.INPROGRESS;
        } catch (Exception e) {
            LOG.log(Level.WARNING, "req_test failed", e);
            return UccConstants.ERR_INVALID_PARAM;
        }
    }

    /**
     * Free a completed request.
     *
     * @param req the request handle
     * @return UCC_OK on success
     */
    private int reqFreeCallback(MemorySegment req) {
        if (UccHelper.isNull(req)) {
            return UccConstants.OK; // Freeing null is OK
        }

        try {
            long reqId = req.get(ValueLayout.JAVA_LONG, 0);
            pendingRequests.remove(reqId);
            return UccConstants.OK;
        } catch (Exception e) {
            LOG.log(Level.WARNING, "req_free failed", e);
            return UccConstants.ERR_INVALID_PARAM;
        }
    }

    // ========================================================================
    // Allgather Protocol
    // ========================================================================

    private void performAllgather(MemorySegment sbuf, MemorySegment rbuf, long size) throws IOException {
        int worldSize = config.worldSize();
        int rank = config.rank();

        if (worldSize == 1) {
            // Single rank - just copy
            MemorySegment.copy(sbuf, 0, rbuf, 0, size);
            return;
        }

        if (rank == 0) {
            performAllgatherAsRoot(sbuf, rbuf, size);
        } else {
            performAllgatherAsWorker(sbuf, rbuf, size);
        }
    }

    private void performAllgatherAsRoot(MemorySegment sbuf, MemorySegment rbuf, long size) throws IOException {
        int worldSize = config.worldSize();

        // Copy our own data to position 0
        MemorySegment.copy(sbuf, 0, rbuf, 0, size);

        // Receive data from all other ranks
        byte[] buffer = new byte[(int) size];
        for (int r = 1; r < worldSize; r++) {
            // Read data from rank r
            peerInputs[r].readFully(buffer);
            // Copy to position r in result buffer
            MemorySegment.copy(MemorySegment.ofArray(buffer), 0, rbuf, r * size, size);
        }

        // Broadcast combined result to all other ranks
        byte[] result = new byte[(int) (worldSize * size)];
        MemorySegment.copy(rbuf, 0, MemorySegment.ofArray(result), 0, worldSize * size);

        for (int r = 1; r < worldSize; r++) {
            peerOutputs[r].write(result);
            peerOutputs[r].flush();
        }
    }

    private void performAllgatherAsWorker(MemorySegment sbuf, MemorySegment rbuf, long size) throws IOException {
        int worldSize = config.worldSize();

        // Send our data to rank 0
        byte[] sendBuffer = new byte[(int) size];
        MemorySegment.copy(sbuf, 0, MemorySegment.ofArray(sendBuffer), 0, size);
        peerOutputs[0].write(sendBuffer);
        peerOutputs[0].flush();

        // Receive combined result from rank 0
        byte[] recvBuffer = new byte[(int) (worldSize * size)];
        peerInputs[0].readFully(recvBuffer);
        MemorySegment.copy(MemorySegment.ofArray(recvBuffer), 0, rbuf, 0, worldSize * size);
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;

        // Close all peer connections
        if (peerSockets != null) {
            for (Socket socket : peerSockets) {
                if (socket != null) {
                    try {
                        socket.close();
                    } catch (IOException e) {
                        LOG.log(Level.FINE, "Error closing peer socket", e);
                    }
                }
            }
        }

        // Close server socket
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException e) {
                LOG.log(Level.FINE, "Error closing server socket", e);
            }
        }

        // Clear pending requests
        pendingRequests.clear();

        LOG.fine("OOB coordinator closed");
    }

    // ========================================================================
    // Request Tracking
    // ========================================================================

    /**
     * Represents a pending OOB allgather request.
     */
    private static class OobRequest {
        private final long id;
        private final MemorySegment rbuf;
        private final long size;
        private volatile boolean complete = false;

        OobRequest(long id, MemorySegment rbuf, long size) {
            this.id = id;
            this.rbuf = rbuf;
            this.size = size;
        }

        void complete() {
            this.complete = true;
        }

        boolean isComplete() {
            return complete;
        }
    }
}
