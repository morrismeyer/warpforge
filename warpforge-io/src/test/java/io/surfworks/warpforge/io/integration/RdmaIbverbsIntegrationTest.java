package io.surfworks.warpforge.io.integration;

import io.surfworks.warpforge.io.rdma.*;
import org.junit.jupiter.api.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Ray-launched integration tests for all ibverbs operations exposed through Java APIs.
 *
 * <p>These tests verify that each RDMA operation works correctly over the actual
 * Mellanox 100GbE network between Mark 1 lab nodes.
 *
 * <h2>ibverbs Operations Tested</h2>
 * <ul>
 *   <li>IBV_WR_SEND - {@link #testSend()}</li>
 *   <li>ibv_post_recv - {@link #testReceive()}</li>
 *   <li>IBV_WR_RDMA_WRITE - {@link #testRdmaWrite()}</li>
 *   <li>IBV_WR_RDMA_WRITE_WITH_IMM - {@link #testRdmaWriteImmediate()}</li>
 *   <li>IBV_WR_RDMA_READ - {@link #testRdmaRead()}</li>
 *   <li>IBV_WR_ATOMIC_CMP_AND_SWP - {@link #testAtomicCompareSwap()}</li>
 *   <li>IBV_WR_ATOMIC_FETCH_AND_ADD - {@link #testAtomicFetchAdd()}</li>
 * </ul>
 *
 * <h2>Running</h2>
 * <pre>{@code
 * ./gradlew :warpforge-io:rayIntegrationTest \
 *     --tests "*RdmaIbverbsIntegrationTest*" \
 *     -Drdma.world.size=2
 * }</pre>
 */
@Tag("ray-integration")
@Tag("rdma")
@DisplayName("RDMA ibverbs Integration Tests")
class RdmaIbverbsIntegrationTest extends RayIntegrationTestBase {

    private Arena arena;
    private RdmaEndpoint endpoint;

    // Server port for tests
    private static final int TEST_PORT = 18515;

    @BeforeEach
    void setUpTest() {
        arena = Arena.ofConfined();

        // Rank 0 listens, Rank 1 connects
        if (rank == 0) {
            log("Starting listener on port %d", TEST_PORT);
            try (RdmaListener listener = rdma.listen(TEST_PORT)) {
                // Signal ready
                sync();
                // Accept connection
                endpoint = listener.accept(30000);
                assertNotNull(endpoint, "Failed to accept connection");
                log("Accepted connection from %s", endpoint.remoteAddress());
            }
        } else {
            // Wait for server to be ready
            sync();
            log("Connecting to %s:%d", masterAddress, TEST_PORT);
            endpoint = rdma.connect(masterAddress, TEST_PORT, 30000);
            assertNotNull(endpoint, "Failed to connect");
            log("Connected to %s", endpoint.remoteAddress());
        }

        // Sync after connection established
        sync();
    }

    @AfterEach
    void tearDownTest() {
        if (endpoint != null) endpoint.close();
        if (arena != null) arena.close();
    }

    // ===== Send/Receive Tests (Two-Sided) =====

    @Test
    @DisplayName("IBV_WR_SEND: Two-sided send operation")
    void testSend() throws Exception {
        final int SIZE = 4096;
        final byte PATTERN = 0x42;

        MemorySegment segment = arena.allocate(SIZE);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            if (rank == 0) {
                // Rank 0 sends
                segment.fill(PATTERN);
                endpoint.send(buffer).get(10, TimeUnit.SECONDS);
                log("Sent %d bytes", SIZE);
            } else {
                // Rank 1 receives
                endpoint.receive(buffer).get(10, TimeUnit.SECONDS);
                log("Received %d bytes", SIZE);

                // Verify received data
                for (int i = 0; i < SIZE; i++) {
                    assertEquals(PATTERN, segment.get(ValueLayout.JAVA_BYTE, i),
                            "Data mismatch at offset " + i);
                }
            }
        }
    }

    @Test
    @DisplayName("ibv_post_recv: Two-sided receive operation")
    void testReceive() throws Exception {
        final int SIZE = 8192;

        MemorySegment segment = arena.allocate(SIZE);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            if (rank == 1) {
                // Rank 1 sends data
                for (int i = 0; i < SIZE / 4; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_INT, i, i);
                }
                endpoint.send(buffer).get(10, TimeUnit.SECONDS);
            } else {
                // Rank 0 receives
                long received = endpoint.receive(buffer).get(10, TimeUnit.SECONDS);
                assertEquals(SIZE, received);

                // Verify
                for (int i = 0; i < SIZE / 4; i++) {
                    assertEquals(i, segment.getAtIndex(ValueLayout.JAVA_INT, i));
                }
            }
        }
    }

    // ===== RDMA Write Tests (One-Sided) =====

    @Test
    @DisplayName("IBV_WR_RDMA_WRITE: One-sided write operation")
    void testRdmaWrite() throws Exception {
        final int SIZE = 1024 * 1024; // 1MB
        final byte PATTERN = 0x5A;

        MemorySegment localSegment = arena.allocate(SIZE);
        MemorySegment remoteSegment = arena.allocate(SIZE);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            // Exchange remote buffer info
            long[] remoteInfo = new long[2]; // [address, rkey]
            if (rank == 0) {
                // Rank 0 will write to Rank 1's buffer
                // First exchange keys (simplified - in real impl use out-of-band)
                remoteInfo[0] = remoteBuffer.address();
                remoteInfo[1] = remoteBuffer.remoteKey();
            }

            sync(); // Exchange complete

            if (rank == 0) {
                // Fill local buffer with pattern
                localSegment.fill(PATTERN);

                // RDMA write to remote
                endpoint.write(localBuffer, remoteInfo[0], remoteInfo[1])
                        .get(10, TimeUnit.SECONDS);
                log("RDMA write completed: %d bytes", SIZE);
            }

            sync(); // Wait for write to complete

            if (rank == 1) {
                // Verify data was written
                for (int i = 0; i < SIZE; i++) {
                    assertEquals(PATTERN, remoteSegment.get(ValueLayout.JAVA_BYTE, i),
                            "RDMA write data mismatch at offset " + i);
                }
                log("RDMA write verified: %d bytes", SIZE);
            }
        }
    }

    @Test
    @DisplayName("IBV_WR_RDMA_WRITE_WITH_IMM: Write with immediate data")
    void testRdmaWriteImmediate() throws Exception {
        final int SIZE = 4096;
        final int IMMEDIATE = 0xDEADBEEF;

        MemorySegment localSegment = arena.allocate(SIZE);
        MemorySegment remoteSegment = arena.allocate(SIZE);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            long remoteAddr = remoteBuffer.address();
            long remoteKey = remoteBuffer.remoteKey();

            sync();

            if (rank == 0) {
                localSegment.fill((byte) 0xAB);
                endpoint.writeImmediate(localBuffer, remoteAddr, remoteKey, IMMEDIATE)
                        .get(10, TimeUnit.SECONDS);
                log("RDMA write with immediate completed");
            }

            sync();
        }
    }

    // ===== RDMA Read Tests (One-Sided) =====

    @Test
    @DisplayName("IBV_WR_RDMA_READ: One-sided read operation")
    void testRdmaRead() throws Exception {
        final int SIZE = 1024 * 1024; // 1MB

        MemorySegment localSegment = arena.allocate(SIZE);
        MemorySegment remoteSegment = arena.allocate(SIZE);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            // Rank 1 fills its buffer, Rank 0 reads it
            if (rank == 1) {
                for (int i = 0; i < SIZE / 8; i++) {
                    remoteSegment.setAtIndex(ValueLayout.JAVA_LONG, i, i * 1000L);
                }
            }

            long remoteAddr = remoteBuffer.address();
            long remoteKey = remoteBuffer.remoteKey();

            sync(); // Ensure data is ready

            if (rank == 0) {
                // RDMA read from remote
                endpoint.read(localBuffer, remoteAddr, remoteKey)
                        .get(10, TimeUnit.SECONDS);
                log("RDMA read completed: %d bytes", SIZE);

                // Verify
                for (int i = 0; i < SIZE / 8; i++) {
                    assertEquals(i * 1000L, localSegment.getAtIndex(ValueLayout.JAVA_LONG, i),
                            "RDMA read data mismatch at index " + i);
                }
            }

            sync();
        }
    }

    // ===== Atomic Operations Tests =====

    @Test
    @DisplayName("IBV_WR_ATOMIC_CMP_AND_SWP: Atomic compare-and-swap")
    void testAtomicCompareSwap() throws Exception {
        MemorySegment localSegment = arena.allocate(8);
        MemorySegment remoteSegment = arena.allocate(8);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            // Initialize remote value
            remoteSegment.set(ValueLayout.JAVA_LONG, 0, 100L);

            long remoteAddr = remoteBuffer.address();
            long remoteKey = remoteBuffer.remoteKey();

            sync();

            if (rank == 0) {
                // CAS: if value == 100, set to 200
                endpoint.atomicCompareSwap(localBuffer, remoteAddr, remoteKey, 100L, 200L)
                        .get(10, TimeUnit.SECONDS);

                // Local buffer should contain old value (100)
                long oldValue = localSegment.get(ValueLayout.JAVA_LONG, 0);
                assertEquals(100L, oldValue, "CAS should return old value");
                log("Atomic CAS completed: old=%d", oldValue);
            }

            sync();

            if (rank == 1) {
                // Verify value was swapped
                long newValue = remoteSegment.get(ValueLayout.JAVA_LONG, 0);
                assertEquals(200L, newValue, "CAS should update remote value");
            }
        }
    }

    @Test
    @DisplayName("IBV_WR_ATOMIC_FETCH_AND_ADD: Atomic fetch-and-add")
    void testAtomicFetchAdd() throws Exception {
        MemorySegment localSegment = arena.allocate(8);
        MemorySegment remoteSegment = arena.allocate(8);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            // Initialize remote value
            remoteSegment.set(ValueLayout.JAVA_LONG, 0, 1000L);

            long remoteAddr = remoteBuffer.address();
            long remoteKey = remoteBuffer.remoteKey();

            sync();

            if (rank == 0) {
                // Fetch-and-add: add 42 to remote value
                endpoint.atomicFetchAdd(localBuffer, remoteAddr, remoteKey, 42L)
                        .get(10, TimeUnit.SECONDS);

                // Local buffer should contain old value (1000)
                long oldValue = localSegment.get(ValueLayout.JAVA_LONG, 0);
                assertEquals(1000L, oldValue, "FAA should return old value");
                log("Atomic FAA completed: old=%d", oldValue);
            }

            sync();

            if (rank == 1) {
                // Verify value was incremented
                long newValue = remoteSegment.get(ValueLayout.JAVA_LONG, 0);
                assertEquals(1042L, newValue, "FAA should increment remote value");
            }
        }
    }

    // ===== Large Transfer Tests =====

    @Test
    @DisplayName("Large RDMA write: 256MB transfer")
    void testLargeRdmaWrite() throws Exception {
        final int SIZE = 256 * 1024 * 1024; // 256MB

        MemorySegment localSegment = arena.allocate(SIZE);
        MemorySegment remoteSegment = arena.allocate(SIZE);

        try (RdmaBuffer localBuffer = rdma.registerMemory(localSegment);
             RdmaBuffer remoteBuffer = rdma.registerMemory(remoteSegment)) {

            long remoteAddr = remoteBuffer.address();
            long remoteKey = remoteBuffer.remoteKey();

            sync();

            if (rank == 0) {
                // Fill with pattern
                for (int i = 0; i < SIZE / 8; i++) {
                    localSegment.setAtIndex(ValueLayout.JAVA_LONG, i, i);
                }

                long start = System.nanoTime();
                endpoint.write(localBuffer, remoteAddr, remoteKey)
                        .get(60, TimeUnit.SECONDS);
                long elapsed = System.nanoTime() - start;

                double gbps = (SIZE * 8.0) / elapsed;
                log("Large RDMA write: %d MB in %.2f ms (%.2f Gbps)",
                        SIZE / (1024 * 1024), elapsed / 1e6, gbps);
            }

            sync();
        }
    }
}
