package io.surfworks.warpforge.io.rdma;

import org.junit.jupiter.api.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for RdmaEndpoint operations using mock implementation.
 * These tests verify all ibverbs operations exposed to Java.
 */
@Tag("unit")
@DisplayName("RDMA Endpoint Unit Tests")
class RdmaEndpointTest {

    private RdmaApi rdma;
    private Arena arena;
    private RdmaEndpoint endpoint;

    @BeforeEach
    void setUp() {
        rdma = Rdma.loadMock();
        arena = Arena.ofConfined();
        endpoint = rdma.connect("127.0.0.1", 12345);
    }

    @AfterEach
    void tearDown() {
        if (endpoint != null) endpoint.close();
        if (rdma != null) rdma.close();
        if (arena != null) arena.close();
    }

    // ===== Send/Receive Tests =====

    @Test
    @DisplayName("Should send data asynchronously")
    void testSend() throws Exception {
        MemorySegment segment = arena.allocate(1024);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            CompletableFuture<Void> future = endpoint.send(buffer);
            assertDoesNotThrow(() -> future.get(5, TimeUnit.SECONDS));
        }
    }

    @Test
    @DisplayName("Should send with offset and length")
    void testSendWithOffsetLength() throws Exception {
        MemorySegment segment = arena.allocate(1024);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            CompletableFuture<Void> future = endpoint.send(buffer, 100, 500);
            assertDoesNotThrow(() -> future.get(5, TimeUnit.SECONDS));
        }
    }

    @Test
    @DisplayName("Should receive data asynchronously")
    void testReceive() throws Exception {
        MemorySegment segment = arena.allocate(1024);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            CompletableFuture<Long> future = endpoint.receive(buffer);
            Long received = future.get(5, TimeUnit.SECONDS);
            assertEquals(1024, received);
        }
    }

    // ===== RDMA Write Tests =====

    @Test
    @DisplayName("Should perform RDMA write")
    void testWrite() throws Exception {
        MemorySegment segment = arena.allocate(1024);
        segment.fill((byte) 0xAB);

        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            CompletableFuture<Void> future = endpoint.write(buffer, 0x1000, 42);
            assertDoesNotThrow(() -> future.get(5, TimeUnit.SECONDS));
        }
    }

    @Test
    @DisplayName("Should perform RDMA write with offset and length")
    void testWriteWithOffsetLength() throws Exception {
        MemorySegment segment = arena.allocate(1024);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            CompletableFuture<Void> future = endpoint.write(buffer, 100, 500, 0x1000, 42);
            assertDoesNotThrow(() -> future.get(5, TimeUnit.SECONDS));
        }
    }

    @Test
    @DisplayName("Should perform RDMA write with immediate")
    void testWriteImmediate() throws Exception {
        MemorySegment segment = arena.allocate(1024);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            CompletableFuture<Void> future = endpoint.writeImmediate(buffer, 0x1000, 42, 0xDEADBEEF);
            assertDoesNotThrow(() -> future.get(5, TimeUnit.SECONDS));
        }
    }

    // ===== RDMA Read Tests =====

    @Test
    @DisplayName("Should perform RDMA read")
    void testRead() throws Exception {
        MemorySegment segment = arena.allocate(1024);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            CompletableFuture<Void> future = endpoint.read(buffer, 0x1000, 42);
            assertDoesNotThrow(() -> future.get(5, TimeUnit.SECONDS));
        }
    }

    @Test
    @DisplayName("Should perform RDMA read with offset and length")
    void testReadWithOffsetLength() throws Exception {
        MemorySegment segment = arena.allocate(1024);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            CompletableFuture<Void> future = endpoint.read(buffer, 100, 500, 0x1000, 42);
            assertDoesNotThrow(() -> future.get(5, TimeUnit.SECONDS));
        }
    }

    // ===== Atomic Operations Tests =====

    @Test
    @DisplayName("Should perform atomic compare-and-swap")
    void testAtomicCompareSwap() throws Exception {
        MemorySegment segment = arena.allocate(8);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            CompletableFuture<Void> future = endpoint.atomicCompareSwap(
                    buffer, 0x1000, 42, 100L, 200L);
            assertDoesNotThrow(() -> future.get(5, TimeUnit.SECONDS));
        }
    }

    @Test
    @DisplayName("Should perform atomic fetch-and-add")
    void testAtomicFetchAdd() throws Exception {
        MemorySegment segment = arena.allocate(8);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            CompletableFuture<Void> future = endpoint.atomicFetchAdd(
                    buffer, 0x1000, 42, 10L);
            assertDoesNotThrow(() -> future.get(5, TimeUnit.SECONDS));
        }
    }

    // ===== Loopback Tests (Mock-specific) =====

    @Test
    @DisplayName("Should copy data in loopback write scenario")
    void testLoopbackWrite() throws Exception {
        // Create source and destination buffers
        MemorySegment srcSegment = arena.allocate(1024);
        MemorySegment dstSegment = arena.allocate(1024);

        // Fill source with test pattern
        for (int i = 0; i < 256; i++) {
            srcSegment.setAtIndex(ValueLayout.JAVA_INT, i, i);
        }

        try (RdmaBuffer srcBuffer = rdma.registerMemory(srcSegment);
             RdmaBuffer dstBuffer = rdma.registerMemory(dstSegment)) {

            // Write from source to destination (using mock loopback)
            endpoint.write(srcBuffer, dstBuffer.address(), dstBuffer.remoteKey())
                    .get(5, TimeUnit.SECONDS);

            // Verify data was copied
            for (int i = 0; i < 256; i++) {
                assertEquals(i, dstSegment.getAtIndex(ValueLayout.JAVA_INT, i));
            }
        }
    }

    @Test
    @DisplayName("Should copy data in loopback read scenario")
    void testLoopbackRead() throws Exception {
        // Create source and destination buffers
        MemorySegment srcSegment = arena.allocate(1024);
        MemorySegment dstSegment = arena.allocate(1024);

        // Fill source with test pattern
        for (int i = 0; i < 256; i++) {
            srcSegment.setAtIndex(ValueLayout.JAVA_INT, i, i * 2);
        }

        try (RdmaBuffer srcBuffer = rdma.registerMemory(srcSegment);
             RdmaBuffer dstBuffer = rdma.registerMemory(dstSegment)) {

            // Read from source into destination (using mock loopback)
            endpoint.read(dstBuffer, srcBuffer.address(), srcBuffer.remoteKey())
                    .get(5, TimeUnit.SECONDS);

            // Verify data was copied
            for (int i = 0; i < 256; i++) {
                assertEquals(i * 2, dstSegment.getAtIndex(ValueLayout.JAVA_INT, i));
            }
        }
    }

    // ===== Control Operations Tests =====

    @Test
    @DisplayName("Should flush pending operations")
    void testFlush() {
        assertDoesNotThrow(() -> endpoint.flush());
    }

    @Test
    @DisplayName("Should await completion with timeout")
    void testAwaitCompletion() {
        assertTrue(endpoint.awaitCompletion(1000));
    }

    @Test
    @DisplayName("Should track endpoint statistics")
    void testStats() throws Exception {
        MemorySegment segment = arena.allocate(1024);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            endpoint.send(buffer).get(5, TimeUnit.SECONDS);
            endpoint.write(buffer, 0, 42).get(5, TimeUnit.SECONDS);

            RdmaEndpoint.EndpointStats stats = endpoint.stats();
            assertEquals(1, stats.sendOperations());
            assertEquals(1, stats.writeOperations());
            assertTrue(stats.bytesSent() > 0);
        }
    }

    // ===== Lifecycle Tests =====

    @Test
    @DisplayName("Should transition to disconnected on close")
    void testClose() {
        assertEquals(RdmaEndpoint.EndpointState.CONNECTED, endpoint.state());
        endpoint.close();
        assertEquals(RdmaEndpoint.EndpointState.DISCONNECTED, endpoint.state());
    }

    @Test
    @DisplayName("Should throw when operating on closed endpoint")
    void testClosedEndpointThrows() {
        MemorySegment segment = arena.allocate(1024);
        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            endpoint.close();
            assertThrows(RdmaException.class, () -> endpoint.send(buffer));
        }
    }
}
