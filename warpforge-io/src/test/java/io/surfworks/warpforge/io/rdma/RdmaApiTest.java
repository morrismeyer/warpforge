package io.surfworks.warpforge.io.rdma;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for RdmaApi interface using mock implementation.
 * These tests run on NUC without RDMA hardware.
 */
@Tag("unit")
@DisplayName("RDMA API Unit Tests")
class RdmaApiTest {

    private RdmaApi rdma;
    private Arena arena;

    @BeforeEach
    void setUp() {
        rdma = Rdma.loadMock();
        arena = Arena.ofConfined();
    }

    @AfterEach
    void tearDown() {
        if (rdma != null) rdma.close();
        if (arena != null) arena.close();
    }

    @Test
    @DisplayName("Should return mock backend name")
    void testBackendName() {
        assertEquals("mock", rdma.backendName());
    }

    @Test
    @DisplayName("Should return default configuration")
    void testConfig() {
        RdmaConfig config = rdma.config();
        assertNotNull(config);
        assertEquals(RdmaConfig.DEFAULT.maxSendWorkRequests(), config.maxSendWorkRequests());
    }

    @Test
    @DisplayName("Should list mock devices")
    void testDevices() {
        var devices = rdma.devices();
        assertFalse(devices.isEmpty());
        assertEquals("mock0", devices.get(0).name());
    }

    @Test
    @DisplayName("Should register memory segment")
    void testRegisterMemory() {
        MemorySegment segment = arena.allocate(1024);

        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            assertNotNull(buffer);
            assertEquals(1024, buffer.byteSize());
            assertTrue(buffer.isValid());
            assertEquals(segment, buffer.segment());
        }
    }

    @Test
    @DisplayName("Should provide remote and local keys")
    void testMemoryKeys() {
        MemorySegment segment = arena.allocate(1024);

        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            assertTrue(buffer.remoteKey() > 0);
            assertTrue(buffer.localKey() > 0);
            assertEquals(segment.address(), buffer.address());
        }
    }

    @Test
    @DisplayName("Should invalidate buffer after close")
    void testBufferClose() {
        MemorySegment segment = arena.allocate(1024);
        RdmaBuffer buffer = rdma.registerMemory(segment);

        assertTrue(buffer.isValid());
        buffer.close();
        assertFalse(buffer.isValid());
    }

    @Test
    @DisplayName("Should throw when accessing closed buffer")
    void testClosedBufferThrows() {
        MemorySegment segment = arena.allocate(1024);
        RdmaBuffer buffer = rdma.registerMemory(segment);
        buffer.close();

        assertThrows(RdmaException.class, buffer::segment);
        assertThrows(RdmaException.class, buffer::remoteKey);
    }

    @Test
    @DisplayName("Should connect to remote endpoint")
    void testConnect() {
        try (RdmaEndpoint endpoint = rdma.connect("127.0.0.1", 12345)) {
            assertNotNull(endpoint);
            assertEquals("127.0.0.1:12345", endpoint.remoteAddress());
            assertEquals(RdmaEndpoint.EndpointState.CONNECTED, endpoint.state());
        }
    }

    @Test
    @DisplayName("Should create listener")
    void testListen() {
        try (RdmaListener listener = rdma.listen(12345)) {
            assertNotNull(listener);
            assertEquals(12345, listener.port());
            assertTrue(listener.isActive());
        }
    }

    @Test
    @DisplayName("Should track statistics")
    void testStats() {
        RdmaApi.RdmaStats stats = rdma.stats();
        assertNotNull(stats);
        assertEquals(0, stats.totalBytesSent());
        assertEquals(0, stats.totalOperations());
    }

    @Test
    @DisplayName("Should report capabilities")
    void testCapabilities() {
        assertFalse(rdma.supportsGpuDirect()); // Mock doesn't support GPU Direct
        assertTrue(rdma.supportsAtomics());
        assertTrue(rdma.maxInlineSize() > 0);
        assertTrue(rdma.maxMemoryRegionSize() > 0);
    }

    @Test
    @DisplayName("Should throw after context close")
    void testCloseContext() {
        rdma.close();
        assertThrows(RdmaException.class, () -> rdma.registerMemory(arena.allocate(1024)));
    }
}
