package io.surfworks.warpforge.io.rdma;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for RdmaConfig record.
 */
@Tag("unit")
@DisplayName("RdmaConfig Unit Tests")
class RdmaConfigTest {

    @Test
    @DisplayName("DEFAULT config should have expected values")
    void testDefaultConfig() {
        RdmaConfig config = RdmaConfig.DEFAULT;

        assertNull(config.deviceName(), "Default should auto-detect device");
        assertEquals(1, config.port());
        assertEquals(0, config.gidIndex());
        assertEquals(1024, config.maxSendWorkRequests());
        assertEquals(1024, config.maxRecvWorkRequests());
        assertEquals(4, config.maxSendSge());
        assertEquals(4, config.maxRecvSge());
        assertEquals(256, config.maxInlineData());
        assertEquals(0, config.serviceLevel());
        assertEquals(0, config.trafficClass());
        assertEquals(7, config.retryCount());
        assertEquals(7, config.rnrRetryCount());
        assertEquals(12, config.minRnrTimer());
        assertEquals(14, config.timeout());
    }

    @Test
    @DisplayName("LOW_LATENCY config should prioritize latency")
    void testLowLatencyConfig() {
        RdmaConfig config = RdmaConfig.LOW_LATENCY;

        // Smaller queues for lower latency
        assertTrue(config.maxSendWorkRequests() < RdmaConfig.DEFAULT.maxSendWorkRequests());
        assertTrue(config.maxRecvWorkRequests() < RdmaConfig.DEFAULT.maxRecvWorkRequests());
        // More inline data for small messages
        assertTrue(config.maxInlineData() >= RdmaConfig.DEFAULT.maxInlineData());
    }

    @Test
    @DisplayName("HIGH_THROUGHPUT config should prioritize throughput")
    void testHighThroughputConfig() {
        RdmaConfig config = RdmaConfig.HIGH_THROUGHPUT;

        // Larger queues for higher throughput
        assertTrue(config.maxSendWorkRequests() > RdmaConfig.DEFAULT.maxSendWorkRequests());
        assertTrue(config.maxRecvWorkRequests() > RdmaConfig.DEFAULT.maxRecvWorkRequests());
        // More SGE for scatter-gather
        assertTrue(config.maxSendSge() >= RdmaConfig.DEFAULT.maxSendSge());
    }

    @Test
    @DisplayName("withDeviceName should create new config with device name")
    void testWithDeviceName() {
        RdmaConfig config = RdmaConfig.DEFAULT.withDeviceName("mlx5_0");

        assertEquals("mlx5_0", config.deviceName());
        // Other values should remain unchanged
        assertEquals(RdmaConfig.DEFAULT.port(), config.port());
        assertEquals(RdmaConfig.DEFAULT.gidIndex(), config.gidIndex());
    }

    @Test
    @DisplayName("withPort should create new config with port")
    void testWithPort() {
        RdmaConfig config = RdmaConfig.DEFAULT.withPort(2);

        assertEquals(2, config.port());
        // Other values should remain unchanged
        assertNull(config.deviceName());
        assertEquals(RdmaConfig.DEFAULT.gidIndex(), config.gidIndex());
    }

    @Test
    @DisplayName("withGidIndex should create new config with GID index")
    void testWithGidIndex() {
        RdmaConfig config = RdmaConfig.DEFAULT.withGidIndex(3);

        assertEquals(3, config.gidIndex());
        // Other values should remain unchanged
        assertEquals(RdmaConfig.DEFAULT.port(), config.port());
    }

    @Test
    @DisplayName("withQueueDepth should create new config with queue depths")
    void testWithQueueDepth() {
        RdmaConfig config = RdmaConfig.DEFAULT.withQueueDepth(512, 256);

        assertEquals(512, config.maxSendWorkRequests());
        assertEquals(256, config.maxRecvWorkRequests());
        // Other values should remain unchanged
        assertEquals(RdmaConfig.DEFAULT.maxSendSge(), config.maxSendSge());
    }

    @Test
    @DisplayName("Builder should create config with custom values")
    void testBuilder() {
        RdmaConfig config = RdmaConfig.builder()
                .deviceName("mlx5_1")
                .port(2)
                .gidIndex(3)
                .maxSendWorkRequests(2048)
                .maxRecvWorkRequests(2048)
                .maxSendSge(8)
                .maxRecvSge(8)
                .maxInlineData(128)
                .serviceLevel(4)
                .trafficClass(8)
                .retryCount(5)
                .rnrRetryCount(6)
                .minRnrTimer(10)
                .timeout(12)
                .build();

        assertEquals("mlx5_1", config.deviceName());
        assertEquals(2, config.port());
        assertEquals(3, config.gidIndex());
        assertEquals(2048, config.maxSendWorkRequests());
        assertEquals(2048, config.maxRecvWorkRequests());
        assertEquals(8, config.maxSendSge());
        assertEquals(8, config.maxRecvSge());
        assertEquals(128, config.maxInlineData());
        assertEquals(4, config.serviceLevel());
        assertEquals(8, config.trafficClass());
        assertEquals(5, config.retryCount());
        assertEquals(6, config.rnrRetryCount());
        assertEquals(10, config.minRnrTimer());
        assertEquals(12, config.timeout());
    }

    @Test
    @DisplayName("Builder should use defaults for unset values")
    void testBuilderDefaults() {
        RdmaConfig config = RdmaConfig.builder()
                .deviceName("mlx5_0")
                .build();

        assertEquals("mlx5_0", config.deviceName());
        // All other values should be defaults
        assertEquals(1, config.port());
        assertEquals(0, config.gidIndex());
        assertEquals(1024, config.maxSendWorkRequests());
        assertEquals(1024, config.maxRecvWorkRequests());
    }

    @Test
    @DisplayName("Record should support equality")
    void testEquality() {
        RdmaConfig config1 = RdmaConfig.builder()
                .deviceName("mlx5_0")
                .port(1)
                .build();
        RdmaConfig config2 = RdmaConfig.builder()
                .deviceName("mlx5_0")
                .port(1)
                .build();

        assertEquals(config1, config2);
        assertEquals(config1.hashCode(), config2.hashCode());
    }

    @Test
    @DisplayName("Record should have meaningful toString")
    void testToString() {
        RdmaConfig config = RdmaConfig.DEFAULT.withDeviceName("mlx5_0");
        String str = config.toString();

        assertNotNull(str);
        assertTrue(str.contains("mlx5_0"));
        assertTrue(str.contains("RdmaConfig"));
    }
}
