package io.surfworks.warpforge.io.collective;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for CollectiveConfig record.
 */
@Tag("unit")
@DisplayName("CollectiveConfig Unit Tests")
class CollectiveConfigTest {

    @Test
    @DisplayName("of(worldSize, rank) should create config with defaults")
    void testOfBasic() {
        CollectiveConfig config = CollectiveConfig.of(4, 2);

        assertEquals(4, config.worldSize());
        assertEquals(2, config.rank());
        assertEquals("localhost", config.masterAddress());
        assertEquals(29500, config.masterPort());
        assertTrue(config.useRdma());
        assertTrue(config.inPlace());
        assertTrue(config.asyncProgress());
        assertEquals(1, config.numProgressThreads());
    }

    @Test
    @DisplayName("of(worldSize, rank, address, port) should create config with master info")
    void testOfWithMaster() {
        CollectiveConfig config = CollectiveConfig.of(8, 3, "192.168.1.100", 12345);

        assertEquals(8, config.worldSize());
        assertEquals(3, config.rank());
        assertEquals("192.168.1.100", config.masterAddress());
        assertEquals(12345, config.masterPort());
        assertTrue(config.useRdma());
        assertTrue(config.inPlace());
    }

    @Test
    @DisplayName("withRdma should create new config with RDMA setting")
    void testWithRdma() {
        CollectiveConfig config = CollectiveConfig.of(4, 0).withRdma(false);

        assertFalse(config.useRdma());
        // Other values should remain unchanged
        assertEquals(4, config.worldSize());
        assertEquals(0, config.rank());
        assertTrue(config.inPlace());
    }

    @Test
    @DisplayName("withInPlace should create new config with in-place setting")
    void testWithInPlace() {
        CollectiveConfig config = CollectiveConfig.of(4, 0).withInPlace(false);

        assertFalse(config.inPlace());
        // Other values should remain unchanged
        assertEquals(4, config.worldSize());
        assertTrue(config.useRdma());
    }

    @Test
    @DisplayName("withAsyncProgress should create new config with async progress settings")
    void testWithAsyncProgress() {
        CollectiveConfig config = CollectiveConfig.of(4, 0).withAsyncProgress(true, 4);

        assertTrue(config.asyncProgress());
        assertEquals(4, config.numProgressThreads());
        // Other values should remain unchanged
        assertEquals(4, config.worldSize());
    }

    @Test
    @DisplayName("Builder should create config with custom values")
    void testBuilder() {
        CollectiveConfig config = CollectiveConfig.builder(16, 5)
                .masterAddress("10.0.0.1")
                .masterPort(30000)
                .useRdma(false)
                .inPlace(false)
                .asyncProgress(false)
                .numProgressThreads(2)
                .build();

        assertEquals(16, config.worldSize());
        assertEquals(5, config.rank());
        assertEquals("10.0.0.1", config.masterAddress());
        assertEquals(30000, config.masterPort());
        assertFalse(config.useRdma());
        assertFalse(config.inPlace());
        assertFalse(config.asyncProgress());
        assertEquals(2, config.numProgressThreads());
    }

    @Test
    @DisplayName("Builder should use defaults for unset values")
    void testBuilderDefaults() {
        CollectiveConfig config = CollectiveConfig.builder(4, 1)
                .masterAddress("192.168.1.1")
                .build();

        assertEquals(4, config.worldSize());
        assertEquals(1, config.rank());
        assertEquals("192.168.1.1", config.masterAddress());
        assertEquals(29500, config.masterPort()); // default
        assertTrue(config.useRdma()); // default
        assertTrue(config.inPlace()); // default
        assertTrue(config.asyncProgress()); // default
        assertEquals(1, config.numProgressThreads()); // default
    }

    @Test
    @DisplayName("Record should support equality")
    void testEquality() {
        CollectiveConfig config1 = CollectiveConfig.of(4, 2, "localhost", 29500);
        CollectiveConfig config2 = CollectiveConfig.of(4, 2, "localhost", 29500);

        assertEquals(config1, config2);
        assertEquals(config1.hashCode(), config2.hashCode());
    }

    @Test
    @DisplayName("Record should have meaningful toString")
    void testToString() {
        CollectiveConfig config = CollectiveConfig.of(8, 3, "192.168.1.100", 12345);
        String str = config.toString();

        assertNotNull(str);
        assertTrue(str.contains("8") || str.contains("worldSize"));
        assertTrue(str.contains("3") || str.contains("rank"));
        assertTrue(str.contains("CollectiveConfig"));
    }

    @Test
    @DisplayName("Should support rank 0")
    void testRankZero() {
        CollectiveConfig config = CollectiveConfig.of(4, 0);

        assertEquals(0, config.rank());
        assertTrue(config.rank() >= 0);
        assertTrue(config.rank() < config.worldSize());
    }

    @Test
    @DisplayName("Should support single node configuration")
    void testSingleNode() {
        CollectiveConfig config = CollectiveConfig.of(1, 0);

        assertEquals(1, config.worldSize());
        assertEquals(0, config.rank());
    }

    @Test
    @DisplayName("Should support large world size")
    void testLargeWorldSize() {
        CollectiveConfig config = CollectiveConfig.of(1024, 512);

        assertEquals(1024, config.worldSize());
        assertEquals(512, config.rank());
    }
}
