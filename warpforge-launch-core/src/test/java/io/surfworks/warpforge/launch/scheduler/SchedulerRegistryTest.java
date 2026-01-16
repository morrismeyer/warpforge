package io.surfworks.warpforge.launch.scheduler;

import io.surfworks.warpforge.launch.testing.MockScheduler;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for SchedulerRegistry.
 */
class SchedulerRegistryTest {

    @BeforeEach
    void setUp() {
        SchedulerRegistry.clear();
    }

    @AfterEach
    void tearDown() {
        SchedulerRegistry.clear();
    }

    @Test
    void registerAndGet() {
        SchedulerRegistry.register("test", MockScheduler::new);

        assertTrue(SchedulerRegistry.isRegistered("test"));
        Scheduler scheduler = SchedulerRegistry.get("test");
        assertNotNull(scheduler);
        assertEquals("mock", scheduler.name());
    }

    @Test
    void getUnregisteredThrows() {
        assertThrows(IllegalArgumentException.class, () ->
                SchedulerRegistry.get("nonexistent"));
    }

    @Test
    void setAndGetDefault() {
        SchedulerRegistry.register("mock", MockScheduler::new);
        SchedulerRegistry.setDefault("mock");

        assertEquals("mock", SchedulerRegistry.getDefaultName());
        Scheduler defaultScheduler = SchedulerRegistry.getDefault();
        assertEquals("mock", defaultScheduler.name());
    }

    @Test
    void setDefaultWithUnregisteredThrows() {
        assertThrows(IllegalArgumentException.class, () ->
                SchedulerRegistry.setDefault("missing"));
    }

    @Test
    void getDefaultWithNoRegistrationsThrows() {
        assertThrows(IllegalStateException.class, () ->
                SchedulerRegistry.getDefault());
    }

    @Test
    void availableReturnsRegisteredNames() {
        SchedulerRegistry.register("a", MockScheduler::new);
        SchedulerRegistry.register("b", MockScheduler::new);
        SchedulerRegistry.register("c", MockScheduler::new);

        List<String> available = SchedulerRegistry.available();
        assertEquals(3, available.size());
        assertTrue(available.contains("a"));
        assertTrue(available.contains("b"));
        assertTrue(available.contains("c"));
    }

    @Test
    void clearRemovesAllRegistrations() {
        SchedulerRegistry.register("test", MockScheduler::new);
        assertTrue(SchedulerRegistry.isRegistered("test"));

        SchedulerRegistry.clear();
        assertFalse(SchedulerRegistry.isRegistered("test"));
    }

    @Test
    void caseInsensitiveLookup() {
        SchedulerRegistry.register("MyScheduler", MockScheduler::new);

        assertTrue(SchedulerRegistry.isRegistered("MYSCHEDULER"));
        assertTrue(SchedulerRegistry.isRegistered("myscheduler"));
        assertTrue(SchedulerRegistry.isRegistered("MyScheduler"));
    }

    @Test
    void unregister() {
        SchedulerRegistry.register("test", MockScheduler::new);
        assertTrue(SchedulerRegistry.isRegistered("test"));

        SchedulerRegistry.unregister("test");
        assertFalse(SchedulerRegistry.isRegistered("test"));
    }
}
