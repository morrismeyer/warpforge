package io.surfworks.warpforge.backend.amd;

import io.surfworks.warpforge.backend.amd.hip.HipContext;
import io.surfworks.warpforge.backend.amd.hip.HipRuntime;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for HIP Event timing functionality.
 *
 * <p>These tests require HIP/ROCm to be available and are tagged with "amd".
 * They verify that the event-based GPU timing works correctly for JFR profiling.
 */
@Tag("amd")
@DisplayName("HIP Event Timing Tests")
class HipEventTimingTest {

    private HipContext ctx;

    @BeforeEach
    void setUp() {
        assumeTrue(HipRuntime.isAvailable(), "HIP/ROCm not available");
        try {
            ctx = HipContext.create(0);
        } catch (HipRuntime.HipException e) {
            // HIP library is available but initialization failed
            // (e.g., no devices, driver not initialized)
            org.junit.jupiter.api.Assumptions.assumeTrue(false,
                "HIP available but initialization failed: " + e.getMessage());
        }
    }

    @AfterEach
    void tearDown() {
        if (ctx != null) {
            ctx.close();
        }
    }

    @Test
    @DisplayName("Event create and destroy")
    void testEventCreateDestroy() {
        long event = ctx.createEvent();
        assertNotEquals(0, event, "Event handle should not be null");
        ctx.destroyEvent(event);
    }

    @Test
    @DisplayName("Event record and synchronize")
    void testEventRecordSync() {
        long event = ctx.createEvent();
        try {
            ctx.recordEvent(event);
            ctx.synchronizeEvent(event);
        } finally {
            ctx.destroyEvent(event);
        }
    }

    @Test
    @DisplayName("Elapsed time between events")
    void testElapsedTime() {
        long start = ctx.createEvent();
        long end = ctx.createEvent();

        try {
            ctx.recordEvent(start);

            // Do some work - allocate and free memory
            long ptr = ctx.allocate(1024 * 1024); // 1MB
            ctx.free(ptr);

            ctx.recordEvent(end);
            ctx.synchronizeEvent(end);

            float elapsedMs = ctx.elapsedTime(start, end);

            // Elapsed time should be non-negative
            assertTrue(elapsedMs >= 0.0f, "Elapsed time should be >= 0");
        } finally {
            ctx.destroyEvent(start);
            ctx.destroyEvent(end);
        }
    }

    @Test
    @DisplayName("timeOperation convenience method")
    void testTimeOperation() {
        float elapsedMs = ctx.timeOperation(() -> {
            // Do some work
            long ptr = ctx.allocate(1024 * 1024);
            ctx.free(ptr);
        });

        assertTrue(elapsedMs >= 0.0f, "Elapsed time should be >= 0");
    }

    @Test
    @DisplayName("HipRuntime event create and destroy")
    void testRuntimeEventCreateDestroy() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            HipRuntime.init();

            long event = HipRuntime.eventCreate(arena);
            assertNotEquals(0, event, "Event handle should not be null");
            HipRuntime.eventDestroy(event);
        }
    }

    @Test
    @DisplayName("HipRuntime elapsed time")
    void testRuntimeElapsedTime() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            HipRuntime.init();

            long start = HipRuntime.eventCreate(arena);
            long end = HipRuntime.eventCreate(arena);

            try {
                HipRuntime.eventRecord(start, 0);
                HipRuntime.synchronize();
                HipRuntime.eventRecord(end, 0);
                HipRuntime.eventSynchronize(end);

                float elapsedMs = HipRuntime.eventElapsedTime(arena, start, end);
                assertTrue(elapsedMs >= 0.0f, "Elapsed time should be >= 0");
            } finally {
                HipRuntime.eventDestroy(start);
                HipRuntime.eventDestroy(end);
            }
        }
    }
}
