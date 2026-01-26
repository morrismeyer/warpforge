package io.surfworks.warpforge.backend.nvidia.cuda;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for CUDA Event timing functionality.
 *
 * <p>These tests require CUDA to be available and are tagged with "nvidia".
 * They verify that the event-based GPU timing works correctly for JFR profiling.
 */
@Tag("nvidia")
@DisplayName("CUDA Event Timing Tests")
class CudaEventTimingTest {

    private CudaContext ctx;

    @BeforeEach
    void setUp() {
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        ctx = CudaContext.create(0);
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
    @DisplayName("CudaRuntime event flags")
    void testEventFlags() {
        // Verify event flag constants are defined
        assertEquals(0, CudaRuntime.CU_EVENT_DEFAULT);
        assertEquals(1, CudaRuntime.CU_EVENT_BLOCKING_SYNC);
        assertEquals(2, CudaRuntime.CU_EVENT_DISABLE_TIMING);
    }

    @Test
    @DisplayName("CudaRuntime.timeOperation with throwing runnable")
    void testRuntimeTimeOperation() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            CudaRuntime.init();

            float elapsedMs = CudaRuntime.timeOperation(arena, () -> {
                // Minimal operation - just synchronize
                CudaRuntime.synchronize();
            });

            assertTrue(elapsedMs >= 0.0f, "Elapsed time should be >= 0");
        }
    }
}
