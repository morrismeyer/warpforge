package io.surfworks.warpforge.io.collective;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for CollectiveException class.
 */
@Tag("unit")
@DisplayName("CollectiveException Unit Tests")
class CollectiveExceptionTest {

    @Test
    @DisplayName("Should create exception with message only")
    void testMessageConstructor() {
        CollectiveException ex = new CollectiveException("Test error");

        assertEquals("Test error", ex.getMessage());
        assertEquals(CollectiveException.ErrorCode.UNKNOWN, ex.errorCode());
        assertEquals(0, ex.nativeErrorCode());
        assertNull(ex.getCause());
    }

    @Test
    @DisplayName("Should create exception with message and cause")
    void testMessageCauseConstructor() {
        Throwable cause = new RuntimeException("Root cause");
        CollectiveException ex = new CollectiveException("Test error", cause);

        assertEquals("Test error", ex.getMessage());
        assertSame(cause, ex.getCause());
        assertEquals(CollectiveException.ErrorCode.UNKNOWN, ex.errorCode());
    }

    @Test
    @DisplayName("Should create exception with message and error code")
    void testMessageErrorCodeConstructor() {
        CollectiveException ex = new CollectiveException("Shape mismatch",
                CollectiveException.ErrorCode.SHAPE_MISMATCH);

        assertEquals("Shape mismatch", ex.getMessage());
        assertEquals(CollectiveException.ErrorCode.SHAPE_MISMATCH, ex.errorCode());
        assertEquals(0, ex.nativeErrorCode());
    }

    @Test
    @DisplayName("Should create exception with message, error code, and native code")
    void testFullConstructor() {
        CollectiveException ex = new CollectiveException("Native error",
                CollectiveException.ErrorCode.COMMUNICATION_ERROR, 42);

        assertEquals("Native error", ex.getMessage());
        assertEquals(CollectiveException.ErrorCode.COMMUNICATION_ERROR, ex.errorCode());
        assertEquals(42, ex.nativeErrorCode());
    }

    // Factory method tests

    @Test
    @DisplayName("Should create notInitialized exception")
    void testNotInitializedFactory() {
        CollectiveException ex = CollectiveException.notInitialized();

        assertNotNull(ex.getMessage());
        assertTrue(ex.getMessage().toLowerCase().contains("not initialized") ||
                   ex.getMessage().toLowerCase().contains("context"));
        assertEquals(CollectiveException.ErrorCode.NOT_INITIALIZED, ex.errorCode());
    }

    @Test
    @DisplayName("Should create invalidRank exception")
    void testInvalidRankFactory() {
        CollectiveException ex = CollectiveException.invalidRank(5, 4);

        assertNotNull(ex.getMessage());
        assertTrue(ex.getMessage().contains("5"));
        assertTrue(ex.getMessage().contains("4"));
        assertEquals(CollectiveException.ErrorCode.INVALID_RANK, ex.errorCode());
    }

    @Test
    @DisplayName("Should create shapeMismatch exception")
    void testShapeMismatchFactory() {
        CollectiveException ex = CollectiveException.shapeMismatch("[2,3,4]", "[2,3,5]");

        assertNotNull(ex.getMessage());
        assertTrue(ex.getMessage().contains("[2,3,4]"));
        assertTrue(ex.getMessage().contains("[2,3,5]"));
        assertEquals(CollectiveException.ErrorCode.SHAPE_MISMATCH, ex.errorCode());
    }

    @Test
    @DisplayName("Should create timeout exception")
    void testTimeoutFactory() {
        CollectiveException ex = CollectiveException.timeout("allreduce", 5000);

        assertNotNull(ex.getMessage());
        assertTrue(ex.getMessage().contains("allreduce"));
        assertTrue(ex.getMessage().contains("5000"));
        assertEquals(CollectiveException.ErrorCode.TIMEOUT, ex.errorCode());
    }

    // ErrorCode enum tests

    @Test
    @DisplayName("ErrorCode enum should have all expected values")
    void testErrorCodeEnumValues() {
        CollectiveException.ErrorCode[] codes = CollectiveException.ErrorCode.values();
        assertTrue(codes.length >= 12, "Should have at least 12 error codes");

        // Verify key error codes exist
        assertNotNull(CollectiveException.ErrorCode.valueOf("UNKNOWN"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("NOT_INITIALIZED"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("ALREADY_INITIALIZED"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("INVALID_RANK"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("INVALID_WORLD_SIZE"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("SHAPE_MISMATCH"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("DTYPE_MISMATCH"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("TIMEOUT"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("COMMUNICATION_ERROR"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("RESOURCE_EXHAUSTED"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("NOT_SUPPORTED"));
        assertNotNull(CollectiveException.ErrorCode.valueOf("INVALID_STATE"));
    }

    @Test
    @DisplayName("Exception should be throwable")
    void testExceptionIsThrowable() {
        assertThrows(CollectiveException.class, () -> {
            throw new CollectiveException("Test", CollectiveException.ErrorCode.UNKNOWN);
        });
    }

    @Test
    @DisplayName("Exception should preserve stack trace")
    void testStackTracePreserved() {
        CollectiveException ex = new CollectiveException("Test");
        StackTraceElement[] trace = ex.getStackTrace();

        assertNotNull(trace);
        assertTrue(trace.length > 0);
        assertEquals("testStackTracePreserved", trace[0].getMethodName());
    }
}
