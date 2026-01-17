package io.surfworks.warpforge.io.rdma;

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
 * Unit tests for RdmaException class.
 */
@Tag("unit")
@DisplayName("RdmaException Unit Tests")
class RdmaExceptionTest {

    @Test
    @DisplayName("Should create exception with message only")
    void testMessageConstructor() {
        RdmaException ex = new RdmaException("Test error");

        assertEquals("Test error", ex.getMessage());
        assertEquals(RdmaException.ErrorCode.UNKNOWN, ex.errorCode());
        assertEquals(0, ex.nativeErrorCode());
        assertNull(ex.getCause());
    }

    @Test
    @DisplayName("Should create exception with message and cause")
    void testMessageCauseConstructor() {
        Throwable cause = new RuntimeException("Root cause");
        RdmaException ex = new RdmaException("Test error", cause);

        assertEquals("Test error", ex.getMessage());
        assertSame(cause, ex.getCause());
    }

    @Test
    @DisplayName("Should create exception with message and error code")
    void testMessageErrorCodeConstructor() {
        RdmaException ex = new RdmaException("Connection failed",
                RdmaException.ErrorCode.CONNECTION_FAILED);

        assertEquals("Connection failed", ex.getMessage());
        assertEquals(RdmaException.ErrorCode.CONNECTION_FAILED, ex.errorCode());
        assertEquals(0, ex.nativeErrorCode());
    }

    @Test
    @DisplayName("Should create exception with message, error code, and native code")
    void testFullConstructor() {
        RdmaException ex = new RdmaException("Native error",
                RdmaException.ErrorCode.UNKNOWN, 42);

        assertEquals("Native error", ex.getMessage());
        assertEquals(RdmaException.ErrorCode.UNKNOWN, ex.errorCode());
        assertEquals(42, ex.nativeErrorCode());
    }

    @Test
    @DisplayName("Should create exception with error code and cause")
    void testErrorCodeCauseConstructor() {
        Throwable cause = new RuntimeException("Root cause");
        RdmaException ex = new RdmaException("Test error",
                RdmaException.ErrorCode.TIMEOUT, cause);

        assertEquals("Test error", ex.getMessage());
        assertEquals(RdmaException.ErrorCode.TIMEOUT, ex.errorCode());
        assertSame(cause, ex.getCause());
    }

    // Factory method tests

    @Test
    @DisplayName("Should create noDevice exception")
    void testNoDeviceFactory() {
        RdmaException ex = RdmaException.noDevice();

        assertNotNull(ex.getMessage());
        assertTrue(ex.getMessage().toLowerCase().contains("device") ||
                   ex.getMessage().toLowerCase().contains("rdma"));
        assertEquals(RdmaException.ErrorCode.NO_DEVICE, ex.errorCode());
    }

    @Test
    @DisplayName("Should create deviceNotFound exception")
    void testDeviceNotFoundFactory() {
        RdmaException ex = RdmaException.deviceNotFound("mlx5_0");

        assertNotNull(ex.getMessage());
        assertTrue(ex.getMessage().contains("mlx5_0"));
        assertEquals(RdmaException.ErrorCode.DEVICE_NOT_FOUND, ex.errorCode());
    }

    @Test
    @DisplayName("Should create connectionFailed exception")
    void testConnectionFailedFactory() {
        RdmaException ex = RdmaException.connectionFailed("192.168.1.100", 18515, "timeout");

        assertNotNull(ex.getMessage());
        assertTrue(ex.getMessage().contains("192.168.1.100"));
        assertTrue(ex.getMessage().contains("18515") || ex.getMessage().contains("timeout"));
        assertEquals(RdmaException.ErrorCode.CONNECTION_FAILED, ex.errorCode());
    }

    @Test
    @DisplayName("Should create timeout exception")
    void testTimeoutFactory() {
        RdmaException ex = RdmaException.timeout("RDMA write operation");

        assertNotNull(ex.getMessage());
        assertTrue(ex.getMessage().toLowerCase().contains("timeout") ||
                   ex.getMessage().contains("RDMA write"));
        assertEquals(RdmaException.ErrorCode.TIMEOUT, ex.errorCode());
    }

    @Test
    @DisplayName("Should create memoryRegistrationFailed exception")
    void testMemoryRegistrationFailedFactory() {
        RdmaException ex = RdmaException.memoryRegistrationFailed(1024 * 1024, "permission denied");

        assertNotNull(ex.getMessage());
        assertEquals(RdmaException.ErrorCode.MEMORY_REGISTRATION_FAILED, ex.errorCode());
    }

    // ErrorCode enum tests

    @Test
    @DisplayName("ErrorCode enum should have all expected values")
    void testErrorCodeEnumValues() {
        RdmaException.ErrorCode[] codes = RdmaException.ErrorCode.values();
        assertTrue(codes.length >= 10, "Should have at least 10 error codes");

        // Verify key error codes exist
        assertNotNull(RdmaException.ErrorCode.valueOf("UNKNOWN"));
        assertNotNull(RdmaException.ErrorCode.valueOf("NO_DEVICE"));
        assertNotNull(RdmaException.ErrorCode.valueOf("DEVICE_NOT_FOUND"));
        assertNotNull(RdmaException.ErrorCode.valueOf("CONNECTION_FAILED"));
        assertNotNull(RdmaException.ErrorCode.valueOf("TIMEOUT"));
        assertNotNull(RdmaException.ErrorCode.valueOf("MEMORY_REGISTRATION_FAILED"));
        assertNotNull(RdmaException.ErrorCode.valueOf("INVALID_STATE"));
        assertNotNull(RdmaException.ErrorCode.valueOf("NOT_SUPPORTED"));
    }

    @Test
    @DisplayName("Exception should be throwable")
    void testExceptionIsThrowable() {
        assertThrows(RdmaException.class, () -> {
            throw new RdmaException("Test", RdmaException.ErrorCode.UNKNOWN);
        });
    }

    @Test
    @DisplayName("Exception should preserve stack trace")
    void testStackTracePreserved() {
        RdmaException ex = new RdmaException("Test");
        StackTraceElement[] trace = ex.getStackTrace();

        assertNotNull(trace);
        assertTrue(trace.length > 0);
        assertEquals("testStackTracePreserved", trace[0].getMethodName());
    }
}
