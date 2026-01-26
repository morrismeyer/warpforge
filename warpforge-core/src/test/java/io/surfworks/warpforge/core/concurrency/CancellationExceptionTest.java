package io.surfworks.warpforge.core.concurrency;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link CancellationException}.
 */
@DisplayName("CancellationException")
class CancellationExceptionTest {

    @Test
    @DisplayName("creates with message")
    void createsWithMessage() {
        CancellationException ex = new CancellationException("operation cancelled");

        assertEquals("operation cancelled", ex.getMessage());
        assertNull(ex.getCause());
    }

    @Test
    @DisplayName("creates with message and cause")
    void createsWithMessageAndCause() {
        Throwable cause = new InterruptedException("thread interrupted");
        CancellationException ex = new CancellationException("operation cancelled", cause);

        assertEquals("operation cancelled", ex.getMessage());
        assertNotNull(ex.getCause());
        assertEquals(cause, ex.getCause());
    }

    @Test
    @DisplayName("is a RuntimeException")
    void isRuntimeException() {
        CancellationException ex = new CancellationException("test");

        assertTrue(ex instanceof RuntimeException);
    }

    @Test
    @DisplayName("can be thrown and caught")
    void canBeThrownAndCaught() {
        boolean caught = false;

        try {
            throw new CancellationException("test throw");
        } catch (CancellationException e) {
            caught = true;
            assertEquals("test throw", e.getMessage());
        }

        assertTrue(caught);
    }

    @Test
    @DisplayName("preserves stack trace")
    void preservesStackTrace() {
        CancellationException ex = new CancellationException("test");

        assertNotNull(ex.getStackTrace());
        assertTrue(ex.getStackTrace().length > 0);
        assertTrue(ex.getStackTrace()[0].getClassName().contains("CancellationExceptionTest"));
    }
}
