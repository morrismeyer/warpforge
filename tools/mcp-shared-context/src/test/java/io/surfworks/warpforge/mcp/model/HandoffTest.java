package io.surfworks.warpforge.mcp.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link Handoff} record.
 */
class HandoffTest {

    @Test
    @DisplayName("Record components are accessible")
    void recordComponents_areAccessible() {
        var context = Map.<String, Object>of("key", "value", "number", 42);

        var handoff = new Handoff(
            "handoff-123",
            "claude-code",
            "cowork",
            "Please continue working on the feature",
            context,
            "2024-01-01T00:00:00Z",
            false
        );

        assertEquals("handoff-123", handoff.id());
        assertEquals("claude-code", handoff.from());
        assertEquals("cowork", handoff.to());
        assertEquals("Please continue working on the feature", handoff.message());
        assertEquals(context, handoff.context());
        assertEquals("2024-01-01T00:00:00Z", handoff.createdAt());
        assertFalse(handoff.read());
    }

    @Test
    @DisplayName("markAsRead creates new handoff with read=true")
    void markAsRead_createsNewHandoff() {
        var original = new Handoff(
            "h1", "from", "to", "message",
            Map.of("key", "value"),
            "2024-01-01T00:00:00Z",
            false
        );

        var marked = original.markAsRead();

        // Original unchanged
        assertFalse(original.read());

        // New handoff is marked as read
        assertTrue(marked.read());

        // Other fields preserved
        assertEquals("h1", marked.id());
        assertEquals("from", marked.from());
        assertEquals("to", marked.to());
        assertEquals("message", marked.message());
        assertEquals(Map.of("key", "value"), marked.context());
        assertEquals("2024-01-01T00:00:00Z", marked.createdAt());
    }

    @Test
    @DisplayName("markAsRead is idempotent")
    void markAsRead_idempotent() {
        var original = new Handoff(
            "h1", "from", "to", "message", null, "now", true
        );

        var marked = original.markAsRead();

        assertTrue(marked.read());
        assertEquals(original, marked); // Already read, so same
    }

    @Test
    @DisplayName("Handoff with null context")
    void handoff_nullContext() {
        var handoff = new Handoff(
            "h1", "from", "to", "message", null, "now", false
        );

        assertNull(handoff.context());
    }

    @Test
    @DisplayName("Handoff with empty context")
    void handoff_emptyContext() {
        var handoff = new Handoff(
            "h1", "from", "to", "message", Map.of(), "now", false
        );

        assertNotNull(handoff.context());
        assertTrue(handoff.context().isEmpty());
    }

    @Test
    @DisplayName("Handoff with complex context")
    void handoff_complexContext() {
        var context = Map.<String, Object>of(
            "string", "value",
            "number", 42,
            "boolean", true,
            "nested", Map.of("a", "b")
        );

        var handoff = new Handoff(
            "h1", "from", "to", "message", context, "now", false
        );

        assertEquals("value", handoff.context().get("string"));
        assertEquals(42, handoff.context().get("number"));
        assertEquals(true, handoff.context().get("boolean"));
    }

    @Test
    @DisplayName("Two handoffs with same data are equal")
    void equals_sameData() {
        var h1 = new Handoff("id", "from", "to", "msg", null, "created", false);
        var h2 = new Handoff("id", "from", "to", "msg", null, "created", false);

        assertEquals(h1, h2);
        assertEquals(h1.hashCode(), h2.hashCode());
    }

    @Test
    @DisplayName("Two handoffs with different IDs are not equal")
    void notEquals_differentIds() {
        var h1 = new Handoff("id1", "from", "to", "msg", null, "created", false);
        var h2 = new Handoff("id2", "from", "to", "msg", null, "created", false);

        assertNotEquals(h1, h2);
    }

    @Test
    @DisplayName("Two handoffs with different read status are not equal")
    void notEquals_differentReadStatus() {
        var h1 = new Handoff("id", "from", "to", "msg", null, "created", false);
        var h2 = new Handoff("id", "from", "to", "msg", null, "created", true);

        assertNotEquals(h1, h2);
    }

    @Test
    @DisplayName("toString includes from and to")
    void toString_includesFromAndTo() {
        var handoff = new Handoff(
            "h1", "claude-code", "desktop", "message", null, "now", false
        );

        String str = handoff.toString();
        assertTrue(str.contains("claude-code"));
        assertTrue(str.contains("desktop"));
    }

    @Test
    @DisplayName("Handoff session identifiers")
    void handoff_sessionIdentifiers() {
        // Document expected session identifiers
        var fromClaudeCode = new Handoff(
            "h1", "claude-code", "cowork", "msg", null, "now", false
        );
        var fromCowork = new Handoff(
            "h2", "cowork", "claude-code", "msg", null, "now", false
        );
        var fromDesktop = new Handoff(
            "h3", "desktop", "claude-code", "msg", null, "now", false
        );

        assertEquals("claude-code", fromClaudeCode.from());
        assertEquals("cowork", fromCowork.from());
        assertEquals("desktop", fromDesktop.from());
    }

    @Test
    @DisplayName("Long message content")
    void handoff_longMessage() {
        var longMessage = "A".repeat(10000);

        var handoff = new Handoff(
            "h1", "from", "to", longMessage, null, "now", false
        );

        assertEquals(10000, handoff.message().length());
    }

    @Test
    @DisplayName("Message with special characters")
    void handoff_specialCharacters() {
        var message = "Hello\nWorld\t\r\n\"quotes\" and 'apostrophes' & <xml>";

        var handoff = new Handoff(
            "h1", "from", "to", message, null, "now", false
        );

        assertEquals(message, handoff.message());
    }
}
