package io.surfworks.warpforge.mcp.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.time.Instant;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link SharedState} record.
 */
class SharedStateTest {

    @Test
    @DisplayName("empty() creates empty state with current timestamp")
    void empty_createsEmptyState() {
        var before = Instant.now();
        var state = SharedState.empty();
        var after = Instant.now();

        assertNotNull(state);
        assertTrue(state.projects().isEmpty());
        assertTrue(state.handoffs().isEmpty());
        assertTrue(state.notes().isEmpty());
        assertTrue(state.keyValue().isEmpty());

        // Verify timestamp is recent
        var timestamp = Instant.parse(state.lastUpdated());
        assertFalse(timestamp.isBefore(before));
        assertFalse(timestamp.isAfter(after));
    }

    @Test
    @DisplayName("Record components are accessible")
    void recordComponents_areAccessible() {
        var project = new Project("p1", "Test", "desc", List.of(), "now", "now");
        var handoff = new Handoff("h1", "from", "to", "msg", null, "now", false);

        var state = new SharedState(
            Map.of("p1", project),
            List.of(handoff),
            Map.of("key", "value"),
            Map.of("k", "v"),
            "2024-01-01T00:00:00Z"
        );

        assertEquals(1, state.projects().size());
        assertEquals(project, state.projects().get("p1"));
        assertEquals(1, state.handoffs().size());
        assertEquals(handoff, state.handoffs().get(0));
        assertEquals("value", state.notes().get("key"));
        assertEquals("v", state.keyValue().get("k"));
        assertEquals("2024-01-01T00:00:00Z", state.lastUpdated());
    }

    @Test
    @DisplayName("Empty state projects map is immutable")
    void empty_projectsImmutable() {
        var state = SharedState.empty();

        assertThrows(UnsupportedOperationException.class, () ->
            state.projects().put("id", null)
        );
    }

    @Test
    @DisplayName("Empty state handoffs list is immutable")
    void empty_handoffsImmutable() {
        var state = SharedState.empty();

        assertThrows(UnsupportedOperationException.class, () ->
            state.handoffs().add(null)
        );
    }

    @Test
    @DisplayName("Empty state notes map is immutable")
    void empty_notesImmutable() {
        var state = SharedState.empty();

        assertThrows(UnsupportedOperationException.class, () ->
            state.notes().put("key", "value")
        );
    }

    @Test
    @DisplayName("Empty state keyValue map is immutable")
    void empty_keyValueImmutable() {
        var state = SharedState.empty();

        assertThrows(UnsupportedOperationException.class, () ->
            state.keyValue().put("key", "value")
        );
    }

    @Test
    @DisplayName("State with multiple projects")
    void state_multipleProjects() {
        var p1 = new Project("p1", "Project 1", null, List.of(), "now", "now");
        var p2 = new Project("p2", "Project 2", null, List.of(), "now", "now");
        var p3 = new Project("p3", "Project 3", null, List.of(), "now", "now");

        var state = new SharedState(
            Map.of("p1", p1, "p2", p2, "p3", p3),
            List.of(),
            Map.of(),
            Map.of(),
            "now"
        );

        assertEquals(3, state.projects().size());
        assertEquals("Project 1", state.projects().get("p1").name());
        assertEquals("Project 2", state.projects().get("p2").name());
        assertEquals("Project 3", state.projects().get("p3").name());
    }

    @Test
    @DisplayName("State with multiple handoffs")
    void state_multipleHandoffs() {
        var h1 = new Handoff("h1", "a", "b", "msg1", null, "now", false);
        var h2 = new Handoff("h2", "b", "c", "msg2", null, "now", true);

        var state = new SharedState(
            Map.of(),
            List.of(h1, h2),
            Map.of(),
            Map.of(),
            "now"
        );

        assertEquals(2, state.handoffs().size());
        assertFalse(state.handoffs().get(0).read());
        assertTrue(state.handoffs().get(1).read());
    }

    @Test
    @DisplayName("lastUpdated is included in toString")
    void toString_includesLastUpdated() {
        var state = new SharedState(
            Map.of(),
            List.of(),
            Map.of(),
            Map.of(),
            "2024-01-15T10:30:00Z"
        );

        String str = state.toString();
        assertTrue(str.contains("2024-01-15T10:30:00Z"));
    }
}
