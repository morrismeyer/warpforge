package io.surfworks.warpforge.mcp;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import io.surfworks.warpforge.mcp.model.Handoff;
import io.surfworks.warpforge.mcp.model.Project;
import io.surfworks.warpforge.mcp.model.SharedState;
import io.surfworks.warpforge.mcp.model.Task;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for SharedContextServer logic.
 *
 * <p>These tests verify the state management logic independent of the MCP transport layer.
 * We test the data models, state persistence, and business logic.
 */
class SharedContextLogicTest {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    @TempDir
    Path tempDir;

    private Path stateFile;

    @BeforeEach
    void setUp() {
        stateFile = tempDir.resolve("shared-context.json");
    }

    // ============ State Persistence Tests ============

    @Test
    @DisplayName("Empty state serializes and deserializes correctly")
    void emptyState_roundTrip() throws IOException {
        var state = SharedState.empty();

        Files.writeString(stateFile, GSON.toJson(state));
        var loaded = GSON.fromJson(Files.readString(stateFile), SharedState.class);

        assertNotNull(loaded);
        assertTrue(loaded.projects().isEmpty());
        assertTrue(loaded.handoffs().isEmpty());
        assertTrue(loaded.notes().isEmpty());
        assertTrue(loaded.keyValue().isEmpty());
    }

    @Test
    @DisplayName("State with project persists correctly")
    void stateWithProject_roundTrip() throws IOException {
        var project = new Project(
            "proj-1",
            "Test Project",
            "A test description",
            List.of(
                new Task("task-1", "First Task", "pending", null, "now", "now"),
                new Task("task-2", "Second Task", "completed", "Done!", "now", "now")
            ),
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z"
        );

        var state = new SharedState(
            Map.of("proj-1", project),
            List.of(),
            Map.of(),
            Map.of(),
            "2024-01-02T00:00:00Z"
        );

        Files.writeString(stateFile, GSON.toJson(state));
        var loaded = GSON.fromJson(Files.readString(stateFile), SharedState.class);

        assertEquals(1, loaded.projects().size());
        var loadedProject = loaded.projects().get("proj-1");
        assertEquals("Test Project", loadedProject.name());
        assertEquals("A test description", loadedProject.description());
        assertEquals(2, loadedProject.tasks().size());
        assertEquals("First Task", loadedProject.tasks().get(0).title());
        assertEquals("completed", loadedProject.tasks().get(1).status());
    }

    @Test
    @DisplayName("State with handoffs persists correctly")
    void stateWithHandoffs_roundTrip() throws IOException {
        var handoffs = List.of(
            new Handoff("h1", "claude-code", "cowork", "Please continue", null, "now", false),
            new Handoff("h2", "cowork", "claude-code", "Done!", Map.of("result", "success"), "now", true)
        );

        var state = new SharedState(
            Map.of(),
            handoffs,
            Map.of(),
            Map.of(),
            "now"
        );

        Files.writeString(stateFile, GSON.toJson(state));
        var loaded = GSON.fromJson(Files.readString(stateFile), SharedState.class);

        assertEquals(2, loaded.handoffs().size());
        assertFalse(loaded.handoffs().get(0).read());
        assertTrue(loaded.handoffs().get(1).read());
        assertEquals("success", loaded.handoffs().get(1).context().get("result"));
    }

    @Test
    @DisplayName("State with notes persists correctly")
    void stateWithNotes_roundTrip() throws IOException {
        var state = new SharedState(
            Map.of(),
            List.of(),
            Map.of(
                "todo", "Complete the feature",
                "architecture", "Use microservices"
            ),
            Map.of(),
            "now"
        );

        Files.writeString(stateFile, GSON.toJson(state));
        var loaded = GSON.fromJson(Files.readString(stateFile), SharedState.class);

        assertEquals(2, loaded.notes().size());
        assertEquals("Complete the feature", loaded.notes().get("todo"));
        assertEquals("Use microservices", loaded.notes().get("architecture"));
    }

    @Test
    @DisplayName("State with key-value pairs persists correctly")
    void stateWithKeyValue_roundTrip() throws IOException {
        var state = new SharedState(
            Map.of(),
            List.of(),
            Map.of(),
            Map.of("count", 42.0, "enabled", true, "name", "test"),
            "now"
        );

        Files.writeString(stateFile, GSON.toJson(state));
        var loaded = GSON.fromJson(Files.readString(stateFile), SharedState.class);

        assertEquals(3, loaded.keyValue().size());
        assertEquals(42.0, loaded.keyValue().get("count"));
        assertEquals(true, loaded.keyValue().get("enabled"));
        assertEquals("test", loaded.keyValue().get("name"));
    }

    // ============ Project Logic Tests ============

    @Test
    @DisplayName("Adding task to project creates new task list")
    void addTaskToProject_createsNewList() {
        var project = new Project(
            "p1", "Project", null,
            List.of(new Task("t1", "Existing", "pending", null, "now", "now")),
            "now", "now"
        );

        var newTasks = new ArrayList<>(project.tasks());
        newTasks.add(new Task("t2", "New Task", "pending", null, "now", "now"));

        var updated = project.withTasks(newTasks);

        assertEquals(1, project.tasks().size());
        assertEquals(2, updated.tasks().size());
    }

    @Test
    @DisplayName("Updating task status within project")
    void updateTaskStatus_withinProject() {
        var originalTasks = List.of(
            new Task("t1", "Task 1", "pending", null, "now", "now"),
            new Task("t2", "Task 2", "pending", null, "now", "now")
        );
        var project = new Project("p1", "Project", null, originalTasks, "now", "now");

        // Update task t1 to completed
        var newTasks = new ArrayList<Task>();
        for (var task : project.tasks()) {
            if (task.id().equals("t1")) {
                newTasks.add(task.withStatus("completed"));
            } else {
                newTasks.add(task);
            }
        }

        var updated = project.withTasks(newTasks);

        assertEquals("completed", updated.tasks().get(0).status());
        assertEquals("pending", updated.tasks().get(1).status());
    }

    // ============ Handoff Logic Tests ============

    @Test
    @DisplayName("Filtering handoffs by recipient")
    void filterHandoffs_byRecipient() {
        var handoffs = List.of(
            new Handoff("h1", "a", "b", "msg", null, "now", false),
            new Handoff("h2", "b", "c", "msg", null, "now", false),
            new Handoff("h3", "a", "c", "msg", null, "now", false),
            new Handoff("h4", "c", "b", "msg", null, "now", false)
        );

        var forB = handoffs.stream()
            .filter(h -> h.to().equals("b"))
            .toList();

        var forC = handoffs.stream()
            .filter(h -> h.to().equals("c"))
            .toList();

        assertEquals(2, forB.size());
        assertEquals(2, forC.size());
    }

    @Test
    @DisplayName("Filtering handoffs excludes read by default")
    void filterHandoffs_excludeRead() {
        var handoffs = List.of(
            new Handoff("h1", "a", "b", "msg", null, "now", false),
            new Handoff("h2", "a", "b", "msg", null, "now", true),
            new Handoff("h3", "a", "b", "msg", null, "now", false)
        );

        var unread = handoffs.stream()
            .filter(h -> h.to().equals("b"))
            .filter(h -> !h.read())
            .toList();

        assertEquals(2, unread.size());
    }

    @Test
    @DisplayName("Marking handoff as read")
    void markHandoff_asRead() {
        var handoffs = new ArrayList<>(List.of(
            new Handoff("h1", "a", "b", "msg", null, "now", false),
            new Handoff("h2", "a", "b", "msg", null, "now", false)
        ));

        // Mark h1 as read
        var updated = new ArrayList<Handoff>();
        for (var h : handoffs) {
            if (h.id().equals("h1")) {
                updated.add(h.markAsRead());
            } else {
                updated.add(h);
            }
        }

        assertTrue(updated.get(0).read());
        assertFalse(updated.get(1).read());
    }

    // ============ ID Generation Tests ============

    @Test
    @DisplayName("Generated UUIDs are unique")
    void generatedUuids_areUnique() {
        var ids = new ArrayList<String>();
        for (int i = 0; i < 100; i++) {
            ids.add(java.util.UUID.randomUUID().toString().substring(0, 8));
        }

        var uniqueIds = new java.util.HashSet<>(ids);
        assertEquals(100, uniqueIds.size());
    }

    // ============ Task Completion Counting ============

    @Test
    @DisplayName("Counting completed tasks in project")
    void countCompletedTasks_inProject() {
        var tasks = List.of(
            new Task("t1", "Task 1", "completed", null, "now", "now"),
            new Task("t2", "Task 2", "pending", null, "now", "now"),
            new Task("t3", "Task 3", "completed", null, "now", "now"),
            new Task("t4", "Task 4", "in_progress", null, "now", "now")
        );
        var project = new Project("p1", "Project", null, tasks, "now", "now");

        long completed = project.tasks().stream()
            .filter(t -> "completed".equals(t.status()))
            .count();

        assertEquals(2, completed);
    }

    @Test
    @DisplayName("Counting unread handoffs")
    void countUnreadHandoffs() {
        var handoffs = List.of(
            new Handoff("h1", "a", "b", "msg", null, "now", false),
            new Handoff("h2", "a", "b", "msg", null, "now", true),
            new Handoff("h3", "a", "b", "msg", null, "now", false),
            new Handoff("h4", "a", "b", "msg", null, "now", true)
        );

        long unread = handoffs.stream()
            .filter(h -> !h.read())
            .count();

        assertEquals(2, unread);
    }

    // ============ Complex State Tests ============

    @Test
    @DisplayName("Complex state with all components")
    void complexState_allComponents() throws IOException {
        var project1 = new Project(
            "p1", "Frontend", "React app",
            List.of(
                new Task("t1", "Setup", "completed", null, "now", "now"),
                new Task("t2", "Components", "in_progress", "Working on Header", "now", "now")
            ),
            "now", "now"
        );

        var project2 = new Project(
            "p2", "Backend", "API server",
            List.of(
                new Task("t3", "Database", "pending", null, "now", "now")
            ),
            "now", "now"
        );

        var state = new SharedState(
            Map.of("p1", project1, "p2", project2),
            List.of(
                new Handoff("h1", "code", "desktop", "Review needed", Map.of("pr", 123), "now", false)
            ),
            Map.of("readme", "Project documentation"),
            Map.of("version", "1.0.0"),
            Instant.now().toString()
        );

        Files.writeString(stateFile, GSON.toJson(state));
        var loaded = GSON.fromJson(Files.readString(stateFile), SharedState.class);

        assertEquals(2, loaded.projects().size());
        assertEquals(1, loaded.handoffs().size());
        assertEquals(1, loaded.notes().size());
        assertEquals(1, loaded.keyValue().size());

        assertEquals("Frontend", loaded.projects().get("p1").name());
        assertEquals(2, loaded.projects().get("p1").tasks().size());
        assertEquals(123.0, loaded.handoffs().get(0).context().get("pr"));
    }
}
