package io.surfworks.warpforge.mcp.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link Project} record.
 */
class ProjectTest {

    @Test
    @DisplayName("Record components are accessible")
    void recordComponents_areAccessible() {
        var tasks = List.of(
            new Task("t1", "Task 1", "pending", null, "now", "now")
        );

        var project = new Project(
            "proj-123",
            "My Project",
            "A test project",
            tasks,
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z"
        );

        assertEquals("proj-123", project.id());
        assertEquals("My Project", project.name());
        assertEquals("A test project", project.description());
        assertEquals(1, project.tasks().size());
        assertEquals("2024-01-01T00:00:00Z", project.createdAt());
        assertEquals("2024-01-02T00:00:00Z", project.updatedAt());
    }

    @Test
    @DisplayName("withTasks creates new project with updated tasks")
    void withTasks_createsNewProject() {
        var original = new Project(
            "proj-123",
            "My Project",
            "Description",
            List.of(),
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z"
        );

        var newTasks = List.of(
            new Task("t1", "New Task", "pending", null, "now", "now")
        );

        var updated = original.withTasks(newTasks);

        // Original unchanged
        assertEquals(0, original.tasks().size());

        // New project has tasks
        assertEquals(1, updated.tasks().size());
        assertEquals("New Task", updated.tasks().get(0).title());

        // Other fields preserved
        assertEquals("proj-123", updated.id());
        assertEquals("My Project", updated.name());
        assertEquals("Description", updated.description());
        assertEquals("2024-01-01T00:00:00Z", updated.createdAt());

        // Updated timestamp changed
        assertNotEquals(original.updatedAt(), updated.updatedAt());
    }

    @Test
    @DisplayName("withTasks updates the updatedAt timestamp")
    void withTasks_updatesTimestamp() throws InterruptedException {
        var original = new Project(
            "p1", "Test", null, List.of(),
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z"
        );

        // Small delay to ensure different timestamp
        Thread.sleep(10);

        var updated = original.withTasks(List.of());

        assertNotEquals("2024-01-01T00:00:00Z", updated.updatedAt());
    }

    @Test
    @DisplayName("Project with null description")
    void project_nullDescription() {
        var project = new Project(
            "p1", "Test", null, List.of(), "now", "now"
        );

        assertNull(project.description());
    }

    @Test
    @DisplayName("Project with empty tasks list")
    void project_emptyTasks() {
        var project = new Project(
            "p1", "Test", "desc", List.of(), "now", "now"
        );

        assertTrue(project.tasks().isEmpty());
    }

    @Test
    @DisplayName("Project with multiple tasks")
    void project_multipleTasks() {
        var tasks = List.of(
            new Task("t1", "Task 1", "pending", null, "now", "now"),
            new Task("t2", "Task 2", "in_progress", "notes", "now", "now"),
            new Task("t3", "Task 3", "completed", null, "now", "now")
        );

        var project = new Project("p1", "Test", null, tasks, "now", "now");

        assertEquals(3, project.tasks().size());
        assertEquals("Task 1", project.tasks().get(0).title());
        assertEquals("Task 2", project.tasks().get(1).title());
        assertEquals("Task 3", project.tasks().get(2).title());
    }

    @Test
    @DisplayName("withTasks preserves task order")
    void withTasks_preservesOrder() {
        var original = new Project("p1", "Test", null, List.of(), "now", "now");

        var tasks = List.of(
            new Task("t1", "First", "pending", null, "now", "now"),
            new Task("t2", "Second", "pending", null, "now", "now"),
            new Task("t3", "Third", "pending", null, "now", "now")
        );

        var updated = original.withTasks(tasks);

        assertEquals("First", updated.tasks().get(0).title());
        assertEquals("Second", updated.tasks().get(1).title());
        assertEquals("Third", updated.tasks().get(2).title());
    }

    @Test
    @DisplayName("Two projects with same data are equal")
    void equals_sameData() {
        var p1 = new Project("id", "name", "desc", List.of(), "created", "updated");
        var p2 = new Project("id", "name", "desc", List.of(), "created", "updated");

        assertEquals(p1, p2);
        assertEquals(p1.hashCode(), p2.hashCode());
    }

    @Test
    @DisplayName("Two projects with different IDs are not equal")
    void notEquals_differentIds() {
        var p1 = new Project("id1", "name", "desc", List.of(), "created", "updated");
        var p2 = new Project("id2", "name", "desc", List.of(), "created", "updated");

        assertNotEquals(p1, p2);
    }

    @Test
    @DisplayName("toString includes project name")
    void toString_includesName() {
        var project = new Project("p1", "My Cool Project", null, List.of(), "now", "now");

        String str = project.toString();
        assertTrue(str.contains("My Cool Project"));
    }
}
