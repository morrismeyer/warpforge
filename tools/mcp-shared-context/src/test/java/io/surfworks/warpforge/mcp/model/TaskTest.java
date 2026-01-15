package io.surfworks.warpforge.mcp.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link Task} record.
 */
class TaskTest {

    @Test
    @DisplayName("Record components are accessible")
    void recordComponents_areAccessible() {
        var task = new Task(
            "task-123",
            "Complete the feature",
            "in_progress",
            "Working on it",
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z"
        );

        assertEquals("task-123", task.id());
        assertEquals("Complete the feature", task.title());
        assertEquals("in_progress", task.status());
        assertEquals("Working on it", task.notes());
        assertEquals("2024-01-01T00:00:00Z", task.createdAt());
        assertEquals("2024-01-02T00:00:00Z", task.updatedAt());
    }

    @Test
    @DisplayName("withStatus creates new task with updated status")
    void withStatus_createsNewTask() {
        var original = new Task(
            "t1", "Task", "pending", "notes",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z"
        );

        var updated = original.withStatus("completed");

        // Original unchanged
        assertEquals("pending", original.status());

        // New task has updated status
        assertEquals("completed", updated.status());

        // Other fields preserved
        assertEquals("t1", updated.id());
        assertEquals("Task", updated.title());
        assertEquals("notes", updated.notes());
        assertEquals("2024-01-01T00:00:00Z", updated.createdAt());

        // Updated timestamp changed
        assertNotEquals(original.updatedAt(), updated.updatedAt());
    }

    @Test
    @DisplayName("withNotes creates new task with updated notes")
    void withNotes_createsNewTask() {
        var original = new Task(
            "t1", "Task", "pending", "old notes",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z"
        );

        var updated = original.withNotes("new notes");

        // Original unchanged
        assertEquals("old notes", original.notes());

        // New task has updated notes
        assertEquals("new notes", updated.notes());

        // Other fields preserved
        assertEquals("t1", updated.id());
        assertEquals("Task", updated.title());
        assertEquals("pending", updated.status());
        assertEquals("2024-01-01T00:00:00Z", updated.createdAt());

        // Updated timestamp changed
        assertNotEquals(original.updatedAt(), updated.updatedAt());
    }

    @Test
    @DisplayName("withStatus updates the updatedAt timestamp")
    void withStatus_updatesTimestamp() throws InterruptedException {
        var original = new Task(
            "t1", "Task", "pending", null,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z"
        );

        Thread.sleep(10);

        var updated = original.withStatus("completed");

        assertNotEquals("2024-01-01T00:00:00Z", updated.updatedAt());
    }

    @Test
    @DisplayName("withNotes updates the updatedAt timestamp")
    void withNotes_updatesTimestamp() throws InterruptedException {
        var original = new Task(
            "t1", "Task", "pending", null,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z"
        );

        Thread.sleep(10);

        var updated = original.withNotes("new notes");

        assertNotEquals("2024-01-01T00:00:00Z", updated.updatedAt());
    }

    @Test
    @DisplayName("Task with null notes")
    void task_nullNotes() {
        var task = new Task("t1", "Task", "pending", null, "now", "now");

        assertNull(task.notes());
    }

    @Test
    @DisplayName("All valid status values")
    void task_allStatusValues() {
        var pending = new Task("t1", "T", "pending", null, "now", "now");
        var inProgress = new Task("t2", "T", "in_progress", null, "now", "now");
        var completed = new Task("t3", "T", "completed", null, "now", "now");
        var blocked = new Task("t4", "T", "blocked", null, "now", "now");

        assertEquals("pending", pending.status());
        assertEquals("in_progress", inProgress.status());
        assertEquals("completed", completed.status());
        assertEquals("blocked", blocked.status());
    }

    @Test
    @DisplayName("Chain withStatus and withNotes")
    void chain_withStatusAndNotes() {
        var original = new Task("t1", "Task", "pending", null, "now", "now");

        var updated = original
            .withStatus("in_progress")
            .withNotes("Started working on it");

        assertEquals("in_progress", updated.status());
        assertEquals("Started working on it", updated.notes());
    }

    @Test
    @DisplayName("Two tasks with same data are equal")
    void equals_sameData() {
        var t1 = new Task("id", "title", "status", "notes", "created", "updated");
        var t2 = new Task("id", "title", "status", "notes", "created", "updated");

        assertEquals(t1, t2);
        assertEquals(t1.hashCode(), t2.hashCode());
    }

    @Test
    @DisplayName("Two tasks with different IDs are not equal")
    void notEquals_differentIds() {
        var t1 = new Task("id1", "title", "status", "notes", "created", "updated");
        var t2 = new Task("id2", "title", "status", "notes", "created", "updated");

        assertNotEquals(t1, t2);
    }

    @Test
    @DisplayName("toString includes task title")
    void toString_includesTitle() {
        var task = new Task("t1", "Very Important Task", "pending", null, "now", "now");

        String str = task.toString();
        assertTrue(str.contains("Very Important Task"));
    }

    @Test
    @DisplayName("withNotes can clear notes by setting null")
    void withNotes_canClearNotes() {
        var original = new Task("t1", "Task", "pending", "some notes", "now", "now");

        var updated = original.withNotes(null);

        assertNull(updated.notes());
    }
}
