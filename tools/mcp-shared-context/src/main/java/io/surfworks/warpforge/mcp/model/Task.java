package io.surfworks.warpforge.mcp.model;

/**
 * A task within a project.
 */
public record Task(
    String id,
    String title,
    String status,  // "pending", "in_progress", "completed", "blocked"
    String notes,
    String createdAt,
    String updatedAt
) {
    public Task withStatus(String newStatus) {
        return new Task(id, title, newStatus, notes, createdAt, java.time.Instant.now().toString());
    }

    public Task withNotes(String newNotes) {
        return new Task(id, title, status, newNotes, createdAt, java.time.Instant.now().toString());
    }
}
