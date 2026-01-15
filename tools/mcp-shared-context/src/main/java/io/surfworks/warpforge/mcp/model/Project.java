package io.surfworks.warpforge.mcp.model;

import java.util.List;

/**
 * A project for tracking work across Claude sessions.
 */
public record Project(
    String id,
    String name,
    String description,
    List<Task> tasks,
    String createdAt,
    String updatedAt
) {
    public Project withTasks(List<Task> newTasks) {
        return new Project(id, name, description, newTasks, createdAt, java.time.Instant.now().toString());
    }
}
