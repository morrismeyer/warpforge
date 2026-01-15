package io.surfworks.warpforge.mcp.model;

import java.util.Map;

/**
 * A handoff message between Claude sessions.
 */
public record Handoff(
    String id,
    String from,      // e.g., "claude-code", "cowork"
    String to,
    String message,
    Map<String, Object> context,
    String createdAt,
    boolean read
) {
    public Handoff markAsRead() {
        return new Handoff(id, from, to, message, context, createdAt, true);
    }
}
