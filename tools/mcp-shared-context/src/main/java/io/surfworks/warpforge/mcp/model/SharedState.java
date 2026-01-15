package io.surfworks.warpforge.mcp.model;

import java.util.List;
import java.util.Map;

/**
 * The complete shared state persisted to disk.
 */
public record SharedState(
    Map<String, Project> projects,
    List<Handoff> handoffs,
    Map<String, String> notes,
    Map<String, Object> keyValue,
    String lastUpdated
) {
    public static SharedState empty() {
        return new SharedState(
            Map.of(),
            List.of(),
            Map.of(),
            Map.of(),
            java.time.Instant.now().toString()
        );
    }
}
