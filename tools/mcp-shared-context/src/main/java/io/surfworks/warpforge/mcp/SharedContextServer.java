package io.surfworks.warpforge.mcp;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import io.modelcontextprotocol.server.McpServer;
import io.modelcontextprotocol.server.McpSyncServerExchange;
import io.modelcontextprotocol.server.transport.StdioServerTransportProvider;
import io.modelcontextprotocol.spec.McpSchema;
import io.surfworks.warpforge.mcp.model.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.function.BiFunction;

/**
 * MCP Shared Context Server
 *
 * <p>Enables context sharing between Claude Code and Claude Desktop/Cowork.
 * Persists state to a JSON file that both applications can access.
 */
public class SharedContextServer {

    private static final Path STATE_FILE = Path.of(
        System.getProperty("user.home"),
        ".config", "claude", "shared-context.json"
    );

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final ObjectMapper JACKSON = new ObjectMapper();

    public static void main(String[] args) {
        var transportProvider = new StdioServerTransportProvider(JACKSON);

        var server = McpServer.sync(transportProvider)
            .serverInfo("warpforge-shared-context", "1.0.0")
            .capabilities(McpSchema.ServerCapabilities.builder()
                .tools(true)
                .build())
            // Project tools
            .tool(tool("create_project", "Create a new project to track work across Claude sessions",
                schema("name", "description").toString()), SharedContextServer::createProject)
            .tool(tool("list_projects", "List all projects", emptySchema()), SharedContextServer::listProjects)
            .tool(tool("get_project", "Get full details of a project including all tasks",
                schemaRequired("projectId").toString()), SharedContextServer::getProject)
            .tool(tool("add_task", "Add a task to a project",
                schemaRequired("projectId", "title").with("notes").toString()), SharedContextServer::addTask)
            .tool(tool("update_task", "Update a task's status or notes",
                schemaRequired("projectId", "taskId").with("status", "notes").toString()), SharedContextServer::updateTask)
            // Handoff tools
            .tool(tool("create_handoff", "Create a handoff message for another Claude session",
                schemaRequired("from", "to", "message").with("context").toString()), SharedContextServer::createHandoff)
            .tool(tool("get_handoffs", "Get pending handoffs for a specific session",
                schemaRequired("for").with("includeRead").toString()), SharedContextServer::getHandoffs)
            .tool(tool("mark_handoff_read", "Mark a handoff as read",
                schemaRequired("handoffId").toString()), SharedContextServer::markHandoffRead)
            // Notes tools
            .tool(tool("set_note", "Save a note that can be accessed across sessions",
                schemaRequired("key", "content").toString()), SharedContextServer::setNote)
            .tool(tool("get_note", "Retrieve a saved note",
                schemaRequired("key").toString()), SharedContextServer::getNote)
            .tool(tool("list_notes", "List all saved notes", emptySchema()), SharedContextServer::listNotes)
            // Key-value tools
            .tool(tool("set_value", "Store a value in the shared key-value store",
                schemaRequired("key", "value").toString()), SharedContextServer::setValue)
            .tool(tool("get_value", "Retrieve a value from the shared key-value store",
                schemaRequired("key").toString()), SharedContextServer::getValue)
            // Utility tools
            .tool(tool("get_state_summary", "Get a summary of all shared state", emptySchema()),
                SharedContextServer::getStateSummary)
            .build();

        // Server keeps running and processing STDIO messages until input stream is closed
    }

    // ============ TOOL IMPLEMENTATIONS ============

    private static McpSchema.CallToolResult createProject(McpSyncServerExchange exchange, Map<String, Object> args) {
        String name = getString(args, "name");
        String description = getString(args, "description");

        var state = loadState();
        String id = generateId();
        String now = Instant.now().toString();

        var projects = new HashMap<>(state.projects());
        projects.put(id, new Project(id, name, description, new ArrayList<>(), now, now));

        saveState(new SharedState(projects, state.handoffs(), state.notes(), state.keyValue(), now));
        return textResult(String.format("Created project \"%s\" with ID: %s", name, id));
    }

    private static McpSchema.CallToolResult listProjects(McpSyncServerExchange exchange, Map<String, Object> args) {
        var state = loadState();
        var projects = state.projects().values();

        if (projects.isEmpty()) {
            return textResult("No projects found.");
        }

        var sb = new StringBuilder("# Projects\n\n");
        for (var p : projects) {
            int taskCount = p.tasks().size();
            long completed = p.tasks().stream().filter(t -> "completed".equals(t.status())).count();
            sb.append(String.format("- **%s** (%s): %d/%d tasks completed", p.name(), p.id(), completed, taskCount));
            if (p.description() != null && !p.description().isBlank()) {
                sb.append("\n  ").append(p.description());
            }
            sb.append("\n");
        }
        return textResult(sb.toString());
    }

    private static McpSchema.CallToolResult getProject(McpSyncServerExchange exchange, Map<String, Object> args) {
        String projectId = getString(args, "projectId");
        var state = loadState();
        var project = state.projects().get(projectId);

        if (project == null) {
            return textResult(String.format("Project %s not found.", projectId));
        }

        var sb = new StringBuilder();
        sb.append("# ").append(project.name()).append("\n\n");
        sb.append("**ID:** ").append(project.id()).append("\n");
        sb.append("**Created:** ").append(project.createdAt()).append("\n");
        sb.append("**Updated:** ").append(project.updatedAt()).append("\n");
        if (project.description() != null) {
            sb.append("\n").append(project.description()).append("\n");
        }
        sb.append("\n## Tasks\n\n");

        if (project.tasks().isEmpty()) {
            sb.append("No tasks yet.");
        } else {
            for (var t : project.tasks()) {
                String icon = switch (t.status()) {
                    case "completed" -> "✅";
                    case "in_progress" -> "🔄";
                    case "blocked" -> "🚫";
                    default -> "⬜";
                };
                sb.append(String.format("%s **%s** (%s)\n", icon, t.title(), t.id()));
                sb.append(String.format("   Status: %s\n", t.status()));
                if (t.notes() != null && !t.notes().isBlank()) {
                    sb.append(String.format("   Notes: %s\n", t.notes()));
                }
            }
        }
        return textResult(sb.toString());
    }

    private static McpSchema.CallToolResult addTask(McpSyncServerExchange exchange, Map<String, Object> args) {
        String projectId = getString(args, "projectId");
        String title = getString(args, "title");
        String notes = getString(args, "notes");

        var state = loadState();
        var project = state.projects().get(projectId);

        if (project == null) {
            return textResult(String.format("Project %s not found.", projectId));
        }

        String now = Instant.now().toString();
        var task = new Task(generateId(), title, "pending", notes, now, now);

        var newTasks = new ArrayList<>(project.tasks());
        newTasks.add(task);

        var projects = new HashMap<>(state.projects());
        projects.put(projectId, project.withTasks(newTasks));

        saveState(new SharedState(projects, state.handoffs(), state.notes(), state.keyValue(), now));
        return textResult(String.format("Added task \"%s\" to project \"%s\"", title, project.name()));
    }

    private static McpSchema.CallToolResult updateTask(McpSyncServerExchange exchange, Map<String, Object> args) {
        String projectId = getString(args, "projectId");
        String taskId = getString(args, "taskId");
        String status = getString(args, "status");
        String notes = getString(args, "notes");

        var state = loadState();
        var project = state.projects().get(projectId);

        if (project == null) {
            return textResult(String.format("Project %s not found.", projectId));
        }

        var newTasks = new ArrayList<Task>();
        Task updatedTask = null;
        for (var t : project.tasks()) {
            if (t.id().equals(taskId)) {
                var updated = t;
                if (status != null && !status.isBlank()) {
                    updated = updated.withStatus(status);
                }
                if (notes != null) {
                    updated = updated.withNotes(notes);
                }
                newTasks.add(updated);
                updatedTask = updated;
            } else {
                newTasks.add(t);
            }
        }

        if (updatedTask == null) {
            return textResult(String.format("Task %s not found in project %s.", taskId, projectId));
        }

        var projects = new HashMap<>(state.projects());
        projects.put(projectId, project.withTasks(newTasks));

        String now = Instant.now().toString();
        saveState(new SharedState(projects, state.handoffs(), state.notes(), state.keyValue(), now));
        return textResult(String.format("Updated task \"%s\" - status: %s", updatedTask.title(), updatedTask.status()));
    }

    private static McpSchema.CallToolResult createHandoff(McpSyncServerExchange exchange, Map<String, Object> args) {
        String from = getString(args, "from");
        String to = getString(args, "to");
        String message = getString(args, "message");
        String contextJson = getString(args, "context");

        var state = loadState();
        String now = Instant.now().toString();

        Map<String, Object> context = null;
        if (contextJson != null && !contextJson.isBlank()) {
            try {
                context = GSON.fromJson(contextJson, new TypeToken<Map<String, Object>>(){}.getType());
            } catch (Exception e) {
                context = Map.of("raw", contextJson);
            }
        }

        var handoff = new Handoff(generateId(), from, to, message, context, now, false);

        var handoffs = new ArrayList<>(state.handoffs());
        handoffs.add(handoff);

        saveState(new SharedState(state.projects(), handoffs, state.notes(), state.keyValue(), now));
        return textResult(String.format("Created handoff from %s to %s. ID: %s", from, to, handoff.id()));
    }

    private static McpSchema.CallToolResult getHandoffs(McpSyncServerExchange exchange, Map<String, Object> args) {
        String forSession = getString(args, "for");
        boolean includeRead = getBoolean(args, "includeRead");

        var state = loadState();
        var handoffs = state.handoffs().stream()
            .filter(h -> h.to().equals(forSession))
            .filter(h -> includeRead || !h.read())
            .toList();

        if (handoffs.isEmpty()) {
            return textResult(String.format("No pending handoffs for %s.", forSession));
        }

        var sb = new StringBuilder("# Handoffs for ").append(forSession).append("\n\n");
        for (var h : handoffs) {
            sb.append("## Handoff from ").append(h.from()).append(" (").append(h.id()).append(")\n");
            sb.append("**Created:** ").append(h.createdAt()).append("\n");
            sb.append("**Read:** ").append(h.read() ? "Yes" : "No").append("\n\n");
            sb.append(h.message()).append("\n\n");
            if (h.context() != null && !h.context().isEmpty()) {
                sb.append("**Context:**\n```json\n").append(GSON.toJson(h.context())).append("\n```\n");
            }
            sb.append("---\n\n");
        }
        return textResult(sb.toString());
    }

    private static McpSchema.CallToolResult markHandoffRead(McpSyncServerExchange exchange, Map<String, Object> args) {
        String handoffId = getString(args, "handoffId");

        var state = loadState();

        var handoffs = new ArrayList<Handoff>();
        boolean found = false;
        for (var h : state.handoffs()) {
            if (h.id().equals(handoffId)) {
                handoffs.add(h.markAsRead());
                found = true;
            } else {
                handoffs.add(h);
            }
        }

        if (!found) {
            return textResult(String.format("Handoff %s not found.", handoffId));
        }

        String now = Instant.now().toString();
        saveState(new SharedState(state.projects(), handoffs, state.notes(), state.keyValue(), now));
        return textResult(String.format("Marked handoff %s as read.", handoffId));
    }

    private static McpSchema.CallToolResult setNote(McpSyncServerExchange exchange, Map<String, Object> args) {
        String key = getString(args, "key");
        String content = getString(args, "content");

        var state = loadState();
        var notes = new HashMap<>(state.notes());
        notes.put(key, content);

        String now = Instant.now().toString();
        saveState(new SharedState(state.projects(), state.handoffs(), notes, state.keyValue(), now));
        return textResult(String.format("Saved note \"%s\".", key));
    }

    private static McpSchema.CallToolResult getNote(McpSyncServerExchange exchange, Map<String, Object> args) {
        String key = getString(args, "key");
        var state = loadState();
        var note = state.notes().get(key);

        if (note == null) {
            return textResult(String.format("Note \"%s\" not found.", key));
        }
        return textResult(note);
    }

    private static McpSchema.CallToolResult listNotes(McpSyncServerExchange exchange, Map<String, Object> args) {
        var state = loadState();
        var keys = state.notes().keySet();

        if (keys.isEmpty()) {
            return textResult("No notes saved.");
        }

        var sb = new StringBuilder("# Saved Notes\n\n");
        for (var k : keys) {
            sb.append("- ").append(k).append("\n");
        }
        return textResult(sb.toString());
    }

    private static McpSchema.CallToolResult setValue(McpSyncServerExchange exchange, Map<String, Object> args) {
        String key = getString(args, "key");
        String value = getString(args, "value");

        var state = loadState();
        var keyValue = new HashMap<>(state.keyValue());

        Object parsed;
        try {
            parsed = GSON.fromJson(value, Object.class);
        } catch (Exception e) {
            parsed = value;
        }
        keyValue.put(key, parsed);

        String now = Instant.now().toString();
        saveState(new SharedState(state.projects(), state.handoffs(), state.notes(), keyValue, now));
        return textResult(String.format("Stored value for key \"%s\".", key));
    }

    private static McpSchema.CallToolResult getValue(McpSyncServerExchange exchange, Map<String, Object> args) {
        String key = getString(args, "key");
        var state = loadState();
        var value = state.keyValue().get(key);

        if (value == null) {
            return textResult(String.format("Key \"%s\" not found.", key));
        }

        if (value instanceof String) {
            return textResult((String) value);
        }
        return textResult(GSON.toJson(value));
    }

    private static McpSchema.CallToolResult getStateSummary(McpSyncServerExchange exchange, Map<String, Object> args) {
        var state = loadState();

        int projectCount = state.projects().size();
        long unreadHandoffs = state.handoffs().stream().filter(h -> !h.read()).count();
        int noteCount = state.notes().size();
        int kvCount = state.keyValue().size();

        return textResult(String.format("""
            # Shared Context Summary

            **Last Updated:** %s
            **State File:** %s

            ## Contents
            - **Projects:** %d
            - **Unread Handoffs:** %d
            - **Notes:** %d
            - **Key-Value Entries:** %d
            """,
            state.lastUpdated(),
            STATE_FILE,
            projectCount,
            unreadHandoffs,
            noteCount,
            kvCount));
    }

    // ============ SCHEMA HELPERS ============

    private static McpSchema.Tool tool(String name, String description, String schema) {
        return new McpSchema.Tool(name, description, schema);
    }

    private static String emptySchema() {
        return "{\"type\": \"object\", \"properties\": {}}";
    }

    private static SchemaBuilder schema(String... optionalProps) {
        return new SchemaBuilder(new String[0], optionalProps);
    }

    private static SchemaBuilder schemaRequired(String... required) {
        return new SchemaBuilder(required, new String[0]);
    }

    private static class SchemaBuilder {
        private final String[] required;
        private String[] optional;

        SchemaBuilder(String[] required, String[] optional) {
            this.required = required;
            this.optional = optional;
        }

        SchemaBuilder with(String... props) {
            this.optional = props;
            return this;
        }

        @Override
        public String toString() {
            var sb = new StringBuilder("{\"type\": \"object\"");
            if (required.length > 0) {
                sb.append(", \"required\": [");
                for (int i = 0; i < required.length; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append("\"").append(required[i]).append("\"");
                }
                sb.append("]");
            }
            sb.append(", \"properties\": {");
            boolean first = true;
            for (String prop : required) {
                if (!first) sb.append(", ");
                sb.append("\"").append(prop).append("\": {\"type\": \"string\"}");
                first = false;
            }
            for (String prop : optional) {
                if (!first) sb.append(", ");
                sb.append("\"").append(prop).append("\": {\"type\": \"string\"}");
                first = false;
            }
            sb.append("}}");
            return sb.toString();
        }
    }

    // ============ HELPERS ============

    private static String getString(Map<String, Object> args, String key) {
        Object value = args.get(key);
        return value != null ? value.toString() : null;
    }

    private static boolean getBoolean(Map<String, Object> args, String key) {
        Object value = args.get(key);
        if (value instanceof Boolean) return (Boolean) value;
        if (value instanceof String) return Boolean.parseBoolean((String) value);
        return false;
    }

    private static McpSchema.CallToolResult textResult(String text) {
        return new McpSchema.CallToolResult(
            List.of(new McpSchema.TextContent(text)),
            false
        );
    }

    private static SharedState loadState() {
        try {
            if (Files.exists(STATE_FILE)) {
                String json = Files.readString(STATE_FILE);
                return GSON.fromJson(json, SharedState.class);
            }
        } catch (IOException e) {
            // Fall through to return empty state
        }
        return SharedState.empty();
    }

    private static void saveState(SharedState state) {
        try {
            Files.createDirectories(STATE_FILE.getParent());
            Files.writeString(STATE_FILE, GSON.toJson(state));
        } catch (IOException e) {
            throw new RuntimeException("Failed to save state", e);
        }
    }

    private static String generateId() {
        return UUID.randomUUID().toString().substring(0, 8);
    }
}
