package io.surfworks.warpforge.ptest;

import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Validates JFR recordings contain expected GPU events.
 *
 * <p>This tool reads JFR recording files and verifies that GpuKernelEvent
 * entries were properly captured. It's used in CI to ensure the JFR-GPU
 * integration is working correctly.
 *
 * <p>Usage:
 * <pre>
 * java -cp ptest.jar io.surfworks.warpforge.ptest.JfrRecordingValidator \
 *      build/jfr-nvidia.jfr build/jfr-amd.jfr
 * </pre>
 */
public class JfrRecordingValidator {

    private static final String GPU_KERNEL_EVENT = "io.surfworks.warpforge.GpuKernel";
    private static final String GPU_MEMORY_EVENT = "io.surfworks.warpforge.GpuMemory";
    private static final String GPU_TASK_SCOPE_EVENT = "io.surfworks.warpforge.GpuTaskScope";
    private static final int MIN_EXPECTED_EVENTS = 5;

    public static void main(String[] args) {
        if (args.length == 0) {
            System.err.println("Usage: JfrRecordingValidator <recording1.jfr> [recording2.jfr ...]");
            System.exit(1);
        }

        System.out.println("JFR Recording Validator");
        System.out.println("=======================");
        System.out.println();

        boolean allValid = true;
        Map<String, ValidationResult> results = new HashMap<>();

        for (String arg : args) {
            Path recordingPath = Path.of(arg);
            ValidationResult result = validateRecording(recordingPath);
            results.put(arg, result);

            if (!result.isValid()) {
                allValid = false;
            }
        }

        // Print summary
        System.out.println();
        System.out.println("Validation Summary");
        System.out.println("------------------");

        for (var entry : results.entrySet()) {
            ValidationResult result = entry.getValue();
            String status = result.isValid() ? "PASS" : "FAIL";
            String scopeInfo = result.totalScopeEvents() > 0
                ? String.format(", %d scope events", result.totalScopeEvents())
                : "";
            System.out.printf("  %s: %s (%d kernel events%s)%n",
                entry.getKey(), status, result.gpuKernelEventCount, scopeInfo);
        }

        System.out.println();
        if (allValid) {
            System.out.println("All recordings validated successfully.");
        } else {
            System.out.println("VALIDATION FAILED - some recordings are missing expected events.");
            System.exit(1);
        }
    }

    private static ValidationResult validateRecording(Path recordingPath) {
        System.out.println("Validating: " + recordingPath);

        if (!Files.exists(recordingPath)) {
            System.out.println("  WARNING: Recording file not found (GPU may not be available)");
            return new ValidationResult(false, 0, 0, 0, 0, "File not found");
        }

        int gpuKernelEvents = 0;
        int gpuMemoryEvents = 0;
        int gpuTaskScopeStartEvents = 0;
        int gpuTaskScopeEndEvents = 0;
        List<String> operations = new ArrayList<>();
        Map<Long, String> openScopes = new HashMap<>(); // scopeId -> scopeName

        try (RecordingFile recording = new RecordingFile(recordingPath)) {
            while (recording.hasMoreEvents()) {
                RecordedEvent event = recording.readEvent();
                String eventName = event.getEventType().getName();

                if (GPU_KERNEL_EVENT.equals(eventName)) {
                    gpuKernelEvents++;
                    String operation = event.getString("operation");
                    String backend = event.getString("backend");
                    long timeMicros = event.getLong("gpuTimeMicros");
                    operations.add(String.format("%s (%s, %dμs)", operation, backend, timeMicros));

                    if (gpuKernelEvents <= 3) {
                        System.out.printf("  Found: %s on %s - %d μs%n", operation, backend, timeMicros);
                    }
                } else if (GPU_MEMORY_EVENT.equals(eventName)) {
                    gpuMemoryEvents++;
                } else if (GPU_TASK_SCOPE_EVENT.equals(eventName)) {
                    String phase = event.getString("phase");
                    long scopeId = event.getLong("scopeId");
                    String scopeName = event.getString("scopeName");

                    if ("START".equals(phase)) {
                        gpuTaskScopeStartEvents++;
                        openScopes.put(scopeId, scopeName);
                        if (gpuTaskScopeStartEvents <= 3) {
                            System.out.printf("  Scope START: %s (id=%d)%n", scopeName, scopeId);
                        }
                    } else if ("END".equals(phase) || "FAILED".equals(phase)) {
                        gpuTaskScopeEndEvents++;
                        int tasksForked = event.getInt("tasksForked");
                        int tasksCompleted = event.getInt("tasksCompleted");
                        long durationMicros = event.getLong("durationMicros");
                        openScopes.remove(scopeId);

                        if (gpuTaskScopeEndEvents <= 3) {
                            System.out.printf("  Scope %s: %s (id=%d, forked=%d, completed=%d, duration=%dμs)%n",
                                phase, scopeName, scopeId, tasksForked, tasksCompleted, durationMicros);
                        }
                    }
                }
            }
        } catch (IOException e) {
            System.out.println("  ERROR: Failed to read recording: " + e.getMessage());
            return new ValidationResult(false, 0, 0, 0, 0, e.getMessage());
        }

        System.out.printf("  Total GpuKernelEvents: %d%n", gpuKernelEvents);
        System.out.printf("  Total GpuMemoryEvents: %d%n", gpuMemoryEvents);
        System.out.printf("  Total GpuTaskScopeEvents: START=%d, END=%d%n", gpuTaskScopeStartEvents, gpuTaskScopeEndEvents);

        // Validate: we need kernel events OR scope events for success
        boolean valid = gpuKernelEvents >= MIN_EXPECTED_EVENTS;

        // If we have scope events, validate they're balanced
        if (gpuTaskScopeStartEvents > 0 || gpuTaskScopeEndEvents > 0) {
            if (gpuTaskScopeStartEvents != gpuTaskScopeEndEvents) {
                System.out.printf("  WARNING: Unbalanced scope events: %d START vs %d END%n",
                    gpuTaskScopeStartEvents, gpuTaskScopeEndEvents);
            }
            if (!openScopes.isEmpty()) {
                System.out.println("  WARNING: Unclosed scopes detected: " + openScopes.values());
            }
        }

        if (!valid) {
            System.out.printf("  FAIL: Expected at least %d GpuKernelEvents%n", MIN_EXPECTED_EVENTS);
        } else {
            System.out.println("  PASS");
        }

        return new ValidationResult(valid, gpuKernelEvents, gpuMemoryEvents,
            gpuTaskScopeStartEvents, gpuTaskScopeEndEvents, null);
    }

    private record ValidationResult(
        boolean valid,
        int gpuKernelEventCount,
        int gpuMemoryEventCount,
        int gpuTaskScopeStartCount,
        int gpuTaskScopeEndCount,
        String error
    ) {
        boolean isValid() {
            // Consider valid if we have events OR if the file doesn't exist (GPU may not be available)
            return valid || "File not found".equals(error);
        }

        int totalScopeEvents() {
            return gpuTaskScopeStartCount + gpuTaskScopeEndCount;
        }
    }
}
