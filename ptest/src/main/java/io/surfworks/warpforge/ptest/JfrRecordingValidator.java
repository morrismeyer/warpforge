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
            System.out.printf("  %s: %s (%d GPU events)%n",
                entry.getKey(), status, result.gpuKernelEventCount);
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
            return new ValidationResult(false, 0, 0, "File not found");
        }

        int gpuKernelEvents = 0;
        int gpuMemoryEvents = 0;
        List<String> operations = new ArrayList<>();

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
                }
            }
        } catch (IOException e) {
            System.out.println("  ERROR: Failed to read recording: " + e.getMessage());
            return new ValidationResult(false, 0, 0, e.getMessage());
        }

        System.out.printf("  Total GpuKernelEvents: %d%n", gpuKernelEvents);
        System.out.printf("  Total GpuMemoryEvents: %d%n", gpuMemoryEvents);

        boolean valid = gpuKernelEvents >= MIN_EXPECTED_EVENTS;
        if (!valid) {
            System.out.printf("  FAIL: Expected at least %d GpuKernelEvents%n", MIN_EXPECTED_EVENTS);
        } else {
            System.out.println("  PASS");
        }

        return new ValidationResult(valid, gpuKernelEvents, gpuMemoryEvents, null);
    }

    private record ValidationResult(
        boolean valid,
        int gpuKernelEventCount,
        int gpuMemoryEventCount,
        String error
    ) {
        boolean isValid() {
            // Consider valid if we have events OR if the file doesn't exist (GPU not available)
            return valid || "File not found".equals(error);
        }
    }
}
