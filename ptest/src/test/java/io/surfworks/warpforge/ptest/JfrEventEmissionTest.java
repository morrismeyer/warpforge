package io.surfworks.warpforge.ptest;

import io.surfworks.warpforge.core.jfr.GpuKernelEvent;
import io.surfworks.warpforge.core.jfr.GpuMemoryEvent;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests that JFR GPU events are properly emitted and captured.
 *
 * <p>These tests verify the JFR event infrastructure without requiring
 * actual GPU hardware - they test the event emission and capture mechanism.
 */
@Tag("cpu")
@DisplayName("JFR GPU Event Emission Tests")
class JfrEventEmissionTest {

    @TempDir
    Path tempDir;

    @Test
    @DisplayName("GpuKernelEvent is captured in JFR recording")
    void testGpuKernelEventCapture() throws Exception {
        Path recordingPath = tempDir.resolve("kernel-test.jfr");

        // Start recording
        try (Recording recording = new Recording()) {
            recording.enable(GpuKernelEvent.class);
            recording.start();

            // Emit some events
            for (int i = 0; i < 5; i++) {
                GpuKernelEvent event = new GpuKernelEvent();
                event.operation = "TestMatMul";
                event.shape = "1024x1024";
                event.gpuTimeMicros = 1000 + i * 100;
                event.backend = "TestBackend";
                event.deviceIndex = 0;
                event.tier = "TEST";
                event.teraflops = 10.5 + i;
                event.commit();
            }

            recording.stop();
            recording.dump(recordingPath);
        }

        // Verify events were captured
        List<RecordedEvent> events = new ArrayList<>();
        try (RecordingFile file = new RecordingFile(recordingPath)) {
            while (file.hasMoreEvents()) {
                RecordedEvent event = file.readEvent();
                if (event.getEventType().getName().equals("io.surfworks.warpforge.GpuKernel")) {
                    events.add(event);
                }
            }
        }

        assertEquals(5, events.size(), "Should have captured 5 GpuKernelEvents");

        // Verify first event content
        RecordedEvent first = events.get(0);
        assertEquals("TestMatMul", first.getString("operation"));
        assertEquals("1024x1024", first.getString("shape"));
        assertEquals(1000, first.getLong("gpuTimeMicros"));
        assertEquals("TestBackend", first.getString("backend"));
        assertEquals(0, first.getInt("deviceIndex"));
    }

    @Test
    @DisplayName("GpuMemoryEvent is captured in JFR recording")
    void testGpuMemoryEventCapture() throws Exception {
        Path recordingPath = tempDir.resolve("memory-test.jfr");

        // Start recording
        try (Recording recording = new Recording()) {
            recording.enable(GpuMemoryEvent.class);
            recording.start();

            // Emit memory transfer events
            GpuMemoryEvent h2d = new GpuMemoryEvent();
            h2d.direction = "H2D";
            h2d.bytes = 1024 * 1024;
            h2d.timeMicros = 500;
            h2d.bandwidthGBps = 12.5;
            h2d.deviceIndex = 0;
            h2d.async = false;
            h2d.commit();

            GpuMemoryEvent d2h = new GpuMemoryEvent();
            d2h.direction = "D2H";
            d2h.bytes = 1024 * 1024;
            d2h.timeMicros = 600;
            d2h.bandwidthGBps = 10.4;
            d2h.deviceIndex = 0;
            d2h.async = true;
            d2h.commit();

            recording.stop();
            recording.dump(recordingPath);
        }

        // Verify events were captured
        List<RecordedEvent> events = new ArrayList<>();
        try (RecordingFile file = new RecordingFile(recordingPath)) {
            while (file.hasMoreEvents()) {
                RecordedEvent event = file.readEvent();
                if (event.getEventType().getName().equals("io.surfworks.warpforge.GpuMemory")) {
                    events.add(event);
                }
            }
        }

        assertEquals(2, events.size(), "Should have captured 2 GpuMemoryEvents");

        // Verify H2D event
        RecordedEvent h2dEvent = events.stream()
            .filter(e -> "H2D".equals(e.getString("direction")))
            .findFirst()
            .orElseThrow();
        assertEquals(1024 * 1024, h2dEvent.getLong("bytes"));
        assertFalse(h2dEvent.getBoolean("async"));

        // Verify D2H event
        RecordedEvent d2hEvent = events.stream()
            .filter(e -> "D2H".equals(e.getString("direction")))
            .findFirst()
            .orElseThrow();
        assertTrue(d2hEvent.getBoolean("async"));
    }

    @Test
    @DisplayName("Multiple event types captured together")
    void testMixedEventCapture() throws Exception {
        Path recordingPath = tempDir.resolve("mixed-test.jfr");

        try (Recording recording = new Recording()) {
            recording.enable(GpuKernelEvent.class);
            recording.enable(GpuMemoryEvent.class);
            recording.start();

            // Simulate a typical operation: transfer -> compute -> transfer
            GpuMemoryEvent upload = new GpuMemoryEvent();
            upload.direction = "H2D";
            upload.bytes = 4096;
            upload.timeMicros = 10;
            upload.commit();

            GpuKernelEvent kernel = new GpuKernelEvent();
            kernel.operation = "Add";
            kernel.shape = "64";
            kernel.gpuTimeMicros = 50;
            kernel.backend = "PTX";
            kernel.commit();

            GpuMemoryEvent download = new GpuMemoryEvent();
            download.direction = "D2H";
            download.bytes = 4096;
            download.timeMicros = 10;
            download.commit();

            recording.stop();
            recording.dump(recordingPath);
        }

        // Count events by type
        int kernelCount = 0;
        int memoryCount = 0;

        try (RecordingFile file = new RecordingFile(recordingPath)) {
            while (file.hasMoreEvents()) {
                RecordedEvent event = file.readEvent();
                String name = event.getEventType().getName();
                if (name.equals("io.surfworks.warpforge.GpuKernel")) {
                    kernelCount++;
                } else if (name.equals("io.surfworks.warpforge.GpuMemory")) {
                    memoryCount++;
                }
            }
        }

        assertEquals(1, kernelCount, "Should have 1 kernel event");
        assertEquals(2, memoryCount, "Should have 2 memory events");
    }

    @Test
    @DisplayName("Events have correct timestamps and ordering")
    void testEventTimestamps() throws Exception {
        Path recordingPath = tempDir.resolve("timestamp-test.jfr");

        try (Recording recording = new Recording()) {
            recording.enable(GpuKernelEvent.class);
            recording.start();

            // Emit events with small delays
            for (int i = 0; i < 3; i++) {
                GpuKernelEvent event = new GpuKernelEvent();
                event.operation = "Op" + i;
                event.gpuTimeMicros = i * 100;
                event.commit();

                Thread.sleep(10); // Small delay between events
            }

            recording.stop();
            recording.dump(recordingPath);
        }

        // Verify timestamps are increasing
        List<Long> timestamps = new ArrayList<>();
        try (RecordingFile file = new RecordingFile(recordingPath)) {
            while (file.hasMoreEvents()) {
                RecordedEvent event = file.readEvent();
                if (event.getEventType().getName().equals("io.surfworks.warpforge.GpuKernel")) {
                    timestamps.add(event.getStartTime().toEpochMilli());
                }
            }
        }

        assertEquals(3, timestamps.size());
        for (int i = 1; i < timestamps.size(); i++) {
            assertTrue(timestamps.get(i) >= timestamps.get(i - 1),
                "Timestamps should be non-decreasing");
        }
    }
}
