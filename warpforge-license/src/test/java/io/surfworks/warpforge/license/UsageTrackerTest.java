package io.surfworks.warpforge.license;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link UsageTracker}.
 */
class UsageTrackerTest {

    @TempDir
    Path tempDir;

    private UsageTracker tracker;

    @BeforeEach
    void setUp() {
        tracker = new UsageTracker(tempDir);
    }

    @Test
    @DisplayName("getTodayCount returns 0 for new tracker")
    void getTodayCount_newTracker_returnsZero() {
        assertEquals(0, tracker.getTodayCount());
    }

    @Test
    @DisplayName("increment increases today's count")
    void increment_increasesCount() {
        assertEquals(0, tracker.getTodayCount());

        tracker.increment();
        assertEquals(1, tracker.getTodayCount());

        tracker.increment();
        assertEquals(2, tracker.getTodayCount());

        tracker.increment();
        assertEquals(3, tracker.getTodayCount());
    }

    @Test
    @DisplayName("resetToday clears today's count")
    void resetToday_clearsCount() {
        tracker.increment();
        tracker.increment();
        assertEquals(2, tracker.getTodayCount());

        tracker.resetToday();
        assertEquals(0, tracker.getTodayCount());
    }

    @Test
    @DisplayName("Usage persists to file")
    void usage_persistsToFile() {
        tracker.increment();
        tracker.increment();

        // Create a new tracker instance
        var newTracker = new UsageTracker(tempDir);
        assertEquals(2, newTracker.getTodayCount());
    }

    @Test
    @DisplayName("Usage file is created in config directory")
    void usageFile_isCreated() {
        tracker.increment();

        Path usageFile = tempDir.resolve("usage.json");
        assertTrue(Files.exists(usageFile));
    }

    @Test
    @DisplayName("Corrupted usage file is handled gracefully")
    void corruptedFile_handledGracefully() throws IOException {
        Path usageFile = tempDir.resolve("usage.json");
        Files.writeString(usageFile, "not valid json{{{");

        var newTracker = new UsageTracker(tempDir);
        assertEquals(0, newTracker.getTodayCount());

        // Should still be able to increment
        newTracker.increment();
        assertEquals(1, newTracker.getTodayCount());
    }

    @Test
    @DisplayName("Empty usage file is handled gracefully")
    void emptyFile_handledGracefully() throws IOException {
        Path usageFile = tempDir.resolve("usage.json");
        Files.writeString(usageFile, "");

        var newTracker = new UsageTracker(tempDir);
        assertEquals(0, newTracker.getTodayCount());
    }

    @Test
    @DisplayName("Null JSON value is handled gracefully")
    void nullJsonValue_handledGracefully() throws IOException {
        Path usageFile = tempDir.resolve("usage.json");
        Files.writeString(usageFile, "null");

        var newTracker = new UsageTracker(tempDir);
        assertEquals(0, newTracker.getTodayCount());
    }

    @Test
    @DisplayName("Old entries are cleaned up after 7 days")
    void oldEntries_cleanedUp() throws IOException {
        // Manually write old entries
        String today = LocalDate.now().format(DateTimeFormatter.ISO_LOCAL_DATE);
        String oldDate = LocalDate.now().minusDays(10).format(DateTimeFormatter.ISO_LOCAL_DATE);
        String recentDate = LocalDate.now().minusDays(3).format(DateTimeFormatter.ISO_LOCAL_DATE);

        Path usageFile = tempDir.resolve("usage.json");
        Files.writeString(usageFile, String.format("""
            {
              "dailyCounts": {
                "%s": 5,
                "%s": 10,
                "%s": 3
              }
            }
            """, oldDate, recentDate, today));

        // Increment to trigger cleanup
        tracker = new UsageTracker(tempDir);
        int beforeIncrement = tracker.getTodayCount();
        tracker.increment();

        // Old date should be cleaned up, recent date should remain
        String json = Files.readString(usageFile);
        assertFalse(json.contains(oldDate), "Old date should be removed");
        assertTrue(json.contains(recentDate), "Recent date should remain");
        assertTrue(json.contains(today), "Today should remain");
    }

    @Test
    @DisplayName("Multiple trackers can share same file")
    void multipleTrackers_shareFile() {
        tracker.increment();

        var tracker2 = new UsageTracker(tempDir);
        assertEquals(1, tracker2.getTodayCount());

        tracker2.increment();

        var tracker3 = new UsageTracker(tempDir);
        assertEquals(2, tracker3.getTodayCount());
    }
}
