package io.surfworks.warpforge.license;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

/**
 * Tracks daily usage for the free tier.
 *
 * <p>Stores usage counts in ~/.config/warpforge/usage.json
 */
public class UsageTracker {

    private static final String USAGE_FILE = "usage.json";
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private final Path configDir;
    private final Path usageFile;

    public UsageTracker(Path configDir) {
        this.configDir = configDir;
        this.usageFile = configDir.resolve(USAGE_FILE);
    }

    /**
     * Get the number of traces used today.
     */
    public int getTodayCount() {
        UsageData data = load();
        String today = todayKey();
        return data.dailyCounts.getOrDefault(today, 0);
    }

    /**
     * Increment today's trace count.
     */
    public void increment() {
        UsageData data = load();
        String today = todayKey();
        int count = data.dailyCounts.getOrDefault(today, 0);
        data.dailyCounts.put(today, count + 1);

        // Clean up old entries (keep last 7 days)
        cleanOldEntries(data);

        save(data);
    }

    /**
     * Reset today's count (for testing).
     */
    public void resetToday() {
        UsageData data = load();
        data.dailyCounts.remove(todayKey());
        save(data);
    }

    private UsageData load() {
        if (!Files.exists(usageFile)) {
            return new UsageData();
        }

        try {
            String json = Files.readString(usageFile);
            UsageData data = GSON.fromJson(json, UsageData.class);
            return data != null ? data : new UsageData();
        } catch (Exception e) {
            return new UsageData();
        }
    }

    private void save(UsageData data) {
        try {
            Files.createDirectories(configDir);
            Files.writeString(usageFile, GSON.toJson(data));
        } catch (IOException e) {
            // Best effort - don't fail the operation
        }
    }

    private void cleanOldEntries(UsageData data) {
        LocalDate cutoff = LocalDate.now().minusDays(7);
        data.dailyCounts.entrySet().removeIf(entry -> {
            try {
                LocalDate date = LocalDate.parse(entry.getKey(), DateTimeFormatter.ISO_LOCAL_DATE);
                return date.isBefore(cutoff);
            } catch (Exception e) {
                return true; // Remove malformed entries
            }
        });
    }

    private String todayKey() {
        return LocalDate.now().format(DateTimeFormatter.ISO_LOCAL_DATE);
    }

    private static class UsageData {
        Map<String, Integer> dailyCounts = new HashMap<>();
    }
}
