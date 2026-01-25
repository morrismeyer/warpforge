package io.surfworks.warpforge.data.cache;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.Duration;
import java.time.Instant;
import java.util.Comparator;
import java.util.stream.Stream;

/**
 * Manages the local cache for downloaded models and datasets.
 *
 * <p>Default cache location: ~/.warpforge/cache/
 */
public final class CacheManager {

    private static final String DEFAULT_CACHE_DIR = ".warpforge/cache";

    private final Path cacheRoot;

    public CacheManager() {
        this(defaultCacheDir());
    }

    public CacheManager(Path cacheRoot) {
        this.cacheRoot = cacheRoot;
    }

    private static Path defaultCacheDir() {
        String home = System.getProperty("user.home");
        return Path.of(home, DEFAULT_CACHE_DIR);
    }

    /**
     * Get the cache root directory.
     */
    public Path cacheRoot() {
        return cacheRoot;
    }

    /**
     * Get the cache directory for a specific model.
     */
    public Path modelCacheDir(String modelId) {
        // Convert "meta-llama/Llama-3.1-8B" to "models/meta-llama/Llama-3.1-8B"
        return cacheRoot.resolve("models").resolve(modelId.replace("/", "/"));
    }

    /**
     * Get the cache directory for a specific dataset.
     */
    public Path datasetCacheDir(String datasetId) {
        return cacheRoot.resolve("datasets").resolve(datasetId.replace("/", "/"));
    }

    /**
     * Check if a model is cached.
     */
    public boolean isModelCached(String modelId) {
        Path dir = modelCacheDir(modelId);
        if (!Files.isDirectory(dir)) {
            return false;
        }
        // Check for at least one .safetensors or .gguf file
        try (Stream<Path> files = Files.list(dir)) {
            return files.anyMatch(p -> {
                String name = p.getFileName().toString();
                return name.endsWith(".safetensors") || name.endsWith(".gguf");
            });
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Check if a dataset is cached.
     */
    public boolean isDatasetCached(String datasetId) {
        Path dir = datasetCacheDir(datasetId);
        return Files.isDirectory(dir);
    }

    /**
     * Ensure cache directories exist.
     */
    public void ensureCacheDir(String modelId) throws IOException {
        Files.createDirectories(modelCacheDir(modelId));
    }

    /**
     * Get total cache size in bytes.
     */
    public long totalCacheSize() throws IOException {
        if (!Files.exists(cacheRoot)) {
            return 0;
        }
        try (Stream<Path> walk = Files.walk(cacheRoot)) {
            return walk.filter(Files::isRegularFile)
                    .mapToLong(p -> {
                        try {
                            return Files.size(p);
                        } catch (IOException e) {
                            return 0;
                        }
                    })
                    .sum();
        }
    }

    /**
     * Delete cached model.
     */
    public void deleteModel(String modelId) throws IOException {
        deleteRecursively(modelCacheDir(modelId));
    }

    /**
     * Delete cached dataset.
     */
    public void deleteDataset(String datasetId) throws IOException {
        deleteRecursively(datasetCacheDir(datasetId));
    }

    /**
     * Delete all cached items older than the specified duration.
     */
    public long pruneOlderThan(Duration age) throws IOException {
        if (!Files.exists(cacheRoot)) {
            return 0;
        }

        Instant cutoff = Instant.now().minus(age);
        long[] freedBytes = {0};

        try (Stream<Path> walk = Files.walk(cacheRoot)) {
            walk.filter(Files::isRegularFile)
                    .filter(p -> {
                        try {
                            BasicFileAttributes attrs = Files.readAttributes(p, BasicFileAttributes.class);
                            return attrs.lastModifiedTime().toInstant().isBefore(cutoff);
                        } catch (IOException e) {
                            return false;
                        }
                    })
                    .forEach(p -> {
                        try {
                            freedBytes[0] += Files.size(p);
                            Files.delete(p);
                        } catch (IOException ignored) {
                        }
                    });
        }

        // Clean up empty directories
        cleanEmptyDirs(cacheRoot);

        return freedBytes[0];
    }

    /**
     * Clear entire cache.
     */
    public void clearAll() throws IOException {
        deleteRecursively(cacheRoot);
    }

    private void deleteRecursively(Path path) throws IOException {
        if (!Files.exists(path)) {
            return;
        }
        try (Stream<Path> walk = Files.walk(path)) {
            walk.sorted(Comparator.reverseOrder())
                    .forEach(p -> {
                        try {
                            Files.delete(p);
                        } catch (IOException ignored) {
                        }
                    });
        }
    }

    private void cleanEmptyDirs(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }
        try (Stream<Path> walk = Files.walk(root)) {
            walk.filter(Files::isDirectory)
                    .sorted(Comparator.reverseOrder())
                    .filter(p -> !p.equals(root))
                    .forEach(p -> {
                        try (Stream<Path> contents = Files.list(p)) {
                            if (contents.findAny().isEmpty()) {
                                Files.delete(p);
                            }
                        } catch (IOException ignored) {
                        }
                    });
        }
    }
}
