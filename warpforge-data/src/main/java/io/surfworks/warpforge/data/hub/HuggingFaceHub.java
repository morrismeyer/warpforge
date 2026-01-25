package io.surfworks.warpforge.data.hub;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * HuggingFace Hub API client for model discovery and download.
 *
 * <p>Provides functionality to:
 * <ul>
 *   <li>Search for models by name, task, or tags</li>
 *   <li>Get model metadata and file listings</li>
 *   <li>Download model files with progress tracking</li>
 *   <li>Manage local cache of downloaded models</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * HuggingFaceHub hub = HuggingFaceHub.create();
 *
 * // Search for models
 * List<ModelInfo> models = hub.search("bert", Filter.task("fill-mask"), 10);
 *
 * // Get model info
 * ModelInfo info = hub.modelInfo("bert-base-uncased");
 *
 * // Download model
 * Path modelPath = hub.download("bert-base-uncased", progress -> {
 *     System.out.printf("Downloaded: %.1f%%\n", progress * 100);
 * });
 * }</pre>
 */
public final class HuggingFaceHub {

    private static final String API_BASE = "https://huggingface.co/api";
    private static final String CDN_BASE = "https://huggingface.co";
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private final Path cacheDir;
    private final String token;
    private final int connectTimeout;
    private final int readTimeout;

    private HuggingFaceHub(Builder builder) {
        this.cacheDir = builder.cacheDir;
        this.token = builder.token;
        this.connectTimeout = builder.connectTimeout;
        this.readTimeout = builder.readTimeout;
    }

    /**
     * Create a hub client with default settings.
     */
    public static HuggingFaceHub create() {
        return builder().build();
    }

    /**
     * Create a builder for customizing the hub client.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Search for models.
     *
     * @param query Search query
     * @param limit Maximum number of results
     * @return List of matching models
     */
    public List<ModelInfo> search(String query, int limit) throws IOException {
        return search(query, Filter.none(), limit);
    }

    /**
     * Search for models with filters.
     *
     * @param query Search query
     * @param filter Search filters
     * @param limit Maximum number of results
     * @return List of matching models
     */
    public List<ModelInfo> search(String query, Filter filter, int limit) throws IOException {
        StringBuilder url = new StringBuilder(API_BASE + "/models?");
        url.append("search=").append(URLEncoder.encode(query, StandardCharsets.UTF_8));
        url.append("&limit=").append(limit);

        if (filter.task != null) {
            url.append("&pipeline_tag=").append(URLEncoder.encode(filter.task, StandardCharsets.UTF_8));
        }
        if (filter.library != null) {
            url.append("&library=").append(URLEncoder.encode(filter.library, StandardCharsets.UTF_8));
        }
        if (filter.author != null) {
            url.append("&author=").append(URLEncoder.encode(filter.author, StandardCharsets.UTF_8));
        }

        JsonArray results = fetchJson(url.toString()).getAsJsonArray();
        List<ModelInfo> models = new ArrayList<>();

        for (JsonElement elem : results) {
            models.add(parseModelInfo(elem.getAsJsonObject()));
        }

        return models;
    }

    /**
     * Get detailed model information.
     *
     * @param modelId Model ID (e.g., "bert-base-uncased" or "google/bert_uncased_L-2_H-128_A-2")
     * @return Model information
     */
    public ModelInfo modelInfo(String modelId) throws IOException {
        String url = API_BASE + "/models/" + URLEncoder.encode(modelId, StandardCharsets.UTF_8);
        JsonObject json = fetchJson(url).getAsJsonObject();
        return parseModelInfo(json);
    }

    /**
     * List files in a model repository.
     *
     * @param modelId Model ID
     * @return List of file information
     */
    public List<FileInfo> listFiles(String modelId) throws IOException {
        return listFiles(modelId, null);
    }

    /**
     * List files in a model repository at a specific revision.
     *
     * @param modelId Model ID
     * @param revision Git revision (branch, tag, or commit)
     * @return List of file information
     */
    public List<FileInfo> listFiles(String modelId, String revision) throws IOException {
        String url = API_BASE + "/models/" + URLEncoder.encode(modelId, StandardCharsets.UTF_8);
        if (revision != null) {
            url += "/tree/" + URLEncoder.encode(revision, StandardCharsets.UTF_8);
        }

        JsonObject json = fetchJson(url).getAsJsonObject();
        List<FileInfo> files = new ArrayList<>();

        if (json.has("siblings")) {
            for (JsonElement sibling : json.getAsJsonArray("siblings")) {
                JsonObject s = sibling.getAsJsonObject();
                files.add(new FileInfo(
                        s.get("rfilename").getAsString(),
                        s.has("size") ? s.get("size").getAsLong() : -1,
                        s.has("lfs") ? s.getAsJsonObject("lfs").get("sha256").getAsString() : null
                ));
            }
        }

        return files;
    }

    /**
     * Download a model to the cache directory.
     *
     * @param modelId Model ID
     * @return Path to downloaded model directory
     */
    public Path download(String modelId) throws IOException {
        return download(modelId, null, null);
    }

    /**
     * Download a model with progress callback.
     *
     * @param modelId Model ID
     * @param progressCallback Called with progress (0.0 to 1.0) during download
     * @return Path to downloaded model directory
     */
    public Path download(String modelId, Consumer<Double> progressCallback) throws IOException {
        return download(modelId, null, progressCallback);
    }

    /**
     * Download specific files from a model.
     *
     * @param modelId Model ID
     * @param filenames Files to download (null for all)
     * @param progressCallback Called with progress during download
     * @return Path to downloaded model directory
     */
    public Path download(String modelId, List<String> filenames, Consumer<Double> progressCallback)
            throws IOException {
        Path modelDir = cacheDir.resolve("models--" + modelId.replace("/", "--"));
        Path snapshotDir = modelDir.resolve("snapshots").resolve("main");
        Files.createDirectories(snapshotDir);

        // Get file list
        List<FileInfo> allFiles = listFiles(modelId);
        List<FileInfo> toDownload = filenames == null ? allFiles :
                allFiles.stream().filter(f -> filenames.contains(f.filename)).toList();

        // Calculate total size
        long totalSize = toDownload.stream().mapToLong(f -> Math.max(0, f.size)).sum();
        final long[] downloadedSize = {0};

        for (FileInfo file : toDownload) {
            Path destPath = snapshotDir.resolve(file.filename);
            Files.createDirectories(destPath.getParent());

            // Skip if already exists with correct size
            if (Files.exists(destPath) && file.size > 0 && Files.size(destPath) == file.size) {
                downloadedSize[0] += file.size;
                if (progressCallback != null && totalSize > 0) {
                    progressCallback.accept((double) downloadedSize[0] / totalSize);
                }
                continue;
            }

            // Download file
            String fileUrl = CDN_BASE + "/" + modelId + "/resolve/main/" +
                    URLEncoder.encode(file.filename, StandardCharsets.UTF_8);

            final long baseDownloaded = downloadedSize[0];
            downloadFile(fileUrl, destPath, downloaded -> {
                if (progressCallback != null && totalSize > 0) {
                    progressCallback.accept((double) (baseDownloaded + downloaded) / totalSize);
                }
            });

            downloadedSize[0] += Math.max(0, file.size);
        }

        return snapshotDir;
    }

    /**
     * Download a single file.
     *
     * @param modelId Model ID
     * @param filename File name
     * @return Path to downloaded file
     */
    public Path downloadFile(String modelId, String filename) throws IOException {
        Path modelDir = cacheDir.resolve("models--" + modelId.replace("/", "--"));
        Path snapshotDir = modelDir.resolve("snapshots").resolve("main");
        Files.createDirectories(snapshotDir);

        Path destPath = snapshotDir.resolve(filename);
        if (Files.exists(destPath)) {
            return destPath;
        }

        String fileUrl = CDN_BASE + "/" + modelId + "/resolve/main/" +
                URLEncoder.encode(filename, StandardCharsets.UTF_8);
        downloadFile(fileUrl, destPath, null);

        return destPath;
    }

    /**
     * Get the local cache path for a model.
     *
     * @param modelId Model ID
     * @return Cache path (may not exist if not downloaded)
     */
    public Path cachePath(String modelId) {
        return cacheDir.resolve("models--" + modelId.replace("/", "--"))
                .resolve("snapshots").resolve("main");
    }

    /**
     * Check if a model is cached locally.
     *
     * @param modelId Model ID
     * @return true if model is in cache
     */
    public boolean isCached(String modelId) {
        Path cachePath = cachePath(modelId);
        return Files.isDirectory(cachePath) && !isEmpty(cachePath);
    }

    /**
     * Clear the model cache.
     */
    public void clearCache() throws IOException {
        if (Files.exists(cacheDir)) {
            try (var walker = Files.walk(cacheDir)) {
                walker.sorted((a, b) -> b.compareTo(a))
                        .forEach(path -> {
                            try {
                                Files.delete(path);
                            } catch (IOException e) {
                                // Ignore
                            }
                        });
            }
        }
    }

    private boolean isEmpty(Path dir) {
        try (var stream = Files.list(dir)) {
            return stream.findFirst().isEmpty();
        } catch (IOException e) {
            return true;
        }
    }

    private JsonElement fetchJson(String url) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) URI.create(url).toURL().openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(connectTimeout);
        conn.setReadTimeout(readTimeout);
        conn.setRequestProperty("Accept", "application/json");

        if (token != null) {
            conn.setRequestProperty("Authorization", "Bearer " + token);
        }

        try {
            if (conn.getResponseCode() != 200) {
                throw new IOException("HTTP " + conn.getResponseCode() + ": " + conn.getResponseMessage());
            }

            try (var reader = new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8)) {
                return JsonParser.parseReader(reader);
            }
        } finally {
            conn.disconnect();
        }
    }

    private void downloadFile(String url, Path destination, Consumer<Long> progressCallback)
            throws IOException {
        HttpURLConnection conn = (HttpURLConnection) URI.create(url).toURL().openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(connectTimeout);
        conn.setReadTimeout(readTimeout);
        conn.setInstanceFollowRedirects(true);

        if (token != null) {
            conn.setRequestProperty("Authorization", "Bearer " + token);
        }

        try {
            if (conn.getResponseCode() != 200) {
                throw new IOException("HTTP " + conn.getResponseCode() + ": " + conn.getResponseMessage());
            }

            Path tempFile = destination.resolveSibling(destination.getFileName() + ".tmp");
            try (InputStream in = conn.getInputStream()) {
                if (progressCallback != null) {
                    long downloaded = 0;
                    byte[] buffer = new byte[8192];
                    try (var out = Files.newOutputStream(tempFile)) {
                        int read;
                        while ((read = in.read(buffer)) != -1) {
                            out.write(buffer, 0, read);
                            downloaded += read;
                            progressCallback.accept(downloaded);
                        }
                    }
                } else {
                    Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
                }
            }

            Files.move(tempFile, destination, StandardCopyOption.REPLACE_EXISTING);
        } finally {
            conn.disconnect();
        }
    }

    private ModelInfo parseModelInfo(JsonObject json) {
        String id = json.get("id").getAsString();
        String modelId = json.has("modelId") ? json.get("modelId").getAsString() : id;

        String author = null;
        if (id.contains("/")) {
            author = id.substring(0, id.indexOf('/'));
        }

        String pipelineTag = json.has("pipeline_tag") ? json.get("pipeline_tag").getAsString() : null;
        long downloads = json.has("downloads") ? json.get("downloads").getAsLong() : 0;
        long likes = json.has("likes") ? json.get("likes").getAsLong() : 0;
        String lastModified = json.has("lastModified") ? json.get("lastModified").getAsString() : null;

        List<String> tags = new ArrayList<>();
        if (json.has("tags")) {
            for (JsonElement tag : json.getAsJsonArray("tags")) {
                tags.add(tag.getAsString());
            }
        }

        Map<String, Object> config = new HashMap<>();
        if (json.has("config")) {
            JsonObject configObj = json.getAsJsonObject("config");
            for (String key : configObj.keySet()) {
                JsonElement elem = configObj.get(key);
                if (elem.isJsonPrimitive()) {
                    if (elem.getAsJsonPrimitive().isNumber()) {
                        config.put(key, elem.getAsNumber());
                    } else if (elem.getAsJsonPrimitive().isBoolean()) {
                        config.put(key, elem.getAsBoolean());
                    } else {
                        config.put(key, elem.getAsString());
                    }
                }
            }
        }

        return new ModelInfo(id, modelId, author, pipelineTag, downloads, likes, lastModified, tags, config);
    }

    /**
     * Search filter.
     */
    public record Filter(String task, String library, String author) {
        public static Filter none() {
            return new Filter(null, null, null);
        }

        public static Filter task(String task) {
            return new Filter(task, null, null);
        }

        public static Filter library(String library) {
            return new Filter(null, library, null);
        }

        public static Filter author(String author) {
            return new Filter(null, null, author);
        }

        public Filter withTask(String task) {
            return new Filter(task, this.library, this.author);
        }

        public Filter withLibrary(String library) {
            return new Filter(this.task, library, this.author);
        }

        public Filter withAuthor(String author) {
            return new Filter(this.task, this.library, author);
        }
    }

    /**
     * Model information.
     */
    public record ModelInfo(
            String id,
            String modelId,
            String author,
            String pipelineTag,
            long downloads,
            long likes,
            String lastModified,
            List<String> tags,
            Map<String, Object> config
    ) {
        public boolean hasTag(String tag) {
            return tags.contains(tag);
        }

        public boolean isTransformer() {
            return hasTag("transformers");
        }

        public boolean isPyTorch() {
            return hasTag("pytorch");
        }

        public boolean isSafetensors() {
            return hasTag("safetensors");
        }
    }

    /**
     * File information.
     */
    public record FileInfo(String filename, long size, String sha256) {
        public boolean isModel() {
            return filename.endsWith(".safetensors") ||
                    filename.endsWith(".bin") ||
                    filename.endsWith(".onnx") ||
                    filename.endsWith(".gguf");
        }

        public boolean isConfig() {
            return filename.equals("config.json") ||
                    filename.equals("tokenizer_config.json") ||
                    filename.equals("tokenizer.json");
        }

        public String sizeFormatted() {
            if (size < 0) return "unknown";
            if (size < 1024) return size + " B";
            if (size < 1024 * 1024) return String.format("%.1f KB", size / 1024.0);
            if (size < 1024 * 1024 * 1024) return String.format("%.1f MB", size / (1024.0 * 1024));
            return String.format("%.2f GB", size / (1024.0 * 1024 * 1024));
        }
    }

    /**
     * Builder for HuggingFaceHub.
     */
    public static class Builder {
        private Path cacheDir = getDefaultCacheDir();
        private String token = System.getenv("HF_TOKEN");
        private int connectTimeout = 30000;
        private int readTimeout = 120000;

        private Builder() {}

        public Builder cacheDir(Path cacheDir) {
            this.cacheDir = cacheDir;
            return this;
        }

        public Builder token(String token) {
            this.token = token;
            return this;
        }

        public Builder connectTimeout(int millis) {
            this.connectTimeout = millis;
            return this;
        }

        public Builder readTimeout(int millis) {
            this.readTimeout = millis;
            return this;
        }

        public HuggingFaceHub build() {
            return new HuggingFaceHub(this);
        }

        private static Path getDefaultCacheDir() {
            String xdgCache = System.getenv("XDG_CACHE_HOME");
            if (xdgCache != null) {
                return Path.of(xdgCache, "huggingface", "hub");
            }
            String hfHome = System.getenv("HF_HOME");
            if (hfHome != null) {
                return Path.of(hfHome, "hub");
            }
            return Path.of(System.getProperty("user.home"), ".cache", "huggingface", "hub");
        }
    }
}
