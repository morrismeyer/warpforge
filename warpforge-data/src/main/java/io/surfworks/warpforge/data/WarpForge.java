package io.surfworks.warpforge.data;

import io.surfworks.warpforge.data.cache.CacheManager;
import io.surfworks.warpforge.data.dataset.DatasetSource;
import io.surfworks.warpforge.data.format.SafeTensors;
import io.surfworks.warpforge.data.hub.HubClient;
import io.surfworks.warpforge.data.hub.ProgressListener;
import io.surfworks.warpforge.data.model.ModelSource;
import io.surfworks.warpforge.data.model.SafeTensorsModel;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * Main entry point for the WarpForge data loading API.
 *
 * <p>This class provides "it just works" model and dataset loading:
 * <pre>{@code
 * // Load a model from HuggingFace
 * var model = WarpForge.model("meta-llama/Llama-3.1-8B");
 *
 * // Load from local path
 * var model = WarpForge.model(Path.of("./my-model.safetensors"));
 *
 * // Load a dataset
 * var dataset = WarpForge.dataset("rajpurkar/squad");
 * }</pre>
 *
 * <p>Models are automatically cached locally after first download.
 */
public final class WarpForge {

    private static volatile Config config = Config.defaults();

    private WarpForge() {}

    /**
     * Load a model by identifier or path.
     *
     * <p>Supported formats:
     * <ul>
     *   <li>HuggingFace repo ID: "meta-llama/Llama-3.1-8B"</li>
     *   <li>Local path: "./models/my-model.safetensors"</li>
     *   <li>Local directory: "./models/llama/" (loads all .safetensors files)</li>
     * </ul>
     *
     * @param identifier Model identifier or path
     * @return ModelSource for accessing the model's tensors
     * @throws IOException if the model cannot be loaded
     */
    public static ModelSource model(String identifier) throws IOException {
        Objects.requireNonNull(identifier, "identifier must not be null");

        // Check if it's a local path
        Path localPath = Path.of(identifier);
        if (Files.exists(localPath)) {
            return loadLocalModel(localPath, identifier);
        }

        // Treat as HuggingFace repo ID
        return loadHubModel(identifier);
    }

    /**
     * Load a model from a local path.
     */
    public static ModelSource model(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        if (!Files.exists(path)) {
            throw new IOException("Model not found: " + path);
        }
        return loadLocalModel(path, path.getFileName().toString());
    }

    /**
     * Load a dataset by identifier.
     *
     * @param identifier Dataset identifier (e.g., "rajpurkar/squad")
     * @return DatasetSource for accessing the dataset
     * @throws IOException if the dataset cannot be loaded
     */
    public static DatasetSource dataset(String identifier) throws IOException {
        Objects.requireNonNull(identifier, "identifier must not be null");

        // Check cache first
        CacheManager cache = config.cacheManager();
        Path cacheDir = cache.datasetCacheDir(identifier);

        if (!cache.isDatasetCached(identifier)) {
            // Download from hub
            HubClient hub = new HubClient(config);
            hub.downloadDataset(identifier, cacheDir);
        }

        return DatasetSource.open(identifier, cacheDir);
    }

    /**
     * Configure WarpForge settings.
     */
    public static ConfigBuilder configure() {
        return new ConfigBuilder();
    }

    /**
     * Set the global configuration.
     */
    public static void setConfig(Config newConfig) {
        config = Objects.requireNonNull(newConfig);
    }

    /**
     * Get the current configuration.
     */
    public static Config config() {
        return config;
    }

    private static ModelSource loadLocalModel(Path path, String id) throws IOException {
        if (Files.isDirectory(path)) {
            // Load all .safetensors files in directory
            return SafeTensorsModel.fromDirectory(id, path);
        } else if (path.toString().endsWith(".safetensors")) {
            return SafeTensorsModel.fromFile(id, path);
        } else if (path.toString().endsWith(".gguf")) {
            throw new UnsupportedOperationException("GGUF format not yet implemented");
        } else {
            throw new IOException("Unknown model format: " + path);
        }
    }

    private static ModelSource loadHubModel(String repoId) throws IOException {
        CacheManager cache = config.cacheManager();
        Path cacheDir = cache.modelCacheDir(repoId);

        if (!cache.isModelCached(repoId)) {
            // Download from hub
            HubClient hub = new HubClient(config);
            hub.downloadModel(repoId, cacheDir);
        }

        return SafeTensorsModel.fromDirectory(repoId, cacheDir);
    }

    /**
     * Global configuration for WarpForge.
     */
    public record Config(
            Path cacheDir,
            String hubToken,
            boolean autoDownload,
            int downloadTimeoutSeconds,
            int downloadRetries,
            ProgressListener progressListener
    ) {
        public static Config defaults() {
            String home = System.getProperty("user.home");
            return new Config(
                    Path.of(home, ".warpforge", "cache"),
                    System.getenv("HF_TOKEN"),
                    true,
                    3600,  // 1 hour timeout
                    3,
                    ProgressListener.NONE
            );
        }

        public CacheManager cacheManager() {
            return new CacheManager(cacheDir);
        }
    }

    /**
     * Builder for WarpForge configuration.
     */
    public static final class ConfigBuilder {
        private Path cacheDir;
        private String hubToken;
        private boolean autoDownload = true;
        private int downloadTimeoutSeconds = 3600;
        private int downloadRetries = 3;
        private ProgressListener progressListener = ProgressListener.NONE;

        ConfigBuilder() {
            Config defaults = Config.defaults();
            this.cacheDir = defaults.cacheDir();
            this.hubToken = defaults.hubToken();
        }

        public ConfigBuilder cacheDir(Path cacheDir) {
            this.cacheDir = cacheDir;
            return this;
        }

        public ConfigBuilder cacheDir(String cacheDir) {
            this.cacheDir = Path.of(cacheDir);
            return this;
        }

        public ConfigBuilder hubToken(String token) {
            this.hubToken = token;
            return this;
        }

        public ConfigBuilder autoDownload(boolean autoDownload) {
            this.autoDownload = autoDownload;
            return this;
        }

        public ConfigBuilder downloadTimeout(int seconds) {
            this.downloadTimeoutSeconds = seconds;
            return this;
        }

        public ConfigBuilder downloadRetries(int retries) {
            this.downloadRetries = retries;
            return this;
        }

        /**
         * Set a progress listener for download operations.
         *
         * <p>Built-in listeners:
         * <ul>
         *   <li>{@link ProgressListener#NONE} - No progress output (default)</li>
         *   <li>{@link ProgressListener#CONSOLE} - Detailed console progress with speed/ETA</li>
         *   <li>{@link ProgressListener#MINIMAL} - Minimal console output (file names only)</li>
         * </ul>
         *
         * <p>Example with console progress:
         * <pre>{@code
         * WarpForge.configure()
         *     .progressListener(ProgressListener.CONSOLE)
         *     .apply();
         * }</pre>
         */
        public ConfigBuilder progressListener(ProgressListener listener) {
            this.progressListener = listener != null ? listener : ProgressListener.NONE;
            return this;
        }

        public Config build() {
            return new Config(cacheDir, hubToken, autoDownload, downloadTimeoutSeconds,
                              downloadRetries, progressListener);
        }

        public void apply() {
            WarpForge.setConfig(build());
        }
    }
}
