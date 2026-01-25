package io.surfworks.warpforge.data.hub;

import io.surfworks.warpforge.data.WarpForge;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * Client for downloading models and datasets from HuggingFace Hub.
 *
 * <p>Handles authentication, rate limiting, and automatic retries.
 */
public final class HubClient {

    private static final String HUB_API_URL = "https://huggingface.co/api";
    private static final String HUB_DOWNLOAD_URL = "https://huggingface.co";
    private static final Gson GSON = new Gson();

    private final WarpForge.Config config;
    private final HttpClient httpClient;

    public HubClient(WarpForge.Config config) {
        this.config = config;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(30))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
    }

    /**
     * Download a model to the specified directory.
     */
    public void downloadModel(String repoId, Path targetDir) throws IOException {
        downloadModel(repoId, targetDir, config.progressListener());
    }

    /**
     * Download a model to the specified directory with progress reporting.
     */
    public void downloadModel(String repoId, Path targetDir, ProgressListener listener) throws IOException {
        downloadRepo(repoId, "model", targetDir, listener);
    }

    /**
     * Download a dataset to the specified directory.
     */
    public void downloadDataset(String repoId, Path targetDir) throws IOException {
        downloadDataset(repoId, targetDir, config.progressListener());
    }

    /**
     * Download a dataset to the specified directory with progress reporting.
     */
    public void downloadDataset(String repoId, Path targetDir, ProgressListener listener) throws IOException {
        downloadRepo(repoId, "dataset", targetDir, listener);
    }

    private void downloadRepo(String repoId, String repoType, Path targetDir,
                              ProgressListener listener) throws IOException {
        Files.createDirectories(targetDir);

        // Get list of files
        List<FileInfo> files = listRepoFiles(repoId, repoType);

        // Filter to relevant files
        List<FileInfo> toDownload = files.stream()
                .filter(f -> shouldDownload(f, repoType))
                .toList();

        if (toDownload.isEmpty()) {
            throw new IOException("No downloadable files found in " + repoId);
        }

        // Calculate total size
        long totalSize = toDownload.stream()
                .mapToLong(FileInfo::size)
                .filter(s -> s > 0)
                .sum();

        // Initialize progress
        DownloadProgress progress = DownloadProgress.starting(repoId, toDownload.size(), totalSize);
        listener.onProgress(progress);

        long totalBytesRead = 0;

        // Download each file
        for (int i = 0; i < toDownload.size(); i++) {
            FileInfo file = toDownload.get(i);
            Path targetPath = targetDir.resolve(file.path());
            Files.createDirectories(targetPath.getParent());

            progress = progress.startFile(file.path(), i, file.size());

            if (Files.exists(targetPath) && Files.size(targetPath) == file.size()) {
                // File already downloaded and correct size - skip
                progress = progress.skipped();
                listener.onProgress(progress);
                totalBytesRead += file.size();
                continue;
            }

            final long currentTotalBytes = totalBytesRead;
            final DownloadProgress currentProgress = progress;

            downloadFile(repoId, repoType, file, targetPath, bytesRead -> {
                DownloadProgress updated = currentProgress.withProgress(bytesRead, currentTotalBytes + bytesRead);
                listener.onProgress(updated);
            });

            totalBytesRead += file.size() > 0 ? file.size() : Files.size(targetPath);
            progress = progress.withProgress(file.size() > 0 ? file.size() : Files.size(targetPath), totalBytesRead);
            progress = progress.fileComplete();
            listener.onProgress(progress);
        }

        // Final complete event
        progress = progress.complete();
        listener.onProgress(progress);
    }

    /**
     * Download a single file with progress callback.
     *
     * @param repoId           Repository ID
     * @param repoType         "model" or "dataset"
     * @param file             File info
     * @param targetPath       Target path for the downloaded file
     * @param progressCallback Called with bytes read so far (may be null)
     */
    public void downloadFile(String repoId, String repoType, FileInfo file, Path targetPath,
                             java.util.function.LongConsumer progressCallback) throws IOException {
        String urlPath = repoType.equals("dataset")
                ? "/datasets/" + repoId + "/resolve/main/" + file.path()
                : "/" + repoId + "/resolve/main/" + file.path();

        URI uri = URI.create(HUB_DOWNLOAD_URL + urlPath);

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(uri)
                .timeout(Duration.ofSeconds(config.downloadTimeoutSeconds()))
                .GET();

        if (config.hubToken() != null && !config.hubToken().isEmpty()) {
            requestBuilder.header("Authorization", "Bearer " + config.hubToken());
        }

        HttpRequest request = requestBuilder.build();

        int retries = config.downloadRetries();
        IOException lastException = null;

        for (int attempt = 0; attempt < retries; attempt++) {
            try {
                HttpResponse<InputStream> response = httpClient.send(request,
                        HttpResponse.BodyHandlers.ofInputStream());

                int status = response.statusCode();
                if (status == 401) {
                    throw new IOException("Authentication required. Set HF_TOKEN environment variable " +
                            "or use WarpForge.configure().hubToken(\"...\")");
                } else if (status == 403) {
                    throw new IOException("Access denied to " + repoId +
                            ". You may need to accept the model's license at https://huggingface.co/" + repoId);
                } else if (status == 404) {
                    throw new IOException("File not found: " + file.path() + " in " + repoId);
                } else if (status >= 400) {
                    throw new IOException("HTTP error " + status + " downloading " + file.path());
                }

                // Stream to file with progress
                try (InputStream in = response.body();
                     OutputStream out = Files.newOutputStream(targetPath)) {
                    byte[] buffer = new byte[8192];
                    long totalRead = 0;
                    int bytesRead;

                    while ((bytesRead = in.read(buffer)) != -1) {
                        out.write(buffer, 0, bytesRead);
                        totalRead += bytesRead;
                        if (progressCallback != null) {
                            progressCallback.accept(totalRead);
                        }
                    }
                }

                return; // Success

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Download interrupted", e);
            } catch (IOException e) {
                lastException = e;
                if (attempt < retries - 1) {
                    try {
                        Thread.sleep(1000L * (attempt + 1)); // Exponential backoff
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new IOException("Download interrupted", ie);
                    }
                }
            }
        }

        throw new IOException("Failed to download " + file.path() + " after " + retries + " attempts", lastException);
    }

    /**
     * List files in a repository.
     */
    public List<FileInfo> listRepoFiles(String repoId, String repoType) throws IOException {
        String apiPath = repoType.equals("dataset")
                ? "/datasets/" + repoId
                : "/models/" + repoId;

        URI uri = URI.create(HUB_API_URL + apiPath);

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(uri)
                .timeout(Duration.ofSeconds(30))
                .GET();

        if (config.hubToken() != null && !config.hubToken().isEmpty()) {
            requestBuilder.header("Authorization", "Bearer " + config.hubToken());
        }

        HttpRequest request = requestBuilder.build();

        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            int status = response.statusCode();
            if (status == 401) {
                throw new IOException("Authentication required for " + repoId);
            } else if (status == 403) {
                throw new IOException("Access denied to " + repoId + ". " +
                        "Visit https://huggingface.co/" + repoId + " to accept the license.");
            } else if (status == 404) {
                throw new IOException("Repository not found: " + repoId);
            } else if (status >= 400) {
                throw new IOException("HTTP error " + status + " accessing " + repoId);
            }

            JsonObject repo = GSON.fromJson(response.body(), JsonObject.class);
            JsonArray siblings = repo.getAsJsonArray("siblings");

            List<FileInfo> files = new ArrayList<>();
            if (siblings != null) {
                for (JsonElement sibling : siblings) {
                    JsonObject obj = sibling.getAsJsonObject();
                    String filename = obj.get("rfilename").getAsString();
                    long size = obj.has("size") ? obj.get("size").getAsLong() : -1;
                    files.add(new FileInfo(filename, size));
                }
            }

            return files;

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Request interrupted", e);
        }
    }

    /**
     * Check if the user has access to a gated model.
     */
    public AccessStatus checkAccess(String repoId) throws IOException {
        String apiPath = "/models/" + repoId;
        URI uri = URI.create(HUB_API_URL + apiPath);

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .uri(uri)
                .timeout(Duration.ofSeconds(30))
                .GET();

        if (config.hubToken() != null && !config.hubToken().isEmpty()) {
            requestBuilder.header("Authorization", "Bearer " + config.hubToken());
        }

        try {
            HttpResponse<String> response = httpClient.send(requestBuilder.build(),
                    HttpResponse.BodyHandlers.ofString());

            return switch (response.statusCode()) {
                case 200 -> AccessStatus.GRANTED;
                case 401 -> AccessStatus.AUTH_REQUIRED;
                case 403 -> AccessStatus.PENDING_OR_DENIED;
                case 404 -> AccessStatus.NOT_FOUND;
                default -> AccessStatus.ERROR;
            };
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Request interrupted", e);
        }
    }

    private boolean shouldDownload(FileInfo file, String repoType) {
        String name = file.path().toLowerCase();

        // Always download these
        if (name.endsWith(".safetensors") || name.endsWith(".gguf")) {
            return true;
        }

        // Config files
        if (name.equals("config.json") || name.equals("tokenizer.json") ||
                name.equals("tokenizer_config.json") || name.equals("special_tokens_map.json") ||
                name.equals("generation_config.json")) {
            return true;
        }

        // Dataset files
        if (repoType.equals("dataset")) {
            return name.endsWith(".json") || name.endsWith(".parquet") ||
                    name.endsWith(".arrow") || name.endsWith(".csv");
        }

        return false;
    }

    /**
     * Information about a file in a repository.
     */
    public record FileInfo(String path, long size) {}

    /**
     * Access status for a gated repository.
     */
    public enum AccessStatus {
        GRANTED,
        AUTH_REQUIRED,
        PENDING_OR_DENIED,
        NOT_FOUND,
        ERROR
    }
}
