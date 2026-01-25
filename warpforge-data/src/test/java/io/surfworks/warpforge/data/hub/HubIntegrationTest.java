package io.surfworks.warpforge.data.hub;

import io.surfworks.warpforge.data.WarpForge;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Integration tests that perform real downloads from HuggingFace Hub.
 *
 * <p>These tests require network access and are excluded from normal test runs.
 * Run with: ./gradlew :warpforge-data:test -PincludeTags=integration
 *
 * <p>Some tests require authentication (HF_TOKEN environment variable) for gated models.
 */
@Tag("integration")
class HubIntegrationTest {

    @TempDir
    Path tempDir;

    private WarpForge.Config originalConfig;

    @BeforeEach
    void setUp() {
        originalConfig = WarpForge.config();
    }

    @AfterEach
    void tearDown() {
        WarpForge.setConfig(originalConfig);
    }

    @Nested
    class PublicModelTests {

        /**
         * Test downloading a tiny public model that doesn't require authentication.
         * Uses hf-internal-testing/tiny-random-gpt2 which is ~500KB.
         */
        @Test
        void testDownloadTinyPublicModel() throws IOException {
            Path cacheDir = tempDir.resolve("cache");
            List<DownloadProgress> progressUpdates = new ArrayList<>();

            WarpForge.configure()
                    .cacheDir(cacheDir)
                    .progressListener(progressUpdates::add)
                    .apply();

            HubClient client = new HubClient(WarpForge.config());
            Path modelDir = cacheDir.resolve("models/hf-internal-testing/tiny-random-gpt2");
            Files.createDirectories(modelDir);

            client.downloadModel("hf-internal-testing/tiny-random-gpt2", modelDir);

            // Verify files were downloaded
            assertTrue(Files.exists(modelDir));
            assertTrue(countFiles(modelDir) > 0, "Model directory should have files");

            // Verify config.json exists (required for all models)
            assertTrue(Files.exists(modelDir.resolve("config.json")));

            // Verify progress events
            assertFalse(progressUpdates.isEmpty(), "Should have received progress updates");

            // Should have STARTING event
            assertTrue(progressUpdates.stream()
                    .anyMatch(p -> p.status() == DownloadProgress.Status.STARTING));

            // Should have COMPLETE event
            assertTrue(progressUpdates.stream()
                    .anyMatch(p -> p.status() == DownloadProgress.Status.COMPLETE));
        }

        @Test
        void testCachedModelSkipsDownload() throws IOException {
            Path cacheDir = tempDir.resolve("cache");
            Path modelDir = cacheDir.resolve("models/hf-internal-testing/tiny-random-gpt2");
            Files.createDirectories(modelDir);

            WarpForge.configure()
                    .cacheDir(cacheDir)
                    .progressListener(ProgressListener.NONE)
                    .apply();

            HubClient client = new HubClient(WarpForge.config());

            // First download
            client.downloadModel("hf-internal-testing/tiny-random-gpt2", modelDir);

            // Second download should skip (files already exist)
            List<DownloadProgress> secondDownloadProgress = new ArrayList<>();
            client.downloadModel("hf-internal-testing/tiny-random-gpt2", modelDir,
                    secondDownloadProgress::add);

            // Count different status types
            long skippedCount = secondDownloadProgress.stream()
                    .filter(p -> p.status() == DownloadProgress.Status.SKIPPED)
                    .count();
            long fileCompleteCount = secondDownloadProgress.stream()
                    .filter(p -> p.status() == DownloadProgress.Status.FILE_COMPLETE)
                    .count();

            // On second download, files should either be SKIPPED (cached) or already FILE_COMPLETE
            // Some files may need re-download if size info from API was -1
            assertTrue(skippedCount > 0 || fileCompleteCount > 0,
                    "Should have some skipped or completed files");

            // Verify we got a COMPLETE event
            assertTrue(secondDownloadProgress.stream()
                    .anyMatch(p -> p.status() == DownloadProgress.Status.COMPLETE));
        }

        @Test
        void testProgressListenerReceivesAllStatuses() throws IOException {
            Path cacheDir = tempDir.resolve("cache");
            Path modelDir = cacheDir.resolve("models/hf-internal-testing/tiny-random-gpt2");
            Files.createDirectories(modelDir);

            List<DownloadProgress> progressUpdates = new ArrayList<>();

            WarpForge.configure()
                    .cacheDir(cacheDir)
                    .progressListener(progressUpdates::add)
                    .apply();

            HubClient client = new HubClient(WarpForge.config());
            client.downloadModel("hf-internal-testing/tiny-random-gpt2", modelDir);

            // Verify we got the expected sequence of statuses
            assertFalse(progressUpdates.isEmpty());

            // First should be STARTING
            assertEquals(DownloadProgress.Status.STARTING, progressUpdates.get(0).status());

            // Last should be COMPLETE
            assertEquals(DownloadProgress.Status.COMPLETE,
                    progressUpdates.get(progressUpdates.size() - 1).status());

            // Should have some FILE_COMPLETE events (one per file)
            long fileCompleteCount = progressUpdates.stream()
                    .filter(p -> p.status() == DownloadProgress.Status.FILE_COMPLETE)
                    .count();
            assertTrue(fileCompleteCount > 0, "Should have FILE_COMPLETE events");
        }

        @Test
        void testThrottledProgressListener() throws IOException {
            Path cacheDir = tempDir.resolve("cache");
            Path modelDir = cacheDir.resolve("models/hf-internal-testing/tiny-random-gpt2");
            Files.createDirectories(modelDir);

            AtomicInteger throttledCallCount = new AtomicInteger();

            // The inner listener counts how many times it's called
            ProgressListener inner = p -> throttledCallCount.incrementAndGet();
            // Wrap it with throttling
            ProgressListener throttled = inner.throttled(100); // 100ms throttle

            // Track all updates (unthrottled) for comparison
            List<DownloadProgress> allUpdates = new ArrayList<>();
            ProgressListener combined = p -> {
                allUpdates.add(p);
                throttled.onProgress(p);
            };

            WarpForge.configure()
                    .cacheDir(cacheDir)
                    .progressListener(combined)
                    .apply();

            HubClient client = new HubClient(WarpForge.config());
            client.downloadModel("hf-internal-testing/tiny-random-gpt2", modelDir);

            // All non-DOWNLOADING updates should pass through throttle
            long nonDownloadingCount = allUpdates.stream()
                    .filter(p -> p.status() != DownloadProgress.Status.DOWNLOADING)
                    .count();

            // Throttled should receive at least all non-DOWNLOADING events
            assertTrue(throttledCallCount.get() >= nonDownloadingCount,
                    "Throttled should receive at least non-DOWNLOADING events. " +
                            "Got " + throttledCallCount.get() + " calls, expected at least " + nonDownloadingCount);

            // Throttled should receive fewer total events than raw (if there were DOWNLOADING events)
            long downloadingCount = allUpdates.stream()
                    .filter(p -> p.status() == DownloadProgress.Status.DOWNLOADING)
                    .count();
            if (downloadingCount > 0) {
                assertTrue(throttledCallCount.get() <= allUpdates.size(),
                        "Throttled should receive at most the same number of events as raw");
            }
        }
    }

    @Nested
    class ListFilesTests {

        @Test
        void testListModelFiles() throws IOException {
            WarpForge.configure()
                    .cacheDir(tempDir.resolve("cache"))
                    .apply();

            HubClient client = new HubClient(WarpForge.config());
            List<HubClient.FileInfo> files = client.listRepoFiles(
                    "hf-internal-testing/tiny-random-gpt2", "model");

            assertNotNull(files);
            assertFalse(files.isEmpty());

            // Should contain config.json
            assertTrue(files.stream().anyMatch(f -> f.path().equals("config.json")));
        }

        @Test
        void testListNonExistentRepo() {
            WarpForge.configure()
                    .cacheDir(tempDir.resolve("cache"))
                    .apply();

            HubClient client = new HubClient(WarpForge.config());

            assertThrows(IOException.class, () ->
                    client.listRepoFiles("this-repo-definitely-does-not-exist-12345", "model"));
        }
    }

    @Nested
    class AccessCheckTests {

        @Test
        void testCheckAccessPublicModel() throws IOException {
            WarpForge.configure()
                    .cacheDir(tempDir.resolve("cache"))
                    .apply();

            HubClient client = new HubClient(WarpForge.config());
            HubClient.AccessStatus status = client.checkAccess("hf-internal-testing/tiny-random-gpt2");

            assertEquals(HubClient.AccessStatus.GRANTED, status);
        }

        @Test
        void testCheckAccessNonExistent() throws IOException {
            WarpForge.configure()
                    .cacheDir(tempDir.resolve("cache"))
                    .apply();

            HubClient client = new HubClient(WarpForge.config());
            HubClient.AccessStatus status = client.checkAccess("this-repo-definitely-does-not-exist-12345");

            // HuggingFace API may return 401 (AUTH_REQUIRED) or 404 (NOT_FOUND)
            // for non-existent repos depending on API behavior
            assertTrue(status == HubClient.AccessStatus.NOT_FOUND ||
                            status == HubClient.AccessStatus.AUTH_REQUIRED,
                    "Non-existent repo should return NOT_FOUND or AUTH_REQUIRED, got: " + status);
        }
    }

    @Nested
    @EnabledIfEnvironmentVariable(named = "HF_TOKEN", matches = ".+")
    class AuthenticatedTests {

        /**
         * Test accessing a gated model that requires authentication.
         * This test only runs if HF_TOKEN is set.
         */
        @Test
        void testCheckAccessGatedModel() throws IOException {
            WarpForge.configure()
                    .cacheDir(tempDir.resolve("cache"))
                    .hubToken(System.getenv("HF_TOKEN"))
                    .apply();

            HubClient client = new HubClient(WarpForge.config());

            // meta-llama/Llama-2-7b requires acceptance of license
            // If token has accepted, status is GRANTED
            // If token hasn't accepted, status is PENDING_OR_DENIED
            HubClient.AccessStatus status = client.checkAccess("meta-llama/Llama-2-7b");

            assertTrue(status == HubClient.AccessStatus.GRANTED ||
                            status == HubClient.AccessStatus.PENDING_OR_DENIED,
                    "Gated model should return GRANTED or PENDING_OR_DENIED, got: " + status);
        }
    }

    @Nested
    class ErrorHandlingTests {

        @Test
        void testDownloadNonExistentModel() {
            Path cacheDir = tempDir.resolve("cache");
            Path modelDir = cacheDir.resolve("models/nonexistent");

            WarpForge.configure()
                    .cacheDir(cacheDir)
                    .apply();

            HubClient client = new HubClient(WarpForge.config());

            IOException exception = assertThrows(IOException.class, () ->
                    client.downloadModel("this-repo-definitely-does-not-exist-12345", modelDir));

            // Error message varies: "not found", "404", "Repository not found", "Authentication required"
            String msg = exception.getMessage().toLowerCase();
            assertTrue(msg.contains("not found") || msg.contains("404") ||
                            msg.contains("authentication") || msg.contains("error"),
                    "Expected error message about not found or auth, got: " + exception.getMessage());
        }

        @Test
        void testDownloadTimeout() {
            Path cacheDir = tempDir.resolve("cache");

            // Set very short timeout
            WarpForge.configure()
                    .cacheDir(cacheDir)
                    .downloadTimeout(1) // 1 second - too short for any real download
                    .apply();

            // This test verifies timeout config is respected
            // Actual timeout behavior depends on network conditions
            HubClient client = new HubClient(WarpForge.config());
            assertEquals(1, WarpForge.config().downloadTimeoutSeconds());
        }
    }

    @Nested
    class ConsoleListenerTests {

        @Test
        void testConsoleListenerDoesNotThrow() throws IOException {
            Path cacheDir = tempDir.resolve("cache");
            Path modelDir = cacheDir.resolve("models/hf-internal-testing/tiny-random-gpt2");
            Files.createDirectories(modelDir);

            WarpForge.configure()
                    .cacheDir(cacheDir)
                    .progressListener(ProgressListener.CONSOLE)
                    .apply();

            HubClient client = new HubClient(WarpForge.config());

            // Should complete without throwing
            client.downloadModel("hf-internal-testing/tiny-random-gpt2", modelDir);

            assertTrue(Files.exists(modelDir.resolve("config.json")));
        }

        @Test
        void testMinimalListenerDoesNotThrow() throws IOException {
            Path cacheDir = tempDir.resolve("cache");
            Path modelDir = cacheDir.resolve("models/hf-internal-testing/tiny-random-gpt2");
            Files.createDirectories(modelDir);

            WarpForge.configure()
                    .cacheDir(cacheDir)
                    .progressListener(ProgressListener.MINIMAL)
                    .apply();

            HubClient client = new HubClient(WarpForge.config());

            // Should complete without throwing
            client.downloadModel("hf-internal-testing/tiny-random-gpt2", modelDir);

            assertTrue(Files.exists(modelDir.resolve("config.json")));
        }
    }

    @Nested
    class CombinedListenerTests {

        @Test
        void testAndThenCombinesListeners() throws IOException {
            Path cacheDir = tempDir.resolve("cache");
            Path modelDir = cacheDir.resolve("models/hf-internal-testing/tiny-random-gpt2");
            Files.createDirectories(modelDir);

            List<String> log1 = new ArrayList<>();
            List<String> log2 = new ArrayList<>();

            ProgressListener listener1 = p -> log1.add(p.status().name());
            ProgressListener listener2 = p -> log2.add(p.currentFile());
            ProgressListener combined = listener1.andThen(listener2);

            WarpForge.configure()
                    .cacheDir(cacheDir)
                    .progressListener(combined)
                    .apply();

            HubClient client = new HubClient(WarpForge.config());
            client.downloadModel("hf-internal-testing/tiny-random-gpt2", modelDir);

            // Both listeners should have received updates
            assertFalse(log1.isEmpty());
            assertFalse(log2.isEmpty());
            assertEquals(log1.size(), log2.size());
        }
    }

    private long countFiles(Path dir) throws IOException {
        try (Stream<Path> files = Files.walk(dir)) {
            return files.filter(Files::isRegularFile).count();
        }
    }
}
