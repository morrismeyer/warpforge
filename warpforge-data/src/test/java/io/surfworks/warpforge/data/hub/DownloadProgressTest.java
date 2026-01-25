package io.surfworks.warpforge.data.hub;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.time.Instant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DownloadProgressTest {

    @Nested
    class ProgressCalculationTests {

        @Test
        void testFileProgress() {
            DownloadProgress progress = new DownloadProgress(
                    "test/repo", "file.bin", 0, 1,
                    500, 1000,
                    500, 1000,
                    Instant.now(), DownloadProgress.Status.DOWNLOADING
            );

            assertEquals(0.5, progress.fileProgress(), 0.001);
        }

        @Test
        void testFileProgressUnknownSize() {
            DownloadProgress progress = new DownloadProgress(
                    "test/repo", "file.bin", 0, 1,
                    500, -1,
                    500, -1,
                    Instant.now(), DownloadProgress.Status.DOWNLOADING
            );

            assertEquals(-1, progress.fileProgress());
        }

        @Test
        void testTotalProgress() {
            DownloadProgress progress = new DownloadProgress(
                    "test/repo", "file2.bin", 1, 3,
                    250, 500,
                    750, 1500,
                    Instant.now(), DownloadProgress.Status.DOWNLOADING
            );

            assertEquals(0.5, progress.totalProgress(), 0.001);
        }

        @Test
        void testTotalProgressUnknownSize() {
            DownloadProgress progress = new DownloadProgress(
                    "test/repo", "file.bin", 0, 1,
                    500, 1000,
                    500, -1,
                    Instant.now(), DownloadProgress.Status.DOWNLOADING
            );

            assertEquals(-1, progress.totalProgress());
        }
    }

    @Nested
    class ProgressStateTransitionTests {

        @Test
        void testStarting() {
            DownloadProgress progress = DownloadProgress.starting("test/repo", 3, 1500);

            assertEquals("test/repo", progress.repoId());
            assertEquals(3, progress.totalFiles());
            assertEquals(1500, progress.totalSize());
            assertEquals(DownloadProgress.Status.STARTING, progress.status());
        }

        @Test
        void testStartFile() {
            DownloadProgress progress = DownloadProgress.starting("test/repo", 3, 1500);
            progress = progress.startFile("file1.bin", 0, 500);

            assertEquals("file1.bin", progress.currentFile());
            assertEquals(0, progress.fileIndex());
            assertEquals(500, progress.fileSize());
            assertEquals(0, progress.bytesRead());
            assertEquals(DownloadProgress.Status.DOWNLOADING, progress.status());
        }

        @Test
        void testWithProgress() {
            DownloadProgress progress = DownloadProgress.starting("test/repo", 3, 1500)
                    .startFile("file1.bin", 0, 500);

            progress = progress.withProgress(250, 250);

            assertEquals(250, progress.bytesRead());
            assertEquals(250, progress.totalBytesRead());
            assertEquals(DownloadProgress.Status.DOWNLOADING, progress.status());
        }

        @Test
        void testFileComplete() {
            DownloadProgress progress = DownloadProgress.starting("test/repo", 3, 1500)
                    .startFile("file1.bin", 0, 500)
                    .withProgress(500, 500)
                    .fileComplete();

            assertEquals(DownloadProgress.Status.FILE_COMPLETE, progress.status());
        }

        @Test
        void testSkipped() {
            DownloadProgress progress = DownloadProgress.starting("test/repo", 3, 1500)
                    .startFile("file1.bin", 0, 500)
                    .skipped();

            assertEquals(DownloadProgress.Status.SKIPPED, progress.status());
            assertEquals(500, progress.totalBytesRead());
        }

        @Test
        void testComplete() {
            DownloadProgress progress = DownloadProgress.starting("test/repo", 1, 500)
                    .startFile("file1.bin", 0, 500)
                    .withProgress(500, 500)
                    .complete();

            assertEquals(DownloadProgress.Status.COMPLETE, progress.status());
        }

        @Test
        void testFailed() {
            DownloadProgress progress = DownloadProgress.starting("test/repo", 1, 500)
                    .startFile("file1.bin", 0, 500)
                    .withProgress(250, 250)
                    .failed();

            assertEquals(DownloadProgress.Status.FAILED, progress.status());
            assertEquals(250, progress.bytesRead());
        }
    }

    @Nested
    class FormattingTests {

        @Test
        void testFormatBytesBytes() {
            assertEquals("0 B", DownloadProgress.formatBytes(0));
            assertEquals("512 B", DownloadProgress.formatBytes(512));
            assertEquals("1023 B", DownloadProgress.formatBytes(1023));
        }

        @Test
        void testFormatBytesKilobytes() {
            assertEquals("1.0 KB", DownloadProgress.formatBytes(1024));
            assertEquals("1.5 KB", DownloadProgress.formatBytes(1536));
            assertEquals("1023.0 KB", DownloadProgress.formatBytes(1023 * 1024));
        }

        @Test
        void testFormatBytesMegabytes() {
            assertEquals("1.0 MB", DownloadProgress.formatBytes(1024 * 1024));
            assertEquals("100.5 MB", DownloadProgress.formatBytes((long) (100.5 * 1024 * 1024)));
        }

        @Test
        void testFormatBytesGigabytes() {
            assertEquals("1.00 GB", DownloadProgress.formatBytes(1024L * 1024 * 1024));
            assertEquals("4.50 GB", DownloadProgress.formatBytes((long) (4.5 * 1024 * 1024 * 1024)));
        }

        @Test
        void testFormatDurationSeconds() {
            assertEquals("0s", DownloadProgress.formatDuration(Duration.ZERO));
            assertEquals("30s", DownloadProgress.formatDuration(Duration.ofSeconds(30)));
            assertEquals("59s", DownloadProgress.formatDuration(Duration.ofSeconds(59)));
        }

        @Test
        void testFormatDurationMinutes() {
            assertEquals("1m 0s", DownloadProgress.formatDuration(Duration.ofMinutes(1)));
            assertEquals("5m 30s", DownloadProgress.formatDuration(Duration.ofSeconds(330)));
            assertEquals("59m 59s", DownloadProgress.formatDuration(Duration.ofSeconds(3599)));
        }

        @Test
        void testFormatDurationHours() {
            assertEquals("1h 0m", DownloadProgress.formatDuration(Duration.ofHours(1)));
            assertEquals("2h 30m", DownloadProgress.formatDuration(Duration.ofMinutes(150)));
        }

        @Test
        void testToProgressStringDownloading() {
            DownloadProgress progress = new DownloadProgress(
                    "test/repo", "model.safetensors", 0, 2,
                    512 * 1024, 1024 * 1024,
                    512 * 1024, 2 * 1024 * 1024,
                    Instant.now().minusSeconds(1), DownloadProgress.Status.DOWNLOADING
            );

            String str = progress.toProgressString();
            assertTrue(str.contains("[1/2]"));
            assertTrue(str.contains("model.safetensors"));
            assertTrue(str.contains("512.0 KB"));
            assertTrue(str.contains("50.0%"));
        }

        @Test
        void testToProgressStringSkipped() {
            DownloadProgress progress = new DownloadProgress(
                    "test/repo", "config.json", 0, 2,
                    1024, 1024,
                    1024, 2 * 1024 * 1024,
                    Instant.now(), DownloadProgress.Status.SKIPPED
            );

            String str = progress.toProgressString();
            assertTrue(str.contains("[1/2]"));
            assertTrue(str.contains("config.json"));
            assertTrue(str.contains("(cached)"));
        }

        @Test
        void testToProgressStringComplete() {
            DownloadProgress progress = new DownloadProgress(
                    "test/repo", "model.safetensors", 0, 1,
                    1024 * 1024, 1024 * 1024,
                    1024 * 1024, 1024 * 1024,
                    Instant.now(), DownloadProgress.Status.FILE_COMPLETE
            );

            String str = progress.toProgressString();
            assertTrue(str.contains("1.0 MB"));
            assertTrue(str.contains("complete"));
        }
    }

    @Nested
    class SpeedCalculationTests {

        @Test
        void testBytesPerSecond() {
            // 1MB downloaded in 2 seconds
            DownloadProgress progress = new DownloadProgress(
                    "test/repo", "file.bin", 0, 1,
                    1024 * 1024, 1024 * 1024,
                    1024 * 1024, 1024 * 1024,
                    Instant.now().minusSeconds(2), DownloadProgress.Status.DOWNLOADING
            );

            long speed = progress.bytesPerSecond();
            // Should be approximately 512 KB/s (1MB / 2s)
            assertTrue(speed > 400 * 1024 && speed < 600 * 1024,
                    "Speed should be around 512 KB/s, was: " + speed);
        }

        @Test
        void testBytesPerSecondZeroDuration() {
            DownloadProgress progress = new DownloadProgress(
                    "test/repo", "file.bin", 0, 1,
                    1024, 1024,
                    1024, 1024,
                    Instant.now(), DownloadProgress.Status.DOWNLOADING
            );

            assertEquals(0, progress.bytesPerSecond());
        }

        @Test
        void testEstimatedTimeRemaining() {
            // 500KB downloaded in 1 second, 500KB remaining
            DownloadProgress progress = new DownloadProgress(
                    "test/repo", "file.bin", 0, 1,
                    512 * 1024, 1024 * 1024,
                    512 * 1024, 1024 * 1024,
                    Instant.now().minusSeconds(1), DownloadProgress.Status.DOWNLOADING
            );

            Duration eta = progress.estimatedTimeRemaining();
            // Should be approximately 1 second
            assertTrue(eta.toSeconds() >= 0 && eta.toSeconds() <= 2,
                    "ETA should be around 1 second, was: " + eta.toSeconds());
        }
    }
}
