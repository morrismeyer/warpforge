package io.surfworks.warpforge.data.hub;

import java.time.Duration;
import java.time.Instant;

/**
 * Progress information for a download operation.
 *
 * <p>Provides detailed information about download progress including:
 * <ul>
 *   <li>Current file being downloaded</li>
 *   <li>Bytes transferred and total size</li>
 *   <li>Download speed and ETA</li>
 *   <li>Overall progress across multiple files</li>
 * </ul>
 *
 * @param repoId        Repository being downloaded (e.g., "meta-llama/Llama-3.1-8B")
 * @param currentFile   Name of the file currently being downloaded
 * @param fileIndex     Index of current file (0-based)
 * @param totalFiles    Total number of files to download
 * @param bytesRead     Bytes downloaded for current file
 * @param fileSize      Total size of current file in bytes (-1 if unknown)
 * @param totalBytesRead Total bytes downloaded across all files
 * @param totalSize     Total size of all files in bytes (-1 if unknown)
 * @param startTime     When the download started
 * @param status        Current download status
 */
public record DownloadProgress(
        String repoId,
        String currentFile,
        int fileIndex,
        int totalFiles,
        long bytesRead,
        long fileSize,
        long totalBytesRead,
        long totalSize,
        Instant startTime,
        Status status
) {

    /**
     * Download status.
     */
    public enum Status {
        /** Download is starting */
        STARTING,
        /** Download is in progress */
        DOWNLOADING,
        /** Current file completed */
        FILE_COMPLETE,
        /** All files completed */
        COMPLETE,
        /** Download failed */
        FAILED,
        /** File was skipped (already cached) */
        SKIPPED
    }

    /**
     * Progress percentage for current file (0.0 to 1.0).
     * Returns -1 if file size is unknown.
     */
    public double fileProgress() {
        if (fileSize <= 0) return -1;
        return (double) bytesRead / fileSize;
    }

    /**
     * Progress percentage for overall download (0.0 to 1.0).
     * Returns -1 if total size is unknown.
     */
    public double totalProgress() {
        if (totalSize <= 0) return -1;
        return (double) totalBytesRead / totalSize;
    }

    /**
     * Download speed in bytes per second.
     */
    public long bytesPerSecond() {
        Duration elapsed = Duration.between(startTime, Instant.now());
        long seconds = elapsed.toSeconds();
        if (seconds <= 0) return 0;
        return totalBytesRead / seconds;
    }

    /**
     * Estimated time remaining based on current speed.
     * Returns Duration.ZERO if cannot be calculated.
     */
    public Duration estimatedTimeRemaining() {
        long speed = bytesPerSecond();
        if (speed <= 0 || totalSize <= 0) return Duration.ZERO;

        long remaining = totalSize - totalBytesRead;
        return Duration.ofSeconds(remaining / speed);
    }

    /**
     * Human-readable progress string.
     */
    public String toProgressString() {
        StringBuilder sb = new StringBuilder();

        // File progress
        sb.append(String.format("[%d/%d] %s: ", fileIndex + 1, totalFiles, currentFile));

        if (status == Status.SKIPPED) {
            sb.append("(cached)");
            return sb.toString();
        }

        if (status == Status.FILE_COMPLETE || status == Status.COMPLETE) {
            sb.append(formatBytes(bytesRead));
            sb.append(" - complete");
            return sb.toString();
        }

        // Bytes progress
        sb.append(formatBytes(bytesRead));
        if (fileSize > 0) {
            sb.append(" / ").append(formatBytes(fileSize));
            sb.append(String.format(" (%.1f%%)", fileProgress() * 100));
        }

        // Speed
        long speed = bytesPerSecond();
        if (speed > 0) {
            sb.append(" - ").append(formatBytes(speed)).append("/s");

            // ETA
            Duration eta = estimatedTimeRemaining();
            if (!eta.isZero()) {
                sb.append(" - ETA: ").append(formatDuration(eta));
            }
        }

        return sb.toString();
    }

    /**
     * Format bytes as human-readable string.
     */
    public static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    /**
     * Format duration as human-readable string.
     */
    public static String formatDuration(Duration duration) {
        long seconds = duration.toSeconds();
        if (seconds < 60) return seconds + "s";
        if (seconds < 3600) return String.format("%dm %ds", seconds / 60, seconds % 60);
        return String.format("%dh %dm", seconds / 3600, (seconds % 3600) / 60);
    }

    /**
     * Create a STARTING progress event.
     */
    public static DownloadProgress starting(String repoId, int totalFiles, long totalSize) {
        return new DownloadProgress(
                repoId, "", 0, totalFiles,
                0, 0, 0, totalSize,
                Instant.now(), Status.STARTING
        );
    }

    /**
     * Create a progress update for current file.
     */
    public DownloadProgress withProgress(long bytesRead, long totalBytesRead) {
        return new DownloadProgress(
                repoId, currentFile, fileIndex, totalFiles,
                bytesRead, fileSize, totalBytesRead, totalSize,
                startTime, Status.DOWNLOADING
        );
    }

    /**
     * Create a FILE_COMPLETE progress event.
     */
    public DownloadProgress fileComplete() {
        return new DownloadProgress(
                repoId, currentFile, fileIndex, totalFiles,
                fileSize > 0 ? fileSize : bytesRead, fileSize,
                totalBytesRead, totalSize,
                startTime, Status.FILE_COMPLETE
        );
    }

    /**
     * Create a SKIPPED progress event.
     */
    public DownloadProgress skipped() {
        return new DownloadProgress(
                repoId, currentFile, fileIndex, totalFiles,
                fileSize, fileSize,
                totalBytesRead + fileSize, totalSize,
                startTime, Status.SKIPPED
        );
    }

    /**
     * Start a new file in the download.
     */
    public DownloadProgress startFile(String fileName, int index, long size) {
        return new DownloadProgress(
                repoId, fileName, index, totalFiles,
                0, size, totalBytesRead, totalSize,
                startTime, Status.DOWNLOADING
        );
    }

    /**
     * Create a COMPLETE progress event.
     */
    public DownloadProgress complete() {
        return new DownloadProgress(
                repoId, currentFile, totalFiles - 1, totalFiles,
                bytesRead, fileSize, totalSize, totalSize,
                startTime, Status.COMPLETE
        );
    }

    /**
     * Create a FAILED progress event.
     */
    public DownloadProgress failed() {
        return new DownloadProgress(
                repoId, currentFile, fileIndex, totalFiles,
                bytesRead, fileSize, totalBytesRead, totalSize,
                startTime, Status.FAILED
        );
    }
}
