package io.surfworks.warpforge.data.hub;

/**
 * Listener interface for receiving download progress updates.
 *
 * <p>Implementations can use this to display progress in a UI, log progress,
 * or perform other actions during downloads.
 *
 * <p>Example console progress display:
 * <pre>{@code
 * ProgressListener listener = progress -> {
 *     System.out.print("\r" + progress.toProgressString());
 *     if (progress.status() == DownloadProgress.Status.FILE_COMPLETE) {
 *         System.out.println();
 *     }
 * };
 *
 * WarpForge.configure()
 *     .progressListener(listener)
 *     .apply();
 * }</pre>
 */
@FunctionalInterface
public interface ProgressListener {

    /**
     * Called when download progress is updated.
     *
     * <p>This method may be called frequently during downloads (potentially
     * thousands of times per file), so implementations should be efficient.
     *
     * @param progress The current progress state
     */
    void onProgress(DownloadProgress progress);

    /**
     * A no-op listener that ignores all progress updates.
     */
    ProgressListener NONE = progress -> {};

    /**
     * A listener that prints progress to stdout with carriage return updates.
     */
    ProgressListener CONSOLE = new ProgressListener() {
        @Override
        public void onProgress(DownloadProgress progress) {
            switch (progress.status()) {
                case STARTING -> {
                    System.out.println("Downloading " + progress.repoId() +
                        " (" + progress.totalFiles() + " files, " +
                        DownloadProgress.formatBytes(progress.totalSize()) + ")");
                }
                case DOWNLOADING -> {
                    // Throttle updates to avoid excessive printing
                    if (shouldUpdate(progress)) {
                        System.out.print("\r" + progress.toProgressString() + "          ");
                    }
                }
                case FILE_COMPLETE -> {
                    System.out.print("\r" + progress.toProgressString() + "          ");
                    System.out.println();
                }
                case SKIPPED -> {
                    System.out.println(progress.toProgressString());
                }
                case COMPLETE -> {
                    System.out.println("Download complete: " + progress.repoId());
                }
                case FAILED -> {
                    System.out.println("\nDownload failed: " + progress.currentFile());
                }
            }
        }

        private long lastUpdateTime = 0;

        private boolean shouldUpdate(DownloadProgress progress) {
            long now = System.currentTimeMillis();
            if (now - lastUpdateTime >= 100) { // Update at most 10 times per second
                lastUpdateTime = now;
                return true;
            }
            return false;
        }
    };

    /**
     * A listener that prints minimal progress (file starts/completions only).
     */
    ProgressListener MINIMAL = progress -> {
        switch (progress.status()) {
            case STARTING -> System.out.println("Downloading " + progress.repoId() + "...");
            case FILE_COMPLETE -> System.out.println("  Downloaded: " + progress.currentFile());
            case SKIPPED -> System.out.println("  Cached: " + progress.currentFile());
            case COMPLETE -> System.out.println("Download complete.");
            case FAILED -> System.out.println("Download failed: " + progress.currentFile());
            default -> {}
        }
    };

    /**
     * Combine this listener with another, calling both on each update.
     */
    default ProgressListener andThen(ProgressListener other) {
        return progress -> {
            this.onProgress(progress);
            other.onProgress(progress);
        };
    }

    /**
     * Create a listener that only receives updates at the specified interval.
     *
     * @param intervalMs Minimum milliseconds between updates
     * @return A throttled listener
     */
    default ProgressListener throttled(long intervalMs) {
        return new ProgressListener() {
            private long lastUpdate = 0;

            @Override
            public void onProgress(DownloadProgress progress) {
                // Always pass through non-DOWNLOADING status
                if (progress.status() != DownloadProgress.Status.DOWNLOADING) {
                    ProgressListener.this.onProgress(progress);
                    return;
                }

                long now = System.currentTimeMillis();
                if (now - lastUpdate >= intervalMs) {
                    lastUpdate = now;
                    ProgressListener.this.onProgress(progress);
                }
            }
        };
    }
}
