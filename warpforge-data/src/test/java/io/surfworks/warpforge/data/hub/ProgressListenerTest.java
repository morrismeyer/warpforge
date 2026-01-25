package io.surfworks.warpforge.data.hub;

import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ProgressListenerTest {

    @Test
    void testNoneListener() {
        // Should not throw
        ProgressListener.NONE.onProgress(createTestProgress());
    }

    @Test
    void testCustomListener() {
        List<DownloadProgress> received = new ArrayList<>();
        ProgressListener listener = received::add;

        DownloadProgress progress = createTestProgress();
        listener.onProgress(progress);

        assertEquals(1, received.size());
        assertEquals(progress, received.get(0));
    }

    @Test
    void testAndThen() {
        AtomicInteger count1 = new AtomicInteger(0);
        AtomicInteger count2 = new AtomicInteger(0);

        ProgressListener listener1 = p -> count1.incrementAndGet();
        ProgressListener listener2 = p -> count2.incrementAndGet();
        ProgressListener combined = listener1.andThen(listener2);

        combined.onProgress(createTestProgress());
        combined.onProgress(createTestProgress());

        assertEquals(2, count1.get());
        assertEquals(2, count2.get());
    }

    @Test
    void testThrottled() throws InterruptedException {
        AtomicInteger callCount = new AtomicInteger(0);
        ProgressListener base = p -> callCount.incrementAndGet();
        ProgressListener throttled = base.throttled(50); // 50ms throttle

        // Send many updates quickly
        for (int i = 0; i < 100; i++) {
            throttled.onProgress(createDownloadingProgress());
        }

        // Should have been throttled to far fewer calls
        assertTrue(callCount.get() < 20, "Expected throttling, got " + callCount.get() + " calls");

        // Non-DOWNLOADING status should always pass through
        callCount.set(0);
        for (int i = 0; i < 10; i++) {
            throttled.onProgress(createCompleteProgress());
        }
        assertEquals(10, callCount.get(), "Non-DOWNLOADING should not be throttled");
    }

    @Test
    void testThrottledPassesThroughNonDownloading() {
        AtomicInteger callCount = new AtomicInteger(0);
        ProgressListener base = p -> callCount.incrementAndGet();
        ProgressListener throttled = base.throttled(1000); // 1 second throttle

        // STARTING, FILE_COMPLETE, COMPLETE, SKIPPED, FAILED should all pass through
        throttled.onProgress(createProgressWithStatus(DownloadProgress.Status.STARTING));
        throttled.onProgress(createProgressWithStatus(DownloadProgress.Status.FILE_COMPLETE));
        throttled.onProgress(createProgressWithStatus(DownloadProgress.Status.COMPLETE));
        throttled.onProgress(createProgressWithStatus(DownloadProgress.Status.SKIPPED));
        throttled.onProgress(createProgressWithStatus(DownloadProgress.Status.FAILED));

        assertEquals(5, callCount.get());
    }

    @Test
    void testCollectingListener() {
        // A practical example: collecting all progress for testing
        List<DownloadProgress> collected = new ArrayList<>();
        ProgressListener collector = collected::add;

        // Simulate a download sequence
        collector.onProgress(DownloadProgress.starting("test/repo", 2, 1000));

        DownloadProgress p = DownloadProgress.starting("test/repo", 2, 1000)
                .startFile("file1.bin", 0, 500);
        collector.onProgress(p);

        p = p.withProgress(250, 250);
        collector.onProgress(p);

        p = p.withProgress(500, 500).fileComplete();
        collector.onProgress(p);

        p = p.startFile("file2.bin", 1, 500);
        collector.onProgress(p);

        p = p.withProgress(500, 1000).complete();
        collector.onProgress(p);

        assertEquals(6, collected.size());
        assertEquals(DownloadProgress.Status.STARTING, collected.get(0).status());
        assertEquals(DownloadProgress.Status.DOWNLOADING, collected.get(1).status());
        assertEquals(DownloadProgress.Status.DOWNLOADING, collected.get(2).status());
        assertEquals(DownloadProgress.Status.FILE_COMPLETE, collected.get(3).status());
        assertEquals(DownloadProgress.Status.DOWNLOADING, collected.get(4).status());
        assertEquals(DownloadProgress.Status.COMPLETE, collected.get(5).status());
    }

    // Helper methods

    private DownloadProgress createTestProgress() {
        return new DownloadProgress(
                "test/repo", "file.bin", 0, 1,
                0, 1000, 0, 1000,
                Instant.now(), DownloadProgress.Status.STARTING
        );
    }

    private DownloadProgress createDownloadingProgress() {
        return new DownloadProgress(
                "test/repo", "file.bin", 0, 1,
                500, 1000, 500, 1000,
                Instant.now(), DownloadProgress.Status.DOWNLOADING
        );
    }

    private DownloadProgress createCompleteProgress() {
        return new DownloadProgress(
                "test/repo", "file.bin", 0, 1,
                1000, 1000, 1000, 1000,
                Instant.now(), DownloadProgress.Status.COMPLETE
        );
    }

    private DownloadProgress createProgressWithStatus(DownloadProgress.Status status) {
        return new DownloadProgress(
                "test/repo", "file.bin", 0, 1,
                500, 1000, 500, 1000,
                Instant.now(), status
        );
    }
}
