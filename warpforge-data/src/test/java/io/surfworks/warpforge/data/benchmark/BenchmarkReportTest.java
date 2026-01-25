package io.surfworks.warpforge.data.benchmark;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BenchmarkReportTest {

    @TempDir
    Path tempDir;

    private BenchmarkResult createSuccessResult(String name, double meanLatencyMs) {
        long[] latencies = new long[10];
        for (int i = 0; i < 10; i++) {
            latencies[i] = (long) (meanLatencyMs * 1_000_000 * (0.9 + Math.random() * 0.2));
        }
        return new BenchmarkResult(
                name, "test-model", "cpu",
                Instant.now().minusSeconds(10), Instant.now(),
                5, 10, latencies,
                List.of(), null, BenchmarkResult.Status.SUCCESS, null
        );
    }

    private BenchmarkResult createErrorResult(String name) {
        return BenchmarkResult.error(name, "test-model", "cpu",
                Instant.now(), "Test error message");
    }

    @Nested
    class BuilderTests {

        @Test
        void testBasicBuilder() {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .build();

            assertNotNull(report);
        }

        @Test
        void testBuilderWithResults() {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .result(createSuccessResult("bench2", 20.0))
                    .build();

            assertNotNull(report);
        }

        @Test
        void testBuilderWithMetadata() {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .metadata("backend", "cpu")
                    .metadata("batch_size", "32")
                    .build();

            assertNotNull(report);
        }
    }

    @Nested
    class JsonOutputTests {

        @Test
        void testToJson() {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .metadata("version", "1.0")
                    .build();

            String json = report.toJson();

            assertNotNull(json);
            assertTrue(json.contains("\"title\""));
            assertTrue(json.contains("Test Report"));
            assertTrue(json.contains("bench1"));
            assertTrue(json.contains("version"));
        }

        @Test
        void testJsonContainsLatencyStats() {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .build();

            String json = report.toJson();

            assertTrue(json.contains("mean_ms"));
            assertTrue(json.contains("median_ms"));
            assertTrue(json.contains("p95_ms"));
            assertTrue(json.contains("p99_ms"));
        }

        @Test
        void testWriteJson() throws IOException {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .build();

            Path jsonPath = tempDir.resolve("report.json");
            report.writeJson(jsonPath);

            assertTrue(Files.exists(jsonPath));
            String content = Files.readString(jsonPath);
            assertTrue(content.contains("Test Report"));
        }

        @Test
        void testWriteJsonToWriter() throws IOException {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .build();

            StringWriter writer = new StringWriter();
            report.writeJson(writer);

            String content = writer.toString();
            assertTrue(content.contains("Test Report"));
        }
    }

    @Nested
    class HtmlOutputTests {

        @Test
        void testToHtml() {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .build();

            String html = report.toHtml();

            assertNotNull(html);
            assertTrue(html.contains("<!DOCTYPE html>"));
            assertTrue(html.contains("Test Report"));
            assertTrue(html.contains("bench1"));
        }

        @Test
        void testHtmlContainsResultsTable() {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .result(createSuccessResult("bench2", 20.0))
                    .build();

            String html = report.toHtml();

            assertTrue(html.contains("<table"));
            assertTrue(html.contains("bench1"));
            assertTrue(html.contains("bench2"));
        }

        @Test
        void testHtmlContainsSummaryCards() {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .result(createErrorResult("bench2"))
                    .build();

            String html = report.toHtml();

            assertTrue(html.contains("Total"));
            assertTrue(html.contains("Passed"));
            assertTrue(html.contains("Errors"));
        }

        @Test
        void testHtmlContainsLatencyChart() {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .result(createSuccessResult("bench2", 20.0))
                    .build();

            String html = report.toHtml();

            assertTrue(html.contains("Latency Distribution"));
            assertTrue(html.contains("chart-bar"));
        }

        @Test
        void testWriteHtml() throws IOException {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .build();

            Path htmlPath = tempDir.resolve("report.html");
            report.writeHtml(htmlPath);

            assertTrue(Files.exists(htmlPath));
            String content = Files.readString(htmlPath);
            assertTrue(content.contains("<!DOCTYPE html>"));
        }
    }

    @Nested
    class StatusTests {

        @Test
        void testMixedStatusResults() {
            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("success1", 10.0))
                    .result(createSuccessResult("success2", 15.0))
                    .result(createErrorResult("error1"))
                    .result(BenchmarkResult.skipped("skipped1", "model", "cpu", "not available"))
                    .build();

            String json = report.toJson();

            assertTrue(json.contains("\"passed\": 2"));
            assertTrue(json.contains("\"errors\": 1"));
            assertTrue(json.contains("\"skipped\": 1"));
        }
    }

    @Nested
    class MemoryProfileTests {

        @Test
        void testReportWithMemorySnapshot() {
            MemoryProfiler.MemorySnapshot snapshot = new MemoryProfiler.MemorySnapshot(
                    1000, 2000, 3000, 2500,
                    4000, 10000,
                    100, 200, 300, 500,
                    1000, 100, 10, 1000000
            );

            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 10.0))
                    .memorySnapshot(snapshot)
                    .build();

            String json = report.toJson();
            assertTrue(json.contains("memory_profiles"));
            assertTrue(json.contains("peak_heap_bytes"));

            String html = report.toHtml();
            assertTrue(html.contains("Memory Profiles"));
        }
    }

    @Nested
    class BaselineComparisonTests {

        @Test
        void testReportWithBaseline() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("bench1/latency_ms", 10.0, 20,
                            PerformanceBaseline.Direction.LOWER_IS_BETTER);

            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Test Report")
                    .result(createSuccessResult("bench1", 12.0))
                    .baseline(baseline)
                    .build();

            PerformanceBaseline.ComparisonResult comparison = report.compareWithBaseline();
            assertNotNull(comparison);

            String html = report.toHtml();
            assertTrue(html.contains("Baseline Comparison"));
        }
    }

    @Nested
    class EscapingTests {

        @Test
        void testHtmlEscaping() {
            BenchmarkResult result = new BenchmarkResult(
                    "bench<script>alert('xss')</script>", "model", "cpu",
                    Instant.now().minusSeconds(1), Instant.now(),
                    0, 1, new long[]{1000000},
                    List.of(), null, BenchmarkResult.Status.SUCCESS, null
            );

            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Title with <html> tags")
                    .result(result)
                    .build();

            String html = report.toHtml();

            assertFalse(html.contains("<script>alert"));
            assertTrue(html.contains("&lt;script&gt;"));
        }

        @Test
        void testJsonEscaping() {
            BenchmarkResult result = new BenchmarkResult(
                    "bench\"with\"quotes", "model", "cpu",
                    Instant.now().minusSeconds(1), Instant.now(),
                    0, 1, new long[]{1000000},
                    List.of(), null, BenchmarkResult.Status.SUCCESS, null
            );

            BenchmarkReport report = BenchmarkReport.builder()
                    .title("Title with \"quotes\"")
                    .result(result)
                    .build();

            String json = report.toJson();

            assertTrue(json.contains("\\\"quotes\\\""));
        }
    }
}
