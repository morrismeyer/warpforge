package io.surfworks.warpforge.data.benchmark;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Generates benchmark reports in HTML and JSON formats.
 *
 * <p>Features:
 * <ul>
 *   <li>JSON export for CI integration</li>
 *   <li>HTML report with tables and visualizations</li>
 *   <li>Comparison with baselines</li>
 *   <li>Historical tracking support</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * List<BenchmarkResult> results = runner.runAll(benchmarks, config);
 *
 * BenchmarkReport report = BenchmarkReport.builder()
 *     .title("Model Benchmark Report")
 *     .results(results)
 *     .baseline(previousBaseline)
 *     .build();
 *
 * report.writeHtml(Path.of("report.html"));
 * report.writeJson(Path.of("report.json"));
 * }</pre>
 */
public final class BenchmarkReport {

    private final String title;
    private final List<BenchmarkResult> results;
    private final PerformanceBaseline baseline;
    private final Instant generatedAt;
    private final Map<String, String> metadata;
    private final List<MemoryProfiler.MemorySnapshot> memorySnapshots;

    private BenchmarkReport(Builder builder) {
        this.title = builder.title;
        this.results = List.copyOf(builder.results);
        this.baseline = builder.baseline;
        this.generatedAt = Instant.now();
        this.metadata = Map.copyOf(builder.metadata);
        this.memorySnapshots = List.copyOf(builder.memorySnapshots);
    }

    /**
     * Create a new report builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Write the report as JSON.
     */
    public void writeJson(Path path) throws IOException {
        try (Writer writer = Files.newBufferedWriter(path)) {
            writeJson(writer);
        }
    }

    /**
     * Write the report as JSON to a writer.
     */
    public void writeJson(Writer writer) throws IOException {
        writer.write(toJson());
    }

    /**
     * Generate JSON string.
     */
    public String toJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"title\": ").append(jsonString(title)).append(",\n");
        sb.append("  \"generated_at\": ").append(jsonString(generatedAt.toString())).append(",\n");

        // Metadata
        sb.append("  \"metadata\": {\n");
        List<Map.Entry<String, String>> metaEntries = new ArrayList<>(metadata.entrySet());
        for (int i = 0; i < metaEntries.size(); i++) {
            Map.Entry<String, String> entry = metaEntries.get(i);
            sb.append("    ").append(jsonString(entry.getKey())).append(": ")
                    .append(jsonString(entry.getValue()));
            if (i < metaEntries.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("  },\n");

        // Summary
        sb.append("  \"summary\": {\n");
        sb.append("    \"total_benchmarks\": ").append(results.size()).append(",\n");
        sb.append("    \"passed\": ").append(countByStatus(BenchmarkResult.Status.SUCCESS)).append(",\n");
        sb.append("    \"failed\": ").append(countByStatus(BenchmarkResult.Status.VALIDATION_FAILED)).append(",\n");
        sb.append("    \"errors\": ").append(countByStatus(BenchmarkResult.Status.ERROR)).append(",\n");
        sb.append("    \"skipped\": ").append(countByStatus(BenchmarkResult.Status.SKIPPED)).append("\n");
        sb.append("  },\n");

        // Results
        sb.append("  \"results\": [\n");
        for (int i = 0; i < results.size(); i++) {
            BenchmarkResult result = results.get(i);
            sb.append("    {\n");
            sb.append("      \"name\": ").append(jsonString(result.benchmarkName())).append(",\n");
            sb.append("      \"model_id\": ").append(jsonString(result.modelId())).append(",\n");
            sb.append("      \"backend\": ").append(jsonString(result.backend())).append(",\n");
            sb.append("      \"status\": ").append(jsonString(result.status().name())).append(",\n");
            sb.append("      \"warmup_iterations\": ").append(result.warmupIterations()).append(",\n");
            sb.append("      \"measurement_iterations\": ").append(result.measurementIterations()).append(",\n");
            sb.append("      \"latency\": {\n");
            sb.append("        \"mean_ms\": ").append(formatDouble(result.meanLatencyMs())).append(",\n");
            sb.append("        \"median_ms\": ").append(formatDouble(result.medianLatencyNanos() / 1e6)).append(",\n");
            sb.append("        \"min_ms\": ").append(formatDouble(result.minLatencyNanos() / 1e6)).append(",\n");
            sb.append("        \"max_ms\": ").append(formatDouble(result.maxLatencyNanos() / 1e6)).append(",\n");
            sb.append("        \"std_ms\": ").append(formatDouble(result.stdLatencyNanos() / 1e6)).append(",\n");
            sb.append("        \"p95_ms\": ").append(formatDouble(result.p95LatencyNanos() / 1e6)).append(",\n");
            sb.append("        \"p99_ms\": ").append(formatDouble(result.p99LatencyNanos() / 1e6)).append("\n");
            sb.append("      },\n");
            sb.append("      \"throughput_per_sec\": ").append(formatDouble(result.throughputPerSecond())).append(",\n");
            sb.append("      \"validation_passed\": ").append(result.allOutputsValid());

            // Memory stats
            if (result.memoryStats() != null) {
                sb.append(",\n");
                sb.append("      \"memory\": {\n");
                sb.append("        \"peak_bytes\": ").append(result.memoryStats().peakUsageBytes()).append(",\n");
                sb.append("        \"allocated_bytes\": ").append(result.memoryStats().allocatedBytes()).append("\n");
                sb.append("      }");
            }

            sb.append("\n    }");
            if (i < results.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("  ],\n");

        // Baseline comparison
        if (baseline != null) {
            sb.append("  \"baseline_comparison\": ");
            PerformanceBaseline.ComparisonResult comparison = compareWithBaseline();
            sb.append(comparison.toJson());
            sb.append(",\n");
        }

        // Memory snapshots
        if (!memorySnapshots.isEmpty()) {
            sb.append("  \"memory_profiles\": [\n");
            for (int i = 0; i < memorySnapshots.size(); i++) {
                MemoryProfiler.MemorySnapshot snapshot = memorySnapshots.get(i);
                sb.append("    ").append(toJsonObject(snapshot.toMap()));
                if (i < memorySnapshots.size() - 1) sb.append(",");
                sb.append("\n");
            }
            sb.append("  ],\n");
        }

        // Remove trailing comma and close
        String json = sb.toString();
        if (json.endsWith(",\n")) {
            json = json.substring(0, json.length() - 2) + "\n";
        }
        return json + "}\n";
    }

    /**
     * Write the report as HTML.
     */
    public void writeHtml(Path path) throws IOException {
        try (Writer writer = Files.newBufferedWriter(path)) {
            writeHtml(writer);
        }
    }

    /**
     * Write the report as HTML to a writer.
     */
    public void writeHtml(Writer writer) throws IOException {
        writer.write(toHtml());
    }

    /**
     * Generate HTML string.
     */
    public String toHtml() {
        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
        sb.append("  <meta charset=\"UTF-8\">\n");
        sb.append("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        sb.append("  <title>").append(escapeHtml(title)).append("</title>\n");
        sb.append(CSS);
        sb.append("</head>\n<body>\n");

        // Header
        sb.append("<div class=\"container\">\n");
        sb.append("  <h1>").append(escapeHtml(title)).append("</h1>\n");
        sb.append("  <p class=\"timestamp\">Generated: ").append(formatTimestamp(generatedAt)).append("</p>\n");

        // Summary cards
        sb.append("  <div class=\"summary-cards\">\n");
        appendSummaryCard(sb, "Total", String.valueOf(results.size()), "card");
        appendSummaryCard(sb, "Passed", String.valueOf(countByStatus(BenchmarkResult.Status.SUCCESS)), "card success");
        appendSummaryCard(sb, "Failed", String.valueOf(countByStatus(BenchmarkResult.Status.VALIDATION_FAILED)), "card warning");
        appendSummaryCard(sb, "Errors", String.valueOf(countByStatus(BenchmarkResult.Status.ERROR)), "card error");
        sb.append("  </div>\n");

        // Metadata
        if (!metadata.isEmpty()) {
            sb.append("  <h2>Configuration</h2>\n");
            sb.append("  <table class=\"metadata-table\">\n");
            for (Map.Entry<String, String> entry : metadata.entrySet()) {
                sb.append("    <tr><td>").append(escapeHtml(entry.getKey()))
                        .append("</td><td>").append(escapeHtml(entry.getValue())).append("</td></tr>\n");
            }
            sb.append("  </table>\n");
        }

        // Results table
        sb.append("  <h2>Results</h2>\n");
        sb.append("  <table class=\"results-table\">\n");
        sb.append("    <thead>\n");
        sb.append("      <tr>\n");
        sb.append("        <th>Benchmark</th>\n");
        sb.append("        <th>Model</th>\n");
        sb.append("        <th>Backend</th>\n");
        sb.append("        <th>Status</th>\n");
        sb.append("        <th>Mean (ms)</th>\n");
        sb.append("        <th>P95 (ms)</th>\n");
        sb.append("        <th>P99 (ms)</th>\n");
        sb.append("        <th>Throughput</th>\n");
        sb.append("        <th>Valid</th>\n");
        sb.append("      </tr>\n");
        sb.append("    </thead>\n");
        sb.append("    <tbody>\n");

        for (BenchmarkResult result : results) {
            String statusClass = switch (result.status()) {
                case SUCCESS -> "status-success";
                case VALIDATION_FAILED -> "status-warning";
                case ERROR -> "status-error";
                case SKIPPED -> "status-skipped";
            };

            sb.append("      <tr>\n");
            sb.append("        <td>").append(escapeHtml(result.benchmarkName())).append("</td>\n");
            sb.append("        <td>").append(escapeHtml(result.modelId())).append("</td>\n");
            sb.append("        <td>").append(escapeHtml(result.backend())).append("</td>\n");
            sb.append("        <td class=\"").append(statusClass).append("\">")
                    .append(result.status()).append("</td>\n");
            sb.append("        <td>").append(formatDouble(result.meanLatencyMs())).append("</td>\n");
            sb.append("        <td>").append(formatDouble(result.p95LatencyNanos() / 1e6)).append("</td>\n");
            sb.append("        <td>").append(formatDouble(result.p99LatencyNanos() / 1e6)).append("</td>\n");
            sb.append("        <td>").append(formatDouble(result.throughputPerSecond())).append("/s</td>\n");
            sb.append("        <td>").append(result.allOutputsValid() ? "Yes" : "No").append("</td>\n");
            sb.append("      </tr>\n");
        }

        sb.append("    </tbody>\n");
        sb.append("  </table>\n");

        // Baseline comparison
        if (baseline != null) {
            sb.append("  <h2>Baseline Comparison</h2>\n");
            appendBaselineComparison(sb);
        }

        // Memory profiles
        if (!memorySnapshots.isEmpty()) {
            sb.append("  <h2>Memory Profiles</h2>\n");
            for (int i = 0; i < memorySnapshots.size(); i++) {
                MemoryProfiler.MemorySnapshot snapshot = memorySnapshots.get(i);
                sb.append("  <div class=\"memory-profile\">\n");
                sb.append("    <h3>Profile ").append(i + 1).append("</h3>\n");
                sb.append("    <table class=\"metadata-table\">\n");
                sb.append("      <tr><td>Peak Heap</td><td>").append(formatBytes(snapshot.peakHeapUsed())).append("</td></tr>\n");
                sb.append("      <tr><td>Avg Heap</td><td>").append(formatBytes(snapshot.avgHeapUsed())).append("</td></tr>\n");
                sb.append("      <tr><td>Heap Utilization</td><td>").append(formatDouble(snapshot.heapUtilization())).append("%</td></tr>\n");
                sb.append("      <tr><td>Total Allocated</td><td>").append(formatBytes(snapshot.totalAllocated())).append("</td></tr>\n");
                sb.append("      <tr><td>Allocation Rate</td><td>").append(formatBytes((long) snapshot.allocationRateBytesPerSec())).append("/s</td></tr>\n");
                sb.append("    </table>\n");
                sb.append("  </div>\n");
            }
        }

        // Latency chart (simple bar chart using CSS)
        if (!results.isEmpty()) {
            sb.append("  <h2>Latency Distribution</h2>\n");
            sb.append("  <div class=\"chart\">\n");
            double maxLatency = results.stream().mapToDouble(BenchmarkResult::meanLatencyMs).max().orElse(1);
            for (BenchmarkResult result : results) {
                double pct = result.meanLatencyMs() / maxLatency * 100;
                sb.append("    <div class=\"chart-row\">\n");
                sb.append("      <div class=\"chart-label\">").append(escapeHtml(truncate(result.benchmarkName(), 20))).append("</div>\n");
                sb.append("      <div class=\"chart-bar-container\">\n");
                sb.append("        <div class=\"chart-bar\" style=\"width: ").append(formatDouble(pct)).append("%;\"></div>\n");
                sb.append("      </div>\n");
                sb.append("      <div class=\"chart-value\">").append(formatDouble(result.meanLatencyMs())).append(" ms</div>\n");
                sb.append("    </div>\n");
            }
            sb.append("  </div>\n");
        }

        sb.append("</div>\n");
        sb.append("</body>\n</html>\n");
        return sb.toString();
    }

    /**
     * Compare results with baseline.
     */
    public PerformanceBaseline.ComparisonResult compareWithBaseline() {
        if (baseline == null) {
            return new PerformanceBaseline.ComparisonResult(
                    List.of(), List.of(), List.of(), List.of());
        }

        Map<String, Double> currentMetrics = new HashMap<>();
        for (BenchmarkResult result : results) {
            String metricName = result.benchmarkName() + "/latency_ms";
            currentMetrics.put(metricName, result.meanLatencyMs());
        }

        return baseline.compare(currentMetrics);
    }

    private void appendBaselineComparison(StringBuilder sb) {
        PerformanceBaseline.ComparisonResult comparison = compareWithBaseline();

        // Count passed (not regression) vs total
        int total = comparison.totalCompared();
        int failed = comparison.regressions().size();
        int passed = total - failed;

        sb.append("  <div class=\"baseline-summary\">\n");
        sb.append("    <p>").append(passed).append("/").append(total)
                .append(" metrics within tolerance</p>\n");
        sb.append("  </div>\n");

        sb.append("  <table class=\"results-table\">\n");
        sb.append("    <thead><tr><th>Metric</th><th>Current</th><th>Baseline</th><th>Change</th><th>Status</th></tr></thead>\n");
        sb.append("    <tbody>\n");

        // Combine all comparisons for display
        List<PerformanceBaseline.MetricComparison> allComparisons = new ArrayList<>();
        allComparisons.addAll(comparison.regressions());
        allComparisons.addAll(comparison.improvements());
        allComparisons.addAll(comparison.unchanged());

        for (PerformanceBaseline.MetricComparison mc : allComparisons) {
            String statusClass = mc.isRegression() ? "status-error" :
                    (mc.isImprovement() ? "status-success" : "status-skipped");
            String statusText = mc.isRegression() ? "REGRESSION" :
                    (mc.isImprovement() ? "IMPROVED" : "OK");
            sb.append("      <tr>\n");
            sb.append("        <td>").append(escapeHtml(mc.name())).append("</td>\n");
            sb.append("        <td>").append(formatDouble(mc.currentValue())).append("</td>\n");
            sb.append("        <td>").append(formatDouble(mc.baselineValue())).append("</td>\n");
            sb.append("        <td>").append(mc.percentChange() >= 0 ? "+" : "").append(formatDouble(mc.percentChange())).append("%</td>\n");
            sb.append("        <td class=\"").append(statusClass).append("\">").append(statusText).append("</td>\n");
            sb.append("      </tr>\n");
        }
        sb.append("    </tbody>\n");
        sb.append("  </table>\n");
    }

    private void appendSummaryCard(StringBuilder sb, String label, String value, String cssClass) {
        sb.append("    <div class=\"").append(cssClass).append("\">\n");
        sb.append("      <div class=\"card-value\">").append(value).append("</div>\n");
        sb.append("      <div class=\"card-label\">").append(label).append("</div>\n");
        sb.append("    </div>\n");
    }

    private long countByStatus(BenchmarkResult.Status status) {
        return results.stream().filter(r -> r.status() == status).count();
    }

    private static String jsonString(String s) {
        if (s == null) return "null";
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"";
    }

    private static String formatDouble(double d) {
        if (Double.isNaN(d) || Double.isInfinite(d)) return "0";
        return String.format("%.2f", d);
    }

    private static String formatTimestamp(Instant instant) {
        return DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss z")
                .withZone(ZoneId.systemDefault())
                .format(instant);
    }

    private static String formatBytes(long bytes) {
        if (bytes < 0) return "N/A";
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }

    private static String escapeHtml(String s) {
        if (s == null) return "";
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\"", "&quot;").replace("'", "&#39;");
    }

    private static String truncate(String s, int maxLen) {
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen - 3) + "...";
    }

    private static String toJsonObject(Map<String, Object> map) {
        StringBuilder sb = new StringBuilder("{");
        List<Map.Entry<String, Object>> entries = new ArrayList<>(map.entrySet());
        for (int i = 0; i < entries.size(); i++) {
            Map.Entry<String, Object> entry = entries.get(i);
            sb.append(jsonString(entry.getKey())).append(": ");
            Object value = entry.getValue();
            if (value instanceof String) {
                sb.append(jsonString((String) value));
            } else if (value instanceof Number) {
                sb.append(value);
            } else {
                sb.append(jsonString(String.valueOf(value)));
            }
            if (i < entries.size() - 1) sb.append(", ");
        }
        sb.append("}");
        return sb.toString();
    }

    private static final String CSS = """
        <style>
        :root {
          --primary: #2563eb;
          --success: #16a34a;
          --warning: #d97706;
          --error: #dc2626;
          --bg: #f8fafc;
          --card-bg: #ffffff;
          --border: #e2e8f0;
          --text: #1e293b;
          --text-muted: #64748b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: var(--bg);
          color: var(--text);
          line-height: 1.5;
          padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { margin-bottom: 0.5rem; }
        h2 { margin: 2rem 0 1rem; border-bottom: 2px solid var(--border); padding-bottom: 0.5rem; }
        .timestamp { color: var(--text-muted); margin-bottom: 2rem; }
        .summary-cards { display: flex; gap: 1rem; margin-bottom: 2rem; }
        .card {
          background: var(--card-bg);
          border-radius: 8px;
          padding: 1.5rem;
          flex: 1;
          text-align: center;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .card.success .card-value { color: var(--success); }
        .card.warning .card-value { color: var(--warning); }
        .card.error .card-value { color: var(--error); }
        .card-value { font-size: 2rem; font-weight: bold; }
        .card-label { color: var(--text-muted); }
        table {
          width: 100%;
          border-collapse: collapse;
          background: var(--card-bg);
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        th, td { padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }
        th { background: var(--bg); font-weight: 600; }
        tr:last-child td { border-bottom: none; }
        .status-success { color: var(--success); font-weight: 600; }
        .status-warning { color: var(--warning); font-weight: 600; }
        .status-error { color: var(--error); font-weight: 600; }
        .status-skipped { color: var(--text-muted); }
        .metadata-table { margin-bottom: 1rem; }
        .memory-profile { background: var(--card-bg); padding: 1rem; margin-bottom: 1rem; border-radius: 8px; }
        .memory-profile h3 { margin-bottom: 0.5rem; }
        .chart { background: var(--card-bg); padding: 1rem; border-radius: 8px; }
        .chart-row { display: flex; align-items: center; margin-bottom: 0.5rem; }
        .chart-label { width: 150px; font-size: 0.875rem; }
        .chart-bar-container { flex: 1; height: 20px; background: var(--bg); border-radius: 4px; margin: 0 1rem; }
        .chart-bar { height: 100%; background: var(--primary); border-radius: 4px; }
        .chart-value { width: 80px; text-align: right; font-size: 0.875rem; }
        .baseline-summary { padding: 1rem; background: var(--card-bg); border-radius: 8px; margin-bottom: 1rem; }
        </style>
        """;

    /**
     * Builder for creating benchmark reports.
     */
    public static final class Builder {
        private String title = "Benchmark Report";
        private final List<BenchmarkResult> results = new ArrayList<>();
        private PerformanceBaseline baseline;
        private final Map<String, String> metadata = new LinkedHashMap<>();
        private final List<MemoryProfiler.MemorySnapshot> memorySnapshots = new ArrayList<>();

        private Builder() {}

        public Builder title(String title) {
            this.title = title;
            return this;
        }

        public Builder result(BenchmarkResult result) {
            this.results.add(result);
            return this;
        }

        public Builder results(List<BenchmarkResult> results) {
            this.results.addAll(results);
            return this;
        }

        public Builder baseline(PerformanceBaseline baseline) {
            this.baseline = baseline;
            return this;
        }

        public Builder metadata(String key, String value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder memorySnapshot(MemoryProfiler.MemorySnapshot snapshot) {
            this.memorySnapshots.add(snapshot);
            return this;
        }

        public Builder memorySnapshots(List<MemoryProfiler.MemorySnapshot> snapshots) {
            this.memorySnapshots.addAll(snapshots);
            return this;
        }

        public BenchmarkReport build() {
            return new BenchmarkReport(this);
        }
    }
}
