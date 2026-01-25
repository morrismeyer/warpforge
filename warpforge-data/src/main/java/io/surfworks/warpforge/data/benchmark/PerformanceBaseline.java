package io.surfworks.warpforge.data.benchmark;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

/**
 * Performance baseline management for regression detection.
 *
 * <p>Baselines store performance metrics with thresholds to detect regressions.
 * Each metric specifies:
 * <ul>
 *   <li>value - the baseline value</li>
 *   <li>threshold_percent - allowable deviation (e.g., 10 means ±10%)</li>
 *   <li>direction - whether higher or lower values are better</li>
 * </ul>
 *
 * <p>Baseline format (JSON):
 * <pre>{@code
 * {
 *   "version": "1.0",
 *   "updated": "2026-01-25T16:00:00Z",
 *   "metrics": {
 *     "matmul_throughput_gbps": {
 *       "value": 85.0,
 *       "threshold_percent": 10,
 *       "direction": "higher_is_better"
 *     },
 *     "inference_latency_ms": {
 *       "value": 12.5,
 *       "threshold_percent": 15,
 *       "direction": "lower_is_better"
 *     }
 *   }
 * }
 * }</pre>
 *
 * <p>Example usage:
 * <pre>{@code
 * // Load baseline
 * PerformanceBaseline baseline = PerformanceBaseline.loadFrom(Path.of("baselines/gpu.json"));
 *
 * // Compare current results
 * Map<String, Double> currentResults = Map.of(
 *     "matmul_throughput_gbps", 82.3,
 *     "inference_latency_ms", 13.1
 * );
 *
 * ComparisonResult result = baseline.compare(currentResults);
 * if (result.hasRegressions()) {
 *     System.err.println("Performance regressions detected!");
 *     result.regressions().forEach(System.err::println);
 *     System.exit(1);
 * }
 *
 * // Update baseline if improved
 * if (result.improvements().size() > 0) {
 *     baseline.updateMetrics(currentResults);
 *     baseline.saveTo(Path.of("baselines/gpu.json"));
 * }
 * }</pre>
 */
public final class PerformanceBaseline {

    private static final String VERSION = "1.0";
    private static final Gson GSON = new GsonBuilder()
            .setPrettyPrinting()
            .create();

    private final Map<String, Metric> metrics;
    private String version;
    private Instant updated;

    private PerformanceBaseline() {
        this.metrics = new HashMap<>();
        this.version = VERSION;
        this.updated = Instant.now();
    }

    /**
     * Create an empty baseline.
     */
    public static PerformanceBaseline create() {
        return new PerformanceBaseline();
    }

    /**
     * Load baseline from JSON file.
     */
    public static PerformanceBaseline loadFrom(Path path) throws IOException {
        try (Reader reader = Files.newBufferedReader(path)) {
            return parse(reader);
        }
    }

    /**
     * Parse baseline from JSON reader.
     */
    public static PerformanceBaseline parse(Reader reader) {
        JsonObject root = JsonParser.parseReader(reader).getAsJsonObject();
        PerformanceBaseline baseline = new PerformanceBaseline();

        baseline.version = root.has("version") ? root.get("version").getAsString() : VERSION;
        baseline.updated = root.has("updated")
                ? Instant.parse(root.get("updated").getAsString())
                : Instant.now();

        if (root.has("metrics")) {
            JsonObject metricsObj = root.getAsJsonObject("metrics");
            for (String key : metricsObj.keySet()) {
                JsonObject metricObj = metricsObj.getAsJsonObject(key);
                if (metricObj.has("value") && !metricObj.get("value").isJsonNull()) {
                    double value = metricObj.get("value").getAsDouble();
                    int threshold = metricObj.has("threshold_percent")
                            ? metricObj.get("threshold_percent").getAsInt() : 10;
                    Direction direction = metricObj.has("direction")
                            ? Direction.fromString(metricObj.get("direction").getAsString())
                            : Direction.HIGHER_IS_BETTER;
                    baseline.metrics.put(key, new Metric(value, threshold, direction));
                }
            }
        }

        return baseline;
    }

    /**
     * Parse baseline from JSON string.
     */
    public static PerformanceBaseline parseJson(String json) {
        return parse(new java.io.StringReader(json));
    }

    /**
     * Save baseline to JSON file.
     */
    public void saveTo(Path path) throws IOException {
        try (Writer writer = Files.newBufferedWriter(path)) {
            writer.write(toJson());
        }
    }

    /**
     * Convert to JSON string.
     */
    public String toJson() {
        JsonObject root = new JsonObject();
        root.addProperty("version", version);
        root.addProperty("updated", DateTimeFormatter.ISO_INSTANT
                .format(updated.atOffset(ZoneOffset.UTC)));

        JsonObject metricsObj = new JsonObject();
        for (var entry : metrics.entrySet()) {
            JsonObject metricObj = new JsonObject();
            metricObj.addProperty("value", entry.getValue().value);
            metricObj.addProperty("threshold_percent", entry.getValue().thresholdPercent);
            metricObj.addProperty("direction", entry.getValue().direction.toJson());
            metricsObj.add(entry.getKey(), metricObj);
        }
        root.add("metrics", metricsObj);

        return GSON.toJson(root);
    }

    /**
     * Add or update a metric.
     */
    public PerformanceBaseline addMetric(String name, double value, int thresholdPercent, Direction direction) {
        metrics.put(name, new Metric(value, thresholdPercent, direction));
        updated = Instant.now();
        return this;
    }

    /**
     * Add metric with default threshold (10%) and direction (higher is better).
     */
    public PerformanceBaseline addMetric(String name, double value) {
        return addMetric(name, value, 10, Direction.HIGHER_IS_BETTER);
    }

    /**
     * Get all metric names.
     */
    public java.util.Set<String> metricNames() {
        return metrics.keySet();
    }

    /**
     * Get a specific metric.
     */
    public Metric metric(String name) {
        return metrics.get(name);
    }

    /**
     * Update metrics from measured values (keeps thresholds and directions).
     */
    public void updateMetrics(Map<String, Double> newValues) {
        for (var entry : newValues.entrySet()) {
            Metric existing = metrics.get(entry.getKey());
            if (existing != null) {
                metrics.put(entry.getKey(), new Metric(
                        entry.getValue(),
                        existing.thresholdPercent,
                        existing.direction
                ));
            }
        }
        updated = Instant.now();
    }

    /**
     * Compare current results against baseline.
     */
    public ComparisonResult compare(Map<String, Double> currentResults) {
        return compare(currentResults, false);
    }

    /**
     * Compare current results against baseline.
     *
     * @param currentResults Map of metric name to current measured value
     * @param strict If true, fail for any missing baseline metric
     * @return Comparison result with regressions and improvements
     */
    public ComparisonResult compare(Map<String, Double> currentResults, boolean strict) {
        var regressions = new java.util.ArrayList<MetricComparison>();
        var improvements = new java.util.ArrayList<MetricComparison>();
        var unchanged = new java.util.ArrayList<MetricComparison>();
        var missing = new java.util.ArrayList<String>();

        for (var entry : currentResults.entrySet()) {
            String name = entry.getKey();
            double currentValue = entry.getValue();

            Metric baseline = metrics.get(name);
            if (baseline == null) {
                missing.add(name);
                continue;
            }

            double percentChange = calculatePercentChange(baseline.value, currentValue, baseline.direction);
            MetricComparison comparison = new MetricComparison(
                    name,
                    baseline.value,
                    currentValue,
                    percentChange,
                    baseline.thresholdPercent,
                    baseline.direction
            );

            if (isRegression(baseline, currentValue)) {
                regressions.add(comparison);
            } else if (isImprovement(baseline, currentValue)) {
                improvements.add(comparison);
            } else {
                unchanged.add(comparison);
            }
        }

        return new ComparisonResult(regressions, improvements, unchanged, missing);
    }

    private boolean isRegression(Metric baseline, double currentValue) {
        double threshold = baseline.value * baseline.thresholdPercent / 100.0;
        return switch (baseline.direction) {
            case HIGHER_IS_BETTER -> currentValue < baseline.value - threshold;
            case LOWER_IS_BETTER -> currentValue > baseline.value + threshold;
        };
    }

    private boolean isImprovement(Metric baseline, double currentValue) {
        double threshold = baseline.value * baseline.thresholdPercent / 100.0;
        return switch (baseline.direction) {
            case HIGHER_IS_BETTER -> currentValue > baseline.value + threshold;
            case LOWER_IS_BETTER -> currentValue < baseline.value - threshold;
        };
    }

    private double calculatePercentChange(double baseline, double current, Direction direction) {
        if (baseline == 0) return 0;
        double rawChange = ((current - baseline) / baseline) * 100.0;
        // For lower_is_better, negative change is improvement
        return direction == Direction.LOWER_IS_BETTER ? -rawChange : rawChange;
    }

    /**
     * Direction indicator for metrics.
     */
    public enum Direction {
        HIGHER_IS_BETTER("higher_is_better"),
        LOWER_IS_BETTER("lower_is_better");

        private final String jsonValue;

        Direction(String jsonValue) {
            this.jsonValue = jsonValue;
        }

        public String toJson() {
            return jsonValue;
        }

        public static Direction fromString(String s) {
            return switch (s.toLowerCase().replace("-", "_")) {
                case "higher_is_better", "higher" -> HIGHER_IS_BETTER;
                case "lower_is_better", "lower" -> LOWER_IS_BETTER;
                default -> HIGHER_IS_BETTER;
            };
        }
    }

    /**
     * A baseline metric with value, threshold, and direction.
     */
    public record Metric(double value, int thresholdPercent, Direction direction) {
        @Override
        public String toString() {
            return String.format("%.4f (±%d%%, %s)", value, thresholdPercent, direction.toJson());
        }
    }

    /**
     * Result of comparing a single metric.
     */
    public record MetricComparison(
            String name,
            double baselineValue,
            double currentValue,
            double percentChange,
            int thresholdPercent,
            Direction direction
    ) {
        public boolean isRegression() {
            return percentChange < -thresholdPercent;
        }

        public boolean isImprovement() {
            return percentChange > thresholdPercent;
        }

        @Override
        public String toString() {
            String changeStr = percentChange >= 0 ? "+" + String.format("%.2f", percentChange)
                    : String.format("%.2f", percentChange);
            String status = isRegression() ? "REGRESSION" : (isImprovement() ? "IMPROVEMENT" : "OK");
            return String.format("%s: %.4f -> %.4f (%s%%, threshold: %d%%) [%s]",
                    name, baselineValue, currentValue, changeStr, thresholdPercent, status);
        }
    }

    /**
     * Result of baseline comparison.
     */
    public record ComparisonResult(
            java.util.List<MetricComparison> regressions,
            java.util.List<MetricComparison> improvements,
            java.util.List<MetricComparison> unchanged,
            java.util.List<String> missing
    ) {
        /**
         * Whether any regressions were detected.
         */
        public boolean hasRegressions() {
            return !regressions.isEmpty();
        }

        /**
         * Whether any improvements were detected.
         */
        public boolean hasImprovements() {
            return !improvements.isEmpty();
        }

        /**
         * Total number of metrics compared.
         */
        public int totalCompared() {
            return regressions.size() + improvements.size() + unchanged.size();
        }

        /**
         * Get exit code (0 if pass, 1 if regressions).
         */
        public int exitCode() {
            return hasRegressions() ? 1 : 0;
        }

        /**
         * Generate human-readable summary.
         */
        public String summary() {
            StringBuilder sb = new StringBuilder();
            sb.append("=== Performance Comparison Summary ===\n");
            sb.append(String.format("Total metrics: %d\n", totalCompared()));
            sb.append(String.format("Regressions: %d\n", regressions.size()));
            sb.append(String.format("Improvements: %d\n", improvements.size()));
            sb.append(String.format("Unchanged: %d\n", unchanged.size()));
            if (!missing.isEmpty()) {
                sb.append(String.format("Missing baselines: %d\n", missing.size()));
            }
            sb.append("\n");

            if (!regressions.isEmpty()) {
                sb.append("--- REGRESSIONS ---\n");
                for (var r : regressions) {
                    sb.append("  ").append(r.toString()).append("\n");
                }
                sb.append("\n");
            }

            if (!improvements.isEmpty()) {
                sb.append("--- IMPROVEMENTS ---\n");
                for (var i : improvements) {
                    sb.append("  ").append(i.toString()).append("\n");
                }
                sb.append("\n");
            }

            if (hasRegressions()) {
                sb.append("RESULT: FAIL - Performance regressions detected!\n");
            } else {
                sb.append("RESULT: PASS - No regressions detected.\n");
            }

            return sb.toString();
        }

        /**
         * Convert to JSON for programmatic consumption.
         */
        public String toJson() {
            JsonObject root = new JsonObject();
            root.addProperty("passed", !hasRegressions());
            root.addProperty("total_compared", totalCompared());
            root.addProperty("regression_count", regressions.size());
            root.addProperty("improvement_count", improvements.size());

            JsonObject regressionsObj = new JsonObject();
            for (var r : regressions) {
                JsonObject m = new JsonObject();
                m.addProperty("baseline", r.baselineValue);
                m.addProperty("current", r.currentValue);
                m.addProperty("percent_change", r.percentChange);
                m.addProperty("threshold", r.thresholdPercent);
                regressionsObj.add(r.name, m);
            }
            root.add("regressions", regressionsObj);

            JsonObject improvementsObj = new JsonObject();
            for (var i : improvements) {
                JsonObject m = new JsonObject();
                m.addProperty("baseline", i.baselineValue);
                m.addProperty("current", i.currentValue);
                m.addProperty("percent_change", i.percentChange);
                improvementsObj.add(i.name, m);
            }
            root.add("improvements", improvementsObj);

            return GSON.toJson(root);
        }
    }
}
