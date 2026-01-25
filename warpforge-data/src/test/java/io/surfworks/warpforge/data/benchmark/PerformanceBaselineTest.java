package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.benchmark.PerformanceBaseline.ComparisonResult;
import io.surfworks.warpforge.data.benchmark.PerformanceBaseline.Direction;
import io.surfworks.warpforge.data.benchmark.PerformanceBaseline.Metric;
import io.surfworks.warpforge.data.benchmark.PerformanceBaseline.MetricComparison;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PerformanceBaselineTest {

    @TempDir
    Path tempDir;

    @Nested
    class CreationTests {

        @Test
        void testCreateEmpty() {
            PerformanceBaseline baseline = PerformanceBaseline.create();
            assertTrue(baseline.metricNames().isEmpty());
        }

        @Test
        void testAddMetricSimple() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("throughput", 100.0);

            assertEquals(1, baseline.metricNames().size());
            assertNotNull(baseline.metric("throughput"));
            assertEquals(100.0, baseline.metric("throughput").value());
            assertEquals(10, baseline.metric("throughput").thresholdPercent());
            assertEquals(Direction.HIGHER_IS_BETTER, baseline.metric("throughput").direction());
        }

        @Test
        void testAddMetricFull() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("latency_ms", 5.0, 15, Direction.LOWER_IS_BETTER);

            Metric metric = baseline.metric("latency_ms");
            assertEquals(5.0, metric.value());
            assertEquals(15, metric.thresholdPercent());
            assertEquals(Direction.LOWER_IS_BETTER, metric.direction());
        }

        @Test
        void testAddMultipleMetrics() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("throughput_gbps", 85.0, 10, Direction.HIGHER_IS_BETTER)
                    .addMetric("latency_ms", 12.5, 15, Direction.LOWER_IS_BETTER)
                    .addMetric("memory_mb", 512.0, 20, Direction.LOWER_IS_BETTER);

            assertEquals(3, baseline.metricNames().size());
        }
    }

    @Nested
    class JsonSerializationTests {

        @Test
        void testToJson() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("throughput", 100.0, 10, Direction.HIGHER_IS_BETTER);

            String json = baseline.toJson();

            assertTrue(json.contains("\"version\": \"1.0\""));
            assertTrue(json.contains("\"throughput\""));
            assertTrue(json.contains("\"value\": 100.0"));
            assertTrue(json.contains("\"threshold_percent\": 10"));
            assertTrue(json.contains("\"higher_is_better\""));
        }

        @Test
        void testParseJson() {
            String json = """
                    {
                      "version": "1.0",
                      "updated": "2026-01-25T12:00:00Z",
                      "metrics": {
                        "throughput_gbps": {
                          "value": 85.0,
                          "threshold_percent": 10,
                          "direction": "higher_is_better"
                        },
                        "latency_ms": {
                          "value": 12.5,
                          "threshold_percent": 15,
                          "direction": "lower_is_better"
                        }
                      }
                    }
                    """;

            PerformanceBaseline baseline = PerformanceBaseline.parseJson(json);

            assertEquals(2, baseline.metricNames().size());
            assertEquals(85.0, baseline.metric("throughput_gbps").value());
            assertEquals(Direction.HIGHER_IS_BETTER, baseline.metric("throughput_gbps").direction());
            assertEquals(12.5, baseline.metric("latency_ms").value());
            assertEquals(Direction.LOWER_IS_BETTER, baseline.metric("latency_ms").direction());
        }

        @Test
        void testParseJsonWithNullValue() {
            String json = """
                    {
                      "version": "1.0",
                      "metrics": {
                        "valid_metric": {
                          "value": 100.0,
                          "threshold_percent": 10,
                          "direction": "higher_is_better"
                        },
                        "null_metric": {
                          "value": null,
                          "threshold_percent": 10,
                          "direction": "higher_is_better"
                        }
                      }
                    }
                    """;

            PerformanceBaseline baseline = PerformanceBaseline.parseJson(json);

            assertEquals(1, baseline.metricNames().size());
            assertNotNull(baseline.metric("valid_metric"));
            assertNull(baseline.metric("null_metric"));
        }

        @Test
        void testRoundTrip() {
            PerformanceBaseline original = PerformanceBaseline.create()
                    .addMetric("throughput", 100.0, 10, Direction.HIGHER_IS_BETTER)
                    .addMetric("latency", 5.0, 15, Direction.LOWER_IS_BETTER);

            String json = original.toJson();
            PerformanceBaseline parsed = PerformanceBaseline.parseJson(json);

            assertEquals(original.metricNames(), parsed.metricNames());
            assertEquals(original.metric("throughput").value(), parsed.metric("throughput").value());
            assertEquals(original.metric("latency").value(), parsed.metric("latency").value());
        }
    }

    @Nested
    class FileIOTests {

        @Test
        void testSaveAndLoad() throws IOException {
            PerformanceBaseline original = PerformanceBaseline.create()
                    .addMetric("metric1", 50.0, 10, Direction.HIGHER_IS_BETTER)
                    .addMetric("metric2", 25.0, 5, Direction.LOWER_IS_BETTER);

            Path file = tempDir.resolve("baseline.json");
            original.saveTo(file);

            assertTrue(Files.exists(file));

            PerformanceBaseline loaded = PerformanceBaseline.loadFrom(file);
            assertEquals(2, loaded.metricNames().size());
            assertEquals(50.0, loaded.metric("metric1").value());
            assertEquals(25.0, loaded.metric("metric2").value());
        }
    }

    @Nested
    class ComparisonTests {

        @Test
        void testNoRegression() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("throughput", 100.0, 10, Direction.HIGHER_IS_BETTER);

            ComparisonResult result = baseline.compare(Map.of("throughput", 98.0));

            assertFalse(result.hasRegressions());
            assertEquals(0, result.regressions().size());
            assertEquals(1, result.unchanged().size());
        }

        @Test
        void testRegressionHigherIsBetter() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("throughput", 100.0, 10, Direction.HIGHER_IS_BETTER);

            // 15% lower - regression (beyond 10% threshold)
            ComparisonResult result = baseline.compare(Map.of("throughput", 85.0));

            assertTrue(result.hasRegressions());
            assertEquals(1, result.regressions().size());
            assertEquals("throughput", result.regressions().get(0).name());
        }

        @Test
        void testRegressionLowerIsBetter() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("latency", 10.0, 10, Direction.LOWER_IS_BETTER);

            // 15% higher latency - regression
            ComparisonResult result = baseline.compare(Map.of("latency", 11.5));

            assertTrue(result.hasRegressions());
            assertEquals(1, result.regressions().size());
        }

        @Test
        void testImprovementHigherIsBetter() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("throughput", 100.0, 10, Direction.HIGHER_IS_BETTER);

            // 15% higher - improvement
            ComparisonResult result = baseline.compare(Map.of("throughput", 115.0));

            assertFalse(result.hasRegressions());
            assertTrue(result.hasImprovements());
            assertEquals(1, result.improvements().size());
        }

        @Test
        void testImprovementLowerIsBetter() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("latency", 10.0, 10, Direction.LOWER_IS_BETTER);

            // 20% lower latency - improvement
            ComparisonResult result = baseline.compare(Map.of("latency", 8.0));

            assertFalse(result.hasRegressions());
            assertTrue(result.hasImprovements());
            assertEquals(1, result.improvements().size());
        }

        @Test
        void testMixedResults() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("throughput", 100.0, 10, Direction.HIGHER_IS_BETTER)
                    .addMetric("latency", 10.0, 10, Direction.LOWER_IS_BETTER)
                    .addMetric("memory", 512.0, 20, Direction.LOWER_IS_BETTER);

            ComparisonResult result = baseline.compare(Map.of(
                    "throughput", 80.0,  // Regression (20% drop)
                    "latency", 8.0,      // Improvement (20% lower)
                    "memory", 500.0      // Unchanged (within threshold)
            ));

            assertEquals(1, result.regressions().size());
            assertEquals(1, result.improvements().size());
            assertEquals(1, result.unchanged().size());
            assertEquals(3, result.totalCompared());
        }

        @Test
        void testMissingBaseline() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("known_metric", 100.0);

            ComparisonResult result = baseline.compare(Map.of(
                    "known_metric", 95.0,
                    "unknown_metric", 50.0
            ));

            assertEquals(1, result.missing().size());
            assertTrue(result.missing().contains("unknown_metric"));
        }

        @Test
        void testExitCode() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("metric", 100.0, 10, Direction.HIGHER_IS_BETTER);

            ComparisonResult noRegression = baseline.compare(Map.of("metric", 95.0));
            assertEquals(0, noRegression.exitCode());

            ComparisonResult withRegression = baseline.compare(Map.of("metric", 80.0));
            assertEquals(1, withRegression.exitCode());
        }
    }

    @Nested
    class MetricComparisonTests {

        @Test
        void testMetricComparisonRegression() {
            MetricComparison comparison = new MetricComparison(
                    "test", 100.0, 85.0, -15.0, 10, Direction.HIGHER_IS_BETTER
            );

            assertTrue(comparison.isRegression());
            assertFalse(comparison.isImprovement());
        }

        @Test
        void testMetricComparisonImprovement() {
            MetricComparison comparison = new MetricComparison(
                    "test", 100.0, 120.0, 20.0, 10, Direction.HIGHER_IS_BETTER
            );

            assertFalse(comparison.isRegression());
            assertTrue(comparison.isImprovement());
        }

        @Test
        void testMetricComparisonToString() {
            MetricComparison regression = new MetricComparison(
                    "throughput", 100.0, 85.0, -15.0, 10, Direction.HIGHER_IS_BETTER
            );

            String str = regression.toString();
            assertTrue(str.contains("throughput"));
            assertTrue(str.contains("100.0"));
            assertTrue(str.contains("85.0"));
            assertTrue(str.contains("REGRESSION"));
        }
    }

    @Nested
    class SummaryTests {

        @Test
        void testSummaryFormat() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("metric1", 100.0, 10, Direction.HIGHER_IS_BETTER)
                    .addMetric("metric2", 50.0, 10, Direction.HIGHER_IS_BETTER);

            ComparisonResult result = baseline.compare(Map.of(
                    "metric1", 80.0,  // Regression (20% drop)
                    "metric2", 60.0   // Improvement (20% increase, beyond 10% threshold)
            ));

            String summary = result.summary();

            assertTrue(summary.contains("Performance Comparison Summary"));
            assertTrue(summary.contains("Total metrics: 2"));
            assertTrue(summary.contains("Regressions: 1"));
            assertTrue(summary.contains("Improvements: 1"));
            assertTrue(summary.contains("FAIL"));
        }

        @Test
        void testSummaryPass() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("metric", 100.0, 10, Direction.HIGHER_IS_BETTER);

            ComparisonResult result = baseline.compare(Map.of("metric", 95.0));

            String summary = result.summary();
            assertTrue(summary.contains("PASS"));
        }

        @Test
        void testToJson() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("metric", 100.0, 10, Direction.HIGHER_IS_BETTER);

            ComparisonResult result = baseline.compare(Map.of("metric", 80.0));

            String json = result.toJson();
            assertTrue(json.contains("\"passed\": false"));
            assertTrue(json.contains("\"regression_count\": 1"));
        }
    }

    @Nested
    class UpdateTests {

        @Test
        void testUpdateMetrics() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("metric1", 100.0, 10, Direction.HIGHER_IS_BETTER)
                    .addMetric("metric2", 50.0, 15, Direction.LOWER_IS_BETTER);

            baseline.updateMetrics(Map.of(
                    "metric1", 110.0,
                    "metric2", 45.0
            ));

            assertEquals(110.0, baseline.metric("metric1").value());
            assertEquals(10, baseline.metric("metric1").thresholdPercent()); // Preserved
            assertEquals(45.0, baseline.metric("metric2").value());
            assertEquals(15, baseline.metric("metric2").thresholdPercent()); // Preserved
        }

        @Test
        void testUpdateIgnoresUnknownMetrics() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("known", 100.0);

            baseline.updateMetrics(Map.of(
                    "known", 110.0,
                    "unknown", 999.0
            ));

            assertEquals(110.0, baseline.metric("known").value());
            assertNull(baseline.metric("unknown"));
        }
    }

    @Nested
    class DirectionTests {

        @Test
        void testFromString() {
            assertEquals(Direction.HIGHER_IS_BETTER, Direction.fromString("higher_is_better"));
            assertEquals(Direction.HIGHER_IS_BETTER, Direction.fromString("higher"));
            assertEquals(Direction.LOWER_IS_BETTER, Direction.fromString("lower_is_better"));
            assertEquals(Direction.LOWER_IS_BETTER, Direction.fromString("lower"));
            assertEquals(Direction.HIGHER_IS_BETTER, Direction.fromString("unknown")); // Default
        }

        @Test
        void testToJson() {
            assertEquals("higher_is_better", Direction.HIGHER_IS_BETTER.toJson());
            assertEquals("lower_is_better", Direction.LOWER_IS_BETTER.toJson());
        }
    }

    @Nested
    class EdgeCaseTests {

        @Test
        void testZeroBaseline() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("zero_metric", 0.0, 10, Direction.HIGHER_IS_BETTER);

            // Should not throw with zero baseline
            ComparisonResult result = baseline.compare(Map.of("zero_metric", 10.0));
            assertNotNull(result);
        }

        @Test
        void testExactMatch() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("metric", 100.0, 10, Direction.HIGHER_IS_BETTER);

            ComparisonResult result = baseline.compare(Map.of("metric", 100.0));

            assertFalse(result.hasRegressions());
            assertFalse(result.hasImprovements());
            assertEquals(1, result.unchanged().size());
        }

        @Test
        void testBoundaryValues() {
            PerformanceBaseline baseline = PerformanceBaseline.create()
                    .addMetric("metric", 100.0, 10, Direction.HIGHER_IS_BETTER);

            // Exactly at threshold - should be unchanged
            ComparisonResult atLowerBound = baseline.compare(Map.of("metric", 90.0));
            assertEquals(0, atLowerBound.regressions().size());
            assertEquals(1, atLowerBound.unchanged().size());

            // Just beyond threshold - regression
            ComparisonResult beyondLowerBound = baseline.compare(Map.of("metric", 89.9));
            assertEquals(1, beyondLowerBound.regressions().size());
        }
    }
}
