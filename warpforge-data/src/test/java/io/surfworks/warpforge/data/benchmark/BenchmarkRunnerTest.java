package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.golden.GoldenOutput;
import io.surfworks.warpforge.data.golden.GoldenStore;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BenchmarkRunnerTest {

    @TempDir
    Path tempDir;

    private BenchmarkRunner runner;
    private ByteArrayOutputStream outputCapture;

    @BeforeEach
    void setUp() {
        outputCapture = new ByteArrayOutputStream();
        runner = new BenchmarkRunner()
                .progressOutput(new PrintStream(outputCapture));
    }

    @Nested
    class BasicExecutionTests {

        @Test
        void testRunSyntheticBenchmark() {
            SyntheticBenchmark benchmark = SyntheticBenchmark.builder("test-basic")
                    .inputShape(1, 10)
                    .outputShape(1, 5)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(2)
                    .measurementIterations(5)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status());
            assertEquals("test-basic", result.benchmarkName());
            assertEquals("synthetic", result.modelId());
            assertEquals(2, result.warmupIterations());
            assertEquals(5, result.measurementIterations());
            assertEquals(5, result.latenciesNanos().length);
        }

        @Test
        void testLatencyMeasurement() {
            SyntheticBenchmark benchmark = SyntheticBenchmark.builder("latency-test")
                    .simulatedLatencyMs(5) // 5ms simulated latency
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(3)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            // Mean latency should be approximately 5ms (with some tolerance)
            assertTrue(result.meanLatencyMs() >= 4.0,
                    "Mean latency should be at least 4ms, was: " + result.meanLatencyMs());
            assertTrue(result.meanLatencyMs() <= 20.0,
                    "Mean latency should be at most 20ms, was: " + result.meanLatencyMs());
        }

        @Test
        void testProgressOutput() {
            SyntheticBenchmark benchmark = SyntheticBenchmark.builder("progress-test")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(2)
                    .validateOutputs(false)
                    .build();

            runner.run(benchmark, config);

            String output = outputCapture.toString();
            assertTrue(output.contains("Starting benchmark"));
            assertTrue(output.contains("progress-test"));
            assertTrue(output.contains("Setting up"));
            assertTrue(output.contains("Warming up"));
            assertTrue(output.contains("Measuring"));
            assertTrue(output.contains("Complete"));
        }
    }

    @Nested
    class ValidationTests {

        @Test
        void testValidationWithGoldenStore() throws IOException {
            // Create a golden output
            GoldenStore store = GoldenStore.file(tempDir);
            try (Arena arena = Arena.ofConfined()) {
                float[] goldenData = new float[5];
                // The synthetic benchmark produces deterministic output based on input
                // We need to match what SyntheticBenchmark.runInference produces
                for (int i = 0; i < 5; i++) {
                    goldenData[i] = 0.5f * 1.0f + (i * 0.001f); // Approximate expected output
                }

                MemorySegment segment = arena.allocate(20);
                for (int i = 0; i < 5; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, goldenData[i]);
                }

                TensorInfo info = new TensorInfo("output", DType.F32, new long[]{1, 5}, 0, 20);
                GoldenOutput golden = GoldenOutput.builder("validation-test/output")
                        .tensor(new TensorView(segment, info))
                        .tolerance(1.0) // Very lenient tolerance for this test
                        .build();
                store.save(golden);
            }

            // Run benchmark with validation
            runner.goldenStore(store);

            SyntheticBenchmark benchmark = SyntheticBenchmark.builder("validation-test")
                    .inputShape(1, 10)
                    .outputShape(1, 5)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(2)
                    .validateOutputs(true)
                    .tolerance(1.0)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertFalse(result.validationResults().isEmpty());
        }

        @Test
        void testValidationWithoutGoldenStore() {
            // No golden store configured - should still succeed
            SyntheticBenchmark benchmark = SyntheticBenchmark.builder("no-golden-test")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(2)
                    .validateOutputs(true)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            assertEquals(BenchmarkResult.Status.SUCCESS, result.status());
        }
    }

    @Nested
    class StatisticsTests {

        @Test
        void testLatencyStatistics() {
            SyntheticBenchmark benchmark = SyntheticBenchmark.builder("stats-test")
                    .simulatedLatencyMs(1)
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(2)
                    .measurementIterations(10)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            // All latencies should be recorded
            assertEquals(10, result.latenciesNanos().length);

            // Statistics should be computed
            assertTrue(result.meanLatencyNanos() > 0);
            assertTrue(result.minLatencyNanos() > 0);
            assertTrue(result.maxLatencyNanos() >= result.minLatencyNanos());
            assertTrue(result.medianLatencyNanos() > 0);
            assertTrue(result.p95LatencyNanos() >= result.medianLatencyNanos());
            assertTrue(result.p99LatencyNanos() >= result.p95LatencyNanos());
            assertTrue(result.throughputPerSecond() > 0);
        }

        @Test
        void testSummaryGeneration() {
            SyntheticBenchmark benchmark = SyntheticBenchmark.builder("summary-test")
                    .build();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(5)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(benchmark, config);

            String summary = result.summary();
            assertNotNull(summary);
            assertTrue(summary.contains("summary-test"));
            assertTrue(summary.contains("synthetic"));
            assertTrue(summary.contains("SUCCESS"));
            assertTrue(summary.contains("Latency"));
            assertTrue(summary.contains("Throughput"));
        }
    }

    @Nested
    class MultipleRunsTests {

        @Test
        void testRunAllBenchmarks() {
            List<ModelBenchmark> benchmarks = List.of(
                    SyntheticBenchmark.builder("benchmark-1").build(),
                    SyntheticBenchmark.builder("benchmark-2").build(),
                    SyntheticBenchmark.builder("benchmark-3").build()
            );

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(2)
                    .validateOutputs(false)
                    .build();

            List<BenchmarkResult> results = runner.runAll(benchmarks, config);

            assertEquals(3, results.size());
            assertTrue(results.stream().allMatch(r -> r.status() == BenchmarkResult.Status.SUCCESS));
        }

        @Test
        void testGenerateReport() {
            List<ModelBenchmark> benchmarks = List.of(
                    SyntheticBenchmark.builder("report-bench-1").build(),
                    SyntheticBenchmark.builder("report-bench-2").build()
            );

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(1)
                    .measurementIterations(3)
                    .validateOutputs(false)
                    .build();

            List<BenchmarkResult> results = runner.runAll(benchmarks, config);
            String report = BenchmarkRunner.generateReport(results);

            assertNotNull(report);
            assertTrue(report.contains("Benchmark Report"));
            assertTrue(report.contains("report-bench-1"));
            assertTrue(report.contains("report-bench-2"));
            assertTrue(report.contains("SUCCESS"));
            assertTrue(report.contains("passed"));
        }
    }

    @Nested
    class ErrorHandlingTests {

        @Test
        void testBenchmarkWithError() {
            // Create a benchmark that throws during inference
            ModelBenchmark failingBenchmark = new FailingBenchmark();

            BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
                    .warmupIterations(0) // Don't warmup to hit error faster
                    .measurementIterations(2)
                    .validateOutputs(false)
                    .build();

            BenchmarkResult result = runner.run(failingBenchmark, config);

            assertEquals(BenchmarkResult.Status.ERROR, result.status());
            assertNotNull(result.errorMessage());
            assertTrue(result.errorMessage().contains("Simulated failure"));
        }
    }

    /**
     * A benchmark that always fails during inference.
     */
    static class FailingBenchmark implements ModelBenchmark {
        @Override
        public String name() { return "failing-test"; }

        @Override
        public String modelId() { return "synthetic"; }

        @Override
        public void setup(BenchmarkConfig config) {}

        @Override
        public java.util.Map<String, TensorView> prepareInputs(BenchmarkConfig config) {
            return java.util.Map.of();
        }

        @Override
        public java.util.Map<String, TensorView> runInference(
                java.util.Map<String, TensorView> inputs) throws IOException {
            throw new IOException("Simulated failure");
        }

        @Override
        public List<String> outputsToValidate() { return List.of(); }

        @Override
        public void teardown() {}
    }
}
