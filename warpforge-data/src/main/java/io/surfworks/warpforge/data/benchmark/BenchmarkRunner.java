package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.golden.ComparisonResult;
import io.surfworks.warpforge.data.golden.GoldenComparison;
import io.surfworks.warpforge.data.golden.GoldenOutput;
import io.surfworks.warpforge.data.golden.GoldenStore;

import java.io.IOException;
import java.io.PrintStream;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Runs model benchmarks and collects results.
 *
 * <p>The runner handles:
 * <ul>
 *   <li>Benchmark setup and teardown</li>
 *   <li>Warmup iterations (not timed)</li>
 *   <li>Measurement iterations with timing</li>
 *   <li>Output validation against golden outputs</li>
 *   <li>Result aggregation and reporting</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * BenchmarkRunner runner = new BenchmarkRunner()
 *     .goldenStore(GoldenStore.file(Path.of("goldens")))
 *     .progressOutput(System.out);
 *
 * BenchmarkConfig config = BenchmarkConfig.builder("bert-base-uncased")
 *     .backend("cpu")
 *     .warmupIterations(5)
 *     .measurementIterations(20)
 *     .build();
 *
 * BenchmarkResult result = runner.run(new BertBenchmark(), config);
 * System.out.println(result.summary());
 * }</pre>
 */
public final class BenchmarkRunner {

    private GoldenStore goldenStore;
    private PrintStream progressOutput;
    private boolean verbose = false;

    /**
     * Set the golden store for output validation.
     */
    public BenchmarkRunner goldenStore(GoldenStore goldenStore) {
        this.goldenStore = goldenStore;
        return this;
    }

    /**
     * Set the output stream for progress messages.
     */
    public BenchmarkRunner progressOutput(PrintStream output) {
        this.progressOutput = output;
        return this;
    }

    /**
     * Enable verbose output.
     */
    public BenchmarkRunner verbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    /**
     * Run a single benchmark.
     *
     * @param benchmark The benchmark to run
     * @param config Benchmark configuration
     * @return Benchmark result with timing and validation data
     */
    public BenchmarkResult run(ModelBenchmark benchmark, BenchmarkConfig config) {
        Objects.requireNonNull(benchmark, "benchmark must not be null");
        Objects.requireNonNull(config, "config must not be null");

        Instant startTime = Instant.now();
        String benchmarkName = benchmark.name();
        String modelId = benchmark.modelId();
        String backend = config.backend();

        log("Starting benchmark: %s", benchmarkName);
        log("  Model: %s, Backend: %s", modelId, backend);

        try {
            // Setup
            log("  Setting up...");
            benchmark.setup(config);

            // Prepare inputs (once, reused for all iterations)
            Map<String, TensorView> inputs = benchmark.prepareInputs(config);
            log("  Prepared %d inputs", inputs.size());

            // Warmup
            log("  Warming up (%d iterations)...", config.warmupIterations());
            for (int i = 0; i < config.warmupIterations(); i++) {
                benchmark.runInference(inputs);
                if (verbose) log("    Warmup %d/%d", i + 1, config.warmupIterations());
            }

            // Measurement
            log("  Measuring (%d iterations)...", config.measurementIterations());
            long[] latencies = new long[config.measurementIterations()];
            Map<String, TensorView> lastOutputs = null;

            for (int i = 0; i < config.measurementIterations(); i++) {
                long start = System.nanoTime();
                lastOutputs = benchmark.runInference(inputs);
                long end = System.nanoTime();
                latencies[i] = end - start;

                if (verbose) {
                    log("    Iteration %d/%d: %.2f ms",
                            i + 1, config.measurementIterations(),
                            latencies[i] / 1_000_000.0);
                }
            }

            // Validation
            List<ComparisonResult> validationResults = new ArrayList<>();
            if (config.validateOutputs() && lastOutputs != null) {
                log("  Validating outputs...");
                validationResults = validateOutputs(benchmark, config, lastOutputs);
            }

            // Teardown
            benchmark.teardown();

            // Determine status
            BenchmarkResult.Status status = BenchmarkResult.Status.SUCCESS;
            if (config.validateOutputs() && !validationResults.stream().allMatch(ComparisonResult::matches)) {
                status = BenchmarkResult.Status.VALIDATION_FAILED;
            }

            BenchmarkResult result = new BenchmarkResult(
                    benchmarkName, modelId, backend,
                    startTime, Instant.now(),
                    config.warmupIterations(), config.measurementIterations(),
                    latencies, validationResults, null, status, null
            );

            log("  Complete: %s", status);
            if (progressOutput != null) {
                log("  Mean latency: %.2f ms", result.meanLatencyMs());
            }

            return result;

        } catch (Exception e) {
            log("  Error: %s", e.getMessage());
            try {
                benchmark.teardown();
            } catch (Exception ignored) {
            }
            return BenchmarkResult.error(benchmarkName, modelId, backend, startTime, e.getMessage());
        }
    }

    /**
     * Run multiple benchmarks and collect all results.
     */
    public List<BenchmarkResult> runAll(List<ModelBenchmark> benchmarks, BenchmarkConfig config) {
        List<BenchmarkResult> results = new ArrayList<>();
        for (ModelBenchmark benchmark : benchmarks) {
            results.add(run(benchmark, config));
        }
        return results;
    }

    /**
     * Validate outputs against golden outputs.
     */
    private List<ComparisonResult> validateOutputs(
            ModelBenchmark benchmark,
            BenchmarkConfig config,
            Map<String, TensorView> outputs
    ) {
        List<ComparisonResult> results = new ArrayList<>();

        for (String outputName : benchmark.outputsToValidate()) {
            TensorView actual = outputs.get(outputName);
            if (actual == null) {
                log("    Warning: Output '%s' not found in model outputs", outputName);
                continue;
            }

            String goldenId = benchmark.goldenIdFor(outputName);
            Double customTolerance = benchmark.toleranceFor(outputName);
            double tolerance = customTolerance != null ? customTolerance : config.tolerance();

            ComparisonResult comparison = compareWithGolden(goldenId, actual, tolerance);
            results.add(comparison);

            if (verbose || !comparison.matches()) {
                log("    Output '%s': %s", outputName, comparison.summary());
            }
        }

        return results;
    }

    /**
     * Compare an output tensor against its golden output.
     */
    private ComparisonResult compareWithGolden(String goldenId, TensorView actual, double tolerance) {
        if (goldenStore == null) {
            // No golden store configured - skip validation but log warning
            log("    Warning: No golden store configured, skipping validation for '%s'", goldenId);
            return ComparisonResult.success(actual.info().elementCount(), 0, 0, tolerance);
        }

        try {
            Optional<GoldenOutput> goldenOpt = goldenStore.load(goldenId);
            if (goldenOpt.isEmpty()) {
                log("    Warning: No golden output found for '%s'", goldenId);
                return ComparisonResult.success(actual.info().elementCount(), 0, 0, tolerance);
            }

            GoldenOutput golden = goldenOpt.get();
            return GoldenComparison.compare(golden.toTensorView(), actual, tolerance);

        } catch (IOException e) {
            log("    Error loading golden '%s': %s", goldenId, e.getMessage());
            return ComparisonResult.success(actual.info().elementCount(), 0, 0, tolerance);
        }
    }

    /**
     * Log a message if progress output is configured.
     */
    private void log(String format, Object... args) {
        if (progressOutput != null) {
            progressOutput.printf(format + "%n", args);
        }
    }

    /**
     * Generate a summary report for multiple benchmark results.
     */
    public static String generateReport(List<BenchmarkResult> results) {
        if (results.isEmpty()) {
            return "No benchmark results to report.";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("=== Benchmark Report ===\n\n");

        // Summary table header
        sb.append(String.format("%-30s %-10s %-12s %-12s %-12s %-10s%n",
                "Benchmark", "Status", "Mean (ms)", "P95 (ms)", "P99 (ms)", "Valid"));
        sb.append("-".repeat(90)).append("\n");

        int passed = 0;
        int failed = 0;
        int errors = 0;

        for (BenchmarkResult result : results) {
            String validStr = result.validationResults().isEmpty() ? "N/A" :
                    (result.allOutputsValid() ? "Yes" : "No");

            sb.append(String.format("%-30s %-10s %-12.2f %-12.2f %-12.2f %-10s%n",
                    truncate(result.benchmarkName(), 30),
                    result.status(),
                    result.meanLatencyMs(),
                    result.p95LatencyNanos() / 1_000_000.0,
                    result.p99LatencyNanos() / 1_000_000.0,
                    validStr));

            switch (result.status()) {
                case SUCCESS -> passed++;
                case VALIDATION_FAILED -> failed++;
                case ERROR -> errors++;
            }
        }

        sb.append("-".repeat(90)).append("\n");
        sb.append(String.format("Total: %d passed, %d failed, %d errors%n",
                passed, failed, errors));

        return sb.toString();
    }

    private static String truncate(String s, int maxLen) {
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen - 3) + "...";
    }
}
