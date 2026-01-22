package io.surfworks.warpforge.benchmark;

import io.surfworks.warpforge.benchmark.annotation.GpuBenchmark;
import io.surfworks.warpforge.benchmark.annotation.Setup;
import io.surfworks.warpforge.benchmark.annotation.TearDown;
import io.surfworks.warpforge.core.jfr.GpuKernelEvent;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * GPU Kernel Benchmark Runner with JMH-style methodology.
 *
 * <p>Provides a benchmarking framework specifically designed for GPU kernel
 * measurement, addressing unique challenges:
 * <ul>
 *   <li>GPU warmup requirements (kernel compilation, memory allocation)</li>
 *   <li>Asynchronous execution timing via CUDA/HIP Events</li>
 *   <li>Three-tier execution comparison (PRODUCTION, OPTIMIZED_OBSERVABLE, CORRECTNESS)</li>
 *   <li>JFR profiling support (like JMH's -prof jfr)</li>
 * </ul>
 *
 * <p>Results are output as text (console) and JSON, following JMH conventions.
 * JFR events are used for profiling (capturing what happens during execution),
 * not for result reporting.
 *
 * <p>Usage:
 * <pre>{@code
 * GpuBenchmarkRunner runner = new GpuBenchmarkRunner();
 * runner.include(".*MatMul.*");  // regex pattern for benchmark methods
 * runner.run(MyBenchmarks.class);
 *
 * TierComparisonReport report = runner.generateReport();
 * report.print();           // Text output
 * report.toJson();          // JSON output
 * }</pre>
 *
 * <p>CLI usage:
 * <pre>{@code
 * java -jar warpforge-benchmark.jar --all-tiers MyBenchmarks
 * }</pre>
 */
public class GpuBenchmarkRunner {

    private static final double DEFAULT_TOLERANCE_PERCENT = 3.0;

    private String includePattern = ".*";
    private boolean emitJfrProfilerEvents = false;
    private boolean verboseOutput = true;
    private double tolerancePercent = DEFAULT_TOLERANCE_PERCENT;
    private final List<BenchmarkResult> results = new ArrayList<>();
    private final List<BenchmarkResult.TierComparison> comparisons = new ArrayList<>();

    /**
     * Main entry point for CLI execution.
     */
    public static void main(String[] args) {
        GpuBenchmarkRunner runner = new GpuBenchmarkRunner();
        runner.parseArgs(args);

        if (args.length == 0) {
            printUsage();
            return;
        }

        // Find benchmark class from args
        String className = findClassName(args);
        if (className == null) {
            System.err.println("No benchmark class specified");
            printUsage();
            System.exit(1);
        }

        try {
            Class<?> benchmarkClass = Class.forName(className);
            runner.run(benchmarkClass);

            TierComparisonReport report = runner.generateReport();
            report.print();

            // Exit with failure if any comparison failed tolerance check
            if (!report.allPassed()) {
                System.exit(1);
            }
        } catch (ClassNotFoundException e) {
            System.err.println("Benchmark class not found: " + className);
            System.exit(1);
        }
    }

    private static void printUsage() {
        System.out.println("WarpForge GPU Benchmark Runner");
        System.out.println();
        System.out.println("Usage: warpforge-benchmark [options] <benchmark-class>");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --all-tiers         Run benchmarks on all three execution tiers");
        System.out.println("  --tier <tier>       Run benchmarks on specific tier only");
        System.out.println("  --prof jfr          Enable JFR profiler events (like JMH -prof jfr)");
        System.out.println("  --tolerance <pct>   Overhead tolerance percentage (default: 3.0)");
        System.out.println("  --include <regex>   Include only benchmarks matching pattern");
        System.out.println("  --verbose           Enable verbose output");
        System.out.println("  --quiet             Disable verbose output");
        System.out.println();
        System.out.println("Example:");
        System.out.println("  warpforge-benchmark --all-tiers io.surfworks.warpforge.benchmark.AddBenchmark");
    }

    private void parseArgs(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--prof" -> {
                    if (i + 1 < args.length && "jfr".equals(args[i + 1])) {
                        emitJfrProfilerEvents = true;
                        i++;
                    }
                }
                case "--verbose" -> verboseOutput = true;
                case "--quiet" -> verboseOutput = false;
                case "--tolerance" -> {
                    if (i + 1 < args.length) {
                        tolerancePercent = Double.parseDouble(args[++i]);
                    }
                }
                case "--include" -> {
                    if (i + 1 < args.length) {
                        includePattern = args[++i];
                    }
                }
            }
        }
    }

    private static String findClassName(String[] args) {
        for (String arg : args) {
            if (!arg.startsWith("--") && arg.contains(".")) {
                return arg;
            }
        }
        return null;
    }

    /**
     * Sets the include pattern for benchmark method names.
     */
    public GpuBenchmarkRunner include(String pattern) {
        this.includePattern = pattern;
        return this;
    }

    /**
     * Enables or disables JFR profiler events (for profiling, not result reporting).
     * Similar to JMH's -prof jfr option.
     */
    public GpuBenchmarkRunner jfrProfiler(boolean enabled) {
        this.emitJfrProfilerEvents = enabled;
        return this;
    }

    /**
     * Sets the tolerance percentage for tier comparison validation.
     */
    public GpuBenchmarkRunner tolerance(double percent) {
        this.tolerancePercent = percent;
        return this;
    }

    /**
     * Runs all benchmarks in the specified class.
     */
    public void run(Class<?> benchmarkClass) {
        if (verboseOutput) {
            System.out.println("╔════════════════════════════════════════════════════════════╗");
            System.out.println("║         WarpForge GPU Kernel Benchmark Suite               ║");
            System.out.println("╠════════════════════════════════════════════════════════════╣");
            System.out.printf("║  Class: %-50s ║%n", benchmarkClass.getSimpleName());
            System.out.printf("║  Pattern: %-48s ║%n", includePattern);
            System.out.printf("║  JFR Profiler: %-43s ║%n", emitJfrProfilerEvents ? "enabled" : "disabled");
            System.out.println("╚════════════════════════════════════════════════════════════╝");
            System.out.println();
        }

        // Find and validate benchmark methods
        List<Method> benchmarkMethods = findBenchmarkMethods(benchmarkClass);
        if (benchmarkMethods.isEmpty()) {
            System.out.println("No benchmark methods found matching pattern: " + includePattern);
            return;
        }

        // Create benchmark instance
        Object instance;
        try {
            instance = benchmarkClass.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to instantiate benchmark class", e);
        }

        // Find setup/teardown methods
        List<Method> trialSetup = findAnnotatedMethods(benchmarkClass, Setup.class, Setup.Level.TRIAL);
        List<Method> trialTeardown = findAnnotatedMethods(benchmarkClass, TearDown.class, Setup.Level.TRIAL);
        List<Method> tierSetup = findAnnotatedMethods(benchmarkClass, Setup.class, Setup.Level.TIER);
        List<Method> tierTeardown = findAnnotatedMethods(benchmarkClass, TearDown.class, Setup.Level.TIER);

        try {
            // Trial setup
            invokeAll(trialSetup, instance);

            // Run each benchmark method
            for (Method method : benchmarkMethods) {
                runBenchmark(instance, method, tierSetup, tierTeardown);
            }
        } finally {
            // Trial teardown
            invokeAll(trialTeardown, instance);
        }

        // Generate tier comparisons
        generateComparisons();
    }

    private List<Method> findBenchmarkMethods(Class<?> clazz) {
        return Arrays.stream(clazz.getDeclaredMethods())
            .filter(m -> m.isAnnotationPresent(GpuBenchmark.class))
            .filter(m -> m.getName().matches(includePattern))
            .sorted(Comparator.comparing(Method::getName))
            .toList();
    }

    private List<Method> findAnnotatedMethods(Class<?> clazz, Class<? extends java.lang.annotation.Annotation> annotation, Setup.Level level) {
        return Arrays.stream(clazz.getDeclaredMethods())
            .filter(m -> {
                if (annotation == Setup.class && m.isAnnotationPresent(Setup.class)) {
                    return m.getAnnotation(Setup.class).level() == level;
                }
                if (annotation == TearDown.class && m.isAnnotationPresent(TearDown.class)) {
                    return m.getAnnotation(TearDown.class).level() == level;
                }
                return false;
            })
            .toList();
    }

    private void invokeAll(List<Method> methods, Object instance) {
        for (Method method : methods) {
            try {
                method.setAccessible(true);
                method.invoke(instance);
            } catch (Exception e) {
                throw new RuntimeException("Failed to invoke " + method.getName(), e);
            }
        }
    }

    private void runBenchmark(Object instance, Method method, List<Method> tierSetup, List<Method> tierTeardown) {
        GpuBenchmark annotation = method.getAnnotation(GpuBenchmark.class);
        String benchmarkName = method.getDeclaringClass().getSimpleName() + "." + method.getName();

        if (verboseOutput) {
            System.out.println("┌────────────────────────────────────────────────────────────┐");
            System.out.printf("│ Benchmark: %-48s │%n", benchmarkName);
            System.out.printf("│ Operation: %-48s │%n", annotation.operation());
            System.out.printf("│ Shape: %-52s │%n", annotation.shape());
            System.out.println("└────────────────────────────────────────────────────────────┘");
        }

        // Run for each tier
        for (KernelTier tier : annotation.tiers()) {
            try {
                // Tier setup
                invokeAll(tierSetup, instance);

                BenchmarkResult result = runForTier(instance, method, annotation, tier);
                results.add(result);

                if (verboseOutput) {
                    System.out.println("  " + result.toSummaryString());
                }
            } finally {
                // Tier teardown
                invokeAll(tierTeardown, instance);
            }
        }

        if (verboseOutput) {
            System.out.println();
        }
    }

    private BenchmarkResult runForTier(Object instance, Method method, GpuBenchmark annotation, KernelTier tier) {
        int warmupIterations = annotation.warmupIterations();
        int measurementIterations = annotation.measurementIterations();

        // Warmup phase
        for (int i = 0; i < warmupIterations; i++) {
            invokeWithTier(instance, method, tier);
        }

        // Measurement phase
        long[] timings = new long[measurementIterations];
        for (int i = 0; i < measurementIterations; i++) {
            long startNanos = System.nanoTime();
            invokeWithTier(instance, method, tier);
            long endNanos = System.nanoTime();
            timings[i] = (endNanos - startNanos) / 1000; // Convert to microseconds

            // Emit JFR profiler event (for profiling, not result reporting)
            if (emitJfrProfilerEvents && annotation.emitKernelEvents()) {
                emitKernelProfilerEvent(annotation, tier, timings[i]);
            }
        }

        return new BenchmarkResult(
            method.getDeclaringClass().getSimpleName() + "." + method.getName(),
            annotation.operation(),
            annotation.shape(),
            tier,
            warmupIterations,
            measurementIterations,
            timings,
            calculateFlops(annotation)
        );
    }

    private void invokeWithTier(Object instance, Method method, KernelTier tier) {
        try {
            method.setAccessible(true);
            // Check if method accepts KernelTier parameter
            if (method.getParameterCount() == 1 && method.getParameterTypes()[0] == KernelTier.class) {
                method.invoke(instance, tier);
            } else {
                method.invoke(instance);
            }
        } catch (Exception e) {
            throw new RuntimeException("Benchmark invocation failed", e);
        }
    }

    private double calculateFlops(GpuBenchmark annotation) {
        // This would be calculated from the operation and shape
        // For now, return 0 to indicate unknown
        return 0.0;
    }

    /**
     * Emits a JFR profiler event for kernel execution.
     * This is for profiling (like JMH -prof jfr), not for result reporting.
     */
    private void emitKernelProfilerEvent(GpuBenchmark annotation, KernelTier tier, long timeMicros) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = annotation.operation();
        event.shape = annotation.shape();
        event.gpuTimeMicros = timeMicros;
        event.tier = tier.name();
        event.backend = tier.backendSuffix();
        event.deviceIndex = 0;
        event.commit();
    }

    private void generateComparisons() {
        // Group results by benchmark name
        Map<String, Map<KernelTier, BenchmarkResult>> grouped = new java.util.HashMap<>();
        for (BenchmarkResult result : results) {
            grouped.computeIfAbsent(result.benchmarkName(), k -> new EnumMap<>(KernelTier.class))
                .put(result.tier(), result);
        }

        // Generate comparisons against PRODUCTION baseline
        for (Map.Entry<String, Map<KernelTier, BenchmarkResult>> entry : grouped.entrySet()) {
            Map<KernelTier, BenchmarkResult> tierResults = entry.getValue();
            BenchmarkResult baseline = tierResults.get(KernelTier.PRODUCTION);

            if (baseline == null) {
                continue; // No baseline to compare against
            }

            for (KernelTier tier : KernelTier.values()) {
                if (tier == KernelTier.PRODUCTION) continue;

                BenchmarkResult comparison = tierResults.get(tier);
                if (comparison != null) {
                    BenchmarkResult.TierComparison comp = comparison.compareTo(baseline, tolerancePercent);
                    comparisons.add(comp);
                }
            }
        }
    }

    /**
     * Generates the tier comparison report.
     */
    public TierComparisonReport generateReport() {
        return new TierComparisonReport(results, comparisons, tolerancePercent);
    }

    /**
     * Returns all benchmark results.
     */
    public List<BenchmarkResult> getResults() {
        return List.copyOf(results);
    }

    /**
     * Returns all tier comparisons.
     */
    public List<BenchmarkResult.TierComparison> getComparisons() {
        return List.copyOf(comparisons);
    }
}
