package io.surfworks.warpforge.benchmark.annotation;

import io.surfworks.warpforge.benchmark.KernelTier;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a method as a GPU kernel benchmark.
 *
 * <p>Similar to JMH's {@code @Benchmark}, but designed for GPU kernel measurement.
 * The annotated method should execute a GPU kernel and return the result for
 * validation. The benchmark framework handles warmup, iteration, and timing.
 *
 * <p>Example usage:
 * <pre>{@code
 * @GpuBenchmark(
 *     operation = "Add",
 *     shape = "1M elements",
 *     tiers = {KernelTier.PRODUCTION, KernelTier.OPTIMIZED_OBSERVABLE},
 *     warmupIterations = 5,
 *     measurementIterations = 20
 * )
 * public Tensor benchmarkAddF32(BenchmarkState state) {
 *     return backend.execute(state.addOp, state.inputs);
 * }
 * }</pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface GpuBenchmark {

    /**
     * The StableHLO operation name being benchmarked (e.g., "Add", "DotGeneral").
     */
    String operation();

    /**
     * Human-readable shape description (e.g., "4096x4096", "1M elements").
     */
    String shape() default "";

    /**
     * Execution tiers to benchmark. Defaults to all three tiers.
     */
    KernelTier[] tiers() default {
        KernelTier.PRODUCTION,
        KernelTier.OPTIMIZED_OBSERVABLE,
        KernelTier.CORRECTNESS
    };

    /**
     * Number of warmup iterations before measurement begins.
     * Warmup allows JIT compilation, kernel caching, and GPU warm-up.
     */
    int warmupIterations() default 5;

    /**
     * Number of measurement iterations for statistical analysis.
     */
    int measurementIterations() default 20;

    /**
     * Whether to emit individual kernel events to JFR.
     * Set to false for high-frequency benchmarks to reduce overhead.
     */
    boolean emitKernelEvents() default true;

    /**
     * Expected compute intensity in FLOPS per byte transferred.
     * Used for calculating theoretical peak performance bounds.
     * Set to 0 for memory-bound operations.
     */
    double flopsPerByte() default 0.0;
}
