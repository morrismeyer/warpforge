package io.surfworks.warpforge.benchmark;

import io.surfworks.warpforge.benchmark.annotation.GpuBenchmark;
import io.surfworks.warpforge.benchmark.annotation.Setup;
import io.surfworks.warpforge.benchmark.annotation.TearDown;

/**
 * Sample benchmark demonstrating the GPU benchmark framework.
 *
 * <p>This benchmark measures elementwise addition performance across
 * all three execution tiers to validate the overhead claims.
 */
public class SampleAddBenchmark {

    // Simulated state - in real benchmarks, these would be GPU tensors
    private float[] inputA;
    private float[] inputB;
    private float[] output;
    private int size;

    @Setup(level = Setup.Level.TRIAL)
    public void setup() {
        // Allocate test data - 1M elements
        size = 1_000_000;
        inputA = new float[size];
        inputB = new float[size];
        output = new float[size];

        // Initialize with random data
        for (int i = 0; i < size; i++) {
            inputA[i] = (float) Math.random();
            inputB[i] = (float) Math.random();
        }
    }

    @TearDown(level = Setup.Level.TRIAL)
    public void teardown() {
        inputA = null;
        inputB = null;
        output = null;
    }

    @GpuBenchmark(
        operation = "Add",
        shape = "1M elements",
        warmupIterations = 3,
        measurementIterations = 10,
        tiers = {KernelTier.PRODUCTION, KernelTier.OPTIMIZED_OBSERVABLE}
    )
    public void benchmarkAddF32(KernelTier tier) {
        // Simulate tier-dependent overhead
        simulateKernelExecution(tier);
    }

    @GpuBenchmark(
        operation = "Add",
        shape = "4M elements",
        warmupIterations = 3,
        measurementIterations = 10,
        tiers = {KernelTier.PRODUCTION, KernelTier.OPTIMIZED_OBSERVABLE}
    )
    public void benchmarkAddF32Large(KernelTier tier) {
        // Simulate with larger data
        simulateLargeKernelExecution(tier);
    }

    private void simulateKernelExecution(KernelTier tier) {
        // Simulate actual computation - in real benchmark this would be GPU kernel
        for (int i = 0; i < size; i++) {
            output[i] = inputA[i] + inputB[i];
        }

        // Simulate tier-dependent overhead
        switch (tier) {
            case PRODUCTION -> {
                // No additional overhead
            }
            case OPTIMIZED_OBSERVABLE -> {
                // Simulate ~7% overhead from salt instrumentation
                simulateInstrumentationOverhead(0.07);
            }
            case CORRECTNESS -> {
                // Simulate significant overhead
                simulateInstrumentationOverhead(0.99);
            }
        }
    }

    private void simulateLargeKernelExecution(KernelTier tier) {
        // Just run 4x to simulate 4M elements
        for (int j = 0; j < 4; j++) {
            simulateKernelExecution(tier);
        }
    }

    private void simulateInstrumentationOverhead(double fraction) {
        // Busy-wait to simulate overhead
        long start = System.nanoTime();
        long baseTime = 100_000; // 100us base
        long targetWait = (long) (baseTime * fraction);
        while (System.nanoTime() - start < targetWait) {
            // Busy wait
        }
    }
}
