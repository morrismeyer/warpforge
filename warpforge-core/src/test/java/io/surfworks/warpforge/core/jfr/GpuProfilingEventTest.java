package io.surfworks.warpforge.core.jfr;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for GpuProfilingEvent.
 */
@DisplayName("GpuProfilingEvent Unit Tests")
class GpuProfilingEventTest {

    @Nested
    @DisplayName("Field Assignment")
    class FieldAssignment {

        @Test
        @DisplayName("identity fields are assignable")
        void identityFieldsAssignable() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            event.kernelName = "gemm_nn_128x128";
            event.correlationId = 12345L;
            event.deviceIndex = 0;
            event.streamId = 100L;

            assertEquals("gemm_nn_128x128", event.kernelName);
            assertEquals(12345L, event.correlationId);
            assertEquals(0, event.deviceIndex);
            assertEquals(100L, event.streamId);
        }

        @Test
        @DisplayName("occupancy fields are assignable")
        void occupancyFieldsAssignable() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            event.achievedOccupancyPercent = 75.5;
            event.theoreticalOccupancyPercent = 100.0;
            event.occupancyEfficiencyPercent = 75.5;

            assertEquals(75.5, event.achievedOccupancyPercent, 0.001);
            assertEquals(100.0, event.theoreticalOccupancyPercent, 0.001);
            assertEquals(75.5, event.occupancyEfficiencyPercent, 0.001);
        }

        @Test
        @DisplayName("SM efficiency fields are assignable")
        void smEfficiencyFieldsAssignable() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            event.smEfficiencyPercent = 92.3;
            event.activeWarpsPerSM = 32.5;
            event.eligibleWarpsPerSM = 28.2;

            assertEquals(92.3, event.smEfficiencyPercent, 0.001);
            assertEquals(32.5, event.activeWarpsPerSM, 0.001);
            assertEquals(28.2, event.eligibleWarpsPerSM, 0.001);
        }

        @Test
        @DisplayName("compute utilization fields are assignable")
        void computeUtilizationFieldsAssignable() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            event.computeThroughputPercent = 85.0;
            event.instructionsPerCycle = 2.4;
            event.warpExecutionEfficiencyPercent = 95.0;

            assertEquals(85.0, event.computeThroughputPercent, 0.001);
            assertEquals(2.4, event.instructionsPerCycle, 0.001);
            assertEquals(95.0, event.warpExecutionEfficiencyPercent, 0.001);
        }

        @Test
        @DisplayName("memory throughput fields are assignable")
        void memoryThroughputFieldsAssignable() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            event.memoryThroughputPercent = 70.0;
            event.dramThroughputGBps = 800.5;
            event.dramReadThroughputGBps = 600.3;
            event.dramWriteThroughputGBps = 200.2;

            assertEquals(70.0, event.memoryThroughputPercent, 0.001);
            assertEquals(800.5, event.dramThroughputGBps, 0.001);
            assertEquals(600.3, event.dramReadThroughputGBps, 0.001);
            assertEquals(200.2, event.dramWriteThroughputGBps, 0.001);
        }

        @Test
        @DisplayName("cache metrics are assignable")
        void cacheMetricsAssignable() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            event.l1CacheHitRatePercent = 85.0;
            event.l2CacheHitRatePercent = 92.0;
            event.l1CacheThroughputGBps = 2000.0;
            event.l2CacheThroughputGBps = 1500.0;
            event.sharedMemoryThroughputGBps = 3000.0;

            assertEquals(85.0, event.l1CacheHitRatePercent, 0.001);
            assertEquals(92.0, event.l2CacheHitRatePercent, 0.001);
            assertEquals(2000.0, event.l1CacheThroughputGBps, 0.001);
            assertEquals(1500.0, event.l2CacheThroughputGBps, 0.001);
            assertEquals(3000.0, event.sharedMemoryThroughputGBps, 0.001);
        }

        @Test
        @DisplayName("stall analysis fields are assignable")
        void stallAnalysisFieldsAssignable() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            event.stallMemoryDependencyPercent = 30.0;
            event.stallExecutionDependencyPercent = 15.0;
            event.stallSynchronizationPercent = 5.0;
            event.stallTexturePercent = 2.0;
            event.stallInstructionFetchPercent = 1.0;
            event.stallOtherPercent = 3.0;

            assertEquals(30.0, event.stallMemoryDependencyPercent, 0.001);
            assertEquals(15.0, event.stallExecutionDependencyPercent, 0.001);
            assertEquals(5.0, event.stallSynchronizationPercent, 0.001);
            assertEquals(2.0, event.stallTexturePercent, 0.001);
            assertEquals(1.0, event.stallInstructionFetchPercent, 0.001);
            assertEquals(3.0, event.stallOtherPercent, 0.001);
        }

        @Test
        @DisplayName("timing fields are assignable")
        void timingFieldsAssignable() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            event.kernelDurationNanos = 1_000_000L;
            event.gpuStartTimestamp = 12345678L;
            event.gpuEndTimestamp = 12346678L;

            assertEquals(1_000_000L, event.kernelDurationNanos);
            assertEquals(12345678L, event.gpuStartTimestamp);
            assertEquals(12346678L, event.gpuEndTimestamp);
        }

        @Test
        @DisplayName("profiler metadata fields are assignable")
        void profilerMetadataFieldsAssignable() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            event.profilerBackend = "CUPTI";
            event.profilingOverheadNanos = 50_000L;

            assertEquals("CUPTI", event.profilerBackend);
            assertEquals(50_000L, event.profilingOverheadNanos);
        }
    }

    @Nested
    @DisplayName("Derived Metrics")
    class DerivedMetrics {

        @Test
        @DisplayName("computeDerivedMetrics() calculates occupancy efficiency")
        void computeOccupancyEfficiency() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.achievedOccupancyPercent = 60.0;
            event.theoreticalOccupancyPercent = 80.0;

            event.computeDerivedMetrics();

            assertEquals(75.0, event.occupancyEfficiencyPercent, 0.001);
        }

        @Test
        @DisplayName("computeDerivedMetrics() handles 100% theoretical occupancy")
        void computeFullTheoreticalOccupancy() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.achievedOccupancyPercent = 75.0;
            event.theoreticalOccupancyPercent = 100.0;

            event.computeDerivedMetrics();

            assertEquals(75.0, event.occupancyEfficiencyPercent, 0.001);
        }

        @Test
        @DisplayName("computeDerivedMetrics() handles zero theoretical occupancy")
        void computeZeroTheoreticalOccupancy() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.achievedOccupancyPercent = 50.0;
            event.theoreticalOccupancyPercent = 0.0;

            event.computeDerivedMetrics();

            // Should not change when theoretical is 0
            assertEquals(0.0, event.occupancyEfficiencyPercent, 0.001);
        }

        @Test
        @DisplayName("computeDerivedMetrics() handles perfect efficiency")
        void computePerfectEfficiency() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.achievedOccupancyPercent = 75.0;
            event.theoreticalOccupancyPercent = 75.0;

            event.computeDerivedMetrics();

            assertEquals(100.0, event.occupancyEfficiencyPercent, 0.001);
        }
    }

    @Nested
    @DisplayName("Bottleneck Analysis")
    class BottleneckAnalysis {

        @Test
        @DisplayName("getPrimaryBottleneck() returns memory_dependency when highest")
        void memoryDependencyBottleneck() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.stallMemoryDependencyPercent = 40.0;
            event.stallExecutionDependencyPercent = 20.0;
            event.stallSynchronizationPercent = 10.0;
            event.stallTexturePercent = 5.0;
            event.stallInstructionFetchPercent = 2.0;
            event.computeThroughputPercent = 30.0;
            event.memoryThroughputPercent = 50.0;

            assertEquals("memory_dependency", event.getPrimaryBottleneck());
        }

        @Test
        @DisplayName("getPrimaryBottleneck() returns execution_dependency when highest")
        void executionDependencyBottleneck() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.stallMemoryDependencyPercent = 10.0;
            event.stallExecutionDependencyPercent = 45.0;
            event.stallSynchronizationPercent = 5.0;
            event.stallTexturePercent = 2.0;
            event.stallInstructionFetchPercent = 1.0;
            event.computeThroughputPercent = 30.0;
            event.memoryThroughputPercent = 40.0;

            assertEquals("execution_dependency", event.getPrimaryBottleneck());
        }

        @Test
        @DisplayName("getPrimaryBottleneck() returns synchronization when highest")
        void synchronizationBottleneck() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.stallMemoryDependencyPercent = 5.0;
            event.stallExecutionDependencyPercent = 10.0;
            event.stallSynchronizationPercent = 50.0;
            event.stallTexturePercent = 2.0;
            event.stallInstructionFetchPercent = 1.0;
            event.computeThroughputPercent = 30.0;
            event.memoryThroughputPercent = 40.0;

            assertEquals("synchronization", event.getPrimaryBottleneck());
        }

        @Test
        @DisplayName("getPrimaryBottleneck() returns texture when highest")
        void textureBottleneck() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.stallMemoryDependencyPercent = 5.0;
            event.stallExecutionDependencyPercent = 10.0;
            event.stallSynchronizationPercent = 8.0;
            event.stallTexturePercent = 35.0;
            event.stallInstructionFetchPercent = 1.0;
            event.computeThroughputPercent = 30.0;
            event.memoryThroughputPercent = 40.0;

            assertEquals("texture", event.getPrimaryBottleneck());
        }

        @Test
        @DisplayName("getPrimaryBottleneck() returns instruction_fetch when highest")
        void instructionFetchBottleneck() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.stallMemoryDependencyPercent = 5.0;
            event.stallExecutionDependencyPercent = 10.0;
            event.stallSynchronizationPercent = 8.0;
            event.stallTexturePercent = 3.0;
            event.stallInstructionFetchPercent = 30.0;
            event.computeThroughputPercent = 30.0;
            event.memoryThroughputPercent = 40.0;

            assertEquals("instruction_fetch", event.getPrimaryBottleneck());
        }

        @Test
        @DisplayName("getPrimaryBottleneck() returns compute_bound when compute is high and stalls are low")
        void computeBound() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.stallMemoryDependencyPercent = 3.0;
            event.stallExecutionDependencyPercent = 2.0;
            event.stallSynchronizationPercent = 1.0;
            event.stallTexturePercent = 1.0;
            event.stallInstructionFetchPercent = 1.0;
            event.computeThroughputPercent = 95.0;
            event.memoryThroughputPercent = 40.0;

            assertEquals("compute_bound", event.getPrimaryBottleneck());
        }

        @Test
        @DisplayName("getPrimaryBottleneck() returns memory_bound when memory throughput is high")
        void memoryBound() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.stallMemoryDependencyPercent = 5.0;
            event.stallExecutionDependencyPercent = 3.0;
            event.stallSynchronizationPercent = 1.0;
            event.stallTexturePercent = 1.0;
            event.stallInstructionFetchPercent = 1.0;
            event.computeThroughputPercent = 40.0;
            event.memoryThroughputPercent = 95.0;

            assertEquals("memory_bound", event.getPrimaryBottleneck());
        }

        @Test
        @DisplayName("getPrimaryBottleneck() returns unknown when all stalls are zero")
        void noStalls() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.stallMemoryDependencyPercent = 0.0;
            event.stallExecutionDependencyPercent = 0.0;
            event.stallSynchronizationPercent = 0.0;
            event.stallTexturePercent = 0.0;
            event.stallInstructionFetchPercent = 0.0;
            event.computeThroughputPercent = 50.0;
            event.memoryThroughputPercent = 50.0;

            assertEquals("unknown", event.getPrimaryBottleneck());
        }
    }

    @Nested
    @DisplayName("Bound Detection")
    class BoundDetection {

        @Test
        @DisplayName("isMemoryBound() returns true when memory > compute and high memory stalls")
        void isMemoryBoundTrue() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.memoryThroughputPercent = 80.0;
            event.computeThroughputPercent = 40.0;
            event.stallMemoryDependencyPercent = 35.0;

            assertTrue(event.isMemoryBound());
        }

        @Test
        @DisplayName("isMemoryBound() returns false when compute > memory")
        void isMemoryBoundFalseWhenComputeHigher() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.memoryThroughputPercent = 40.0;
            event.computeThroughputPercent = 80.0;
            event.stallMemoryDependencyPercent = 35.0;

            assertFalse(event.isMemoryBound());
        }

        @Test
        @DisplayName("isMemoryBound() returns false when low memory stalls")
        void isMemoryBoundFalseWhenLowStalls() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.memoryThroughputPercent = 80.0;
            event.computeThroughputPercent = 40.0;
            event.stallMemoryDependencyPercent = 10.0;

            assertFalse(event.isMemoryBound());
        }

        @Test
        @DisplayName("isComputeBound() returns true when compute > memory and execution stalls > memory stalls")
        void isComputeBoundTrue() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.computeThroughputPercent = 85.0;
            event.memoryThroughputPercent = 40.0;
            event.stallExecutionDependencyPercent = 30.0;
            event.stallMemoryDependencyPercent = 10.0;

            assertTrue(event.isComputeBound());
        }

        @Test
        @DisplayName("isComputeBound() returns false when memory > compute")
        void isComputeBoundFalseWhenMemoryHigher() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.computeThroughputPercent = 40.0;
            event.memoryThroughputPercent = 80.0;
            event.stallExecutionDependencyPercent = 30.0;
            event.stallMemoryDependencyPercent = 10.0;

            assertFalse(event.isComputeBound());
        }

        @Test
        @DisplayName("isComputeBound() returns false when memory stalls > execution stalls")
        void isComputeBoundFalseWhenMemoryStallsHigher() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.computeThroughputPercent = 85.0;
            event.memoryThroughputPercent = 40.0;
            event.stallExecutionDependencyPercent = 10.0;
            event.stallMemoryDependencyPercent = 30.0;

            assertFalse(event.isComputeBound());
        }
    }

    @Nested
    @DisplayName("Real-World Scenarios")
    class RealWorldScenarios {

        @Test
        @DisplayName("GEMM kernel profile")
        void gemmKernelProfile() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            // Identity
            event.kernelName = "ampere_sgemm_128x64_nn";
            event.correlationId = 1001;
            event.deviceIndex = 0;
            event.streamId = 1;
            event.profilerBackend = "CUPTI";

            // Occupancy - GEMM typically has high occupancy
            event.achievedOccupancyPercent = 87.5;
            event.theoreticalOccupancyPercent = 100.0;
            event.computeDerivedMetrics();

            // SM efficiency - GEMM should be compute-bound
            event.smEfficiencyPercent = 95.0;
            event.computeThroughputPercent = 92.0;
            event.memoryThroughputPercent = 45.0;

            // Stalls - mostly execution dependencies for GEMM
            event.stallExecutionDependencyPercent = 25.0;
            event.stallMemoryDependencyPercent = 8.0;

            assertEquals(87.5, event.occupancyEfficiencyPercent, 0.001);
            assertTrue(event.isComputeBound());
            assertFalse(event.isMemoryBound());
        }

        @Test
        @DisplayName("memory-bound reduction kernel profile")
        void reductionKernelProfile() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            // Identity
            event.kernelName = "reduce_sum_f32";
            event.correlationId = 2001;
            event.deviceIndex = 0;
            event.streamId = 2;
            event.profilerBackend = "CUPTI";

            // Occupancy - reductions often have lower occupancy
            event.achievedOccupancyPercent = 50.0;
            event.theoreticalOccupancyPercent = 75.0;
            event.computeDerivedMetrics();

            // Memory-bound characteristics
            event.computeThroughputPercent = 30.0;
            event.memoryThroughputPercent = 85.0;
            event.stallMemoryDependencyPercent = 45.0;
            event.stallExecutionDependencyPercent = 5.0;

            assertTrue(event.isMemoryBound());
            assertFalse(event.isComputeBound());
            // When memoryThroughputPercent > 80, getPrimaryBottleneck returns "memory_bound"
            assertEquals("memory_bound", event.getPrimaryBottleneck());
        }

        @Test
        @DisplayName("attention kernel profile")
        void attentionKernelProfile() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            // Identity
            event.kernelName = "flash_attention_v2_fwd";
            event.correlationId = 3001;
            event.deviceIndex = 0;
            event.streamId = 3;
            event.profilerBackend = "CUPTI";

            // High occupancy for fused attention
            event.achievedOccupancyPercent = 81.25;
            event.theoreticalOccupancyPercent = 93.75;
            event.computeDerivedMetrics();

            // Balanced workload
            event.computeThroughputPercent = 75.0;
            event.memoryThroughputPercent = 70.0;
            event.l1CacheHitRatePercent = 92.0;
            event.l2CacheHitRatePercent = 88.0;

            // Low stalls due to good caching
            event.stallMemoryDependencyPercent = 12.0;
            event.stallExecutionDependencyPercent = 10.0;

            assertEquals(86.67, event.occupancyEfficiencyPercent, 0.01);
            assertFalse(event.isMemoryBound());
            assertFalse(event.isComputeBound());
        }

        @Test
        @DisplayName("roctracer backend profile")
        void roctracerProfile() {
            GpuProfilingEvent event = new GpuProfilingEvent();

            event.kernelName = "hipblasGemmEx";
            event.correlationId = 4001;
            event.deviceIndex = 0;
            event.streamId = 1;
            event.profilerBackend = "roctracer";

            event.achievedOccupancyPercent = 82.0;
            event.theoreticalOccupancyPercent = 100.0;
            event.computeDerivedMetrics();

            assertEquals("roctracer", event.profilerBackend);
            assertEquals(82.0, event.occupancyEfficiencyPercent, 0.001);
        }
    }

    @Nested
    @DisplayName("JFR Integration")
    class JfrIntegration {

        @Test
        @DisplayName("event extends jdk.jfr.Event")
        void extendsJfrEvent() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            assertTrue(event instanceof jdk.jfr.Event);
        }

        @Test
        @DisplayName("event can be committed")
        void eventCanBeCommitted() {
            GpuProfilingEvent event = new GpuProfilingEvent();
            event.kernelName = "test_kernel";
            event.correlationId = 1;
            event.achievedOccupancyPercent = 75.0;

            // Should not throw
            event.commit();
        }
    }
}
