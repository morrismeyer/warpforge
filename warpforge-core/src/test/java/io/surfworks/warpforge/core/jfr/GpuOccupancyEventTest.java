package io.surfworks.warpforge.core.jfr;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for GpuOccupancyEvent.
 */
@DisplayName("GpuOccupancyEvent Unit Tests")
class GpuOccupancyEventTest {

    @Nested
    @DisplayName("Field Assignment")
    class FieldAssignment {

        @Test
        @DisplayName("all fields are assignable")
        void allFieldsAssignable() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();

            event.deviceIndex = 0;
            event.deviceName = "NVIDIA RTX 4090";
            event.activeStreams = 4;
            event.activeKernels = 8;
            event.queuedKernels = 2;
            event.estimatedTotalOccupancyPercent = 75;
            event.activeWarpsEstimate = 3072;
            event.estimatedActiveSMs = 64;
            event.gpuUtilizationPercent = 85;
            event.memoryUtilizationPercent = 45;
            event.temperatureCelsius = 72;
            event.powerDrawWatts = 350;
            event.smCount = 128;
            event.maxWarpsPerSM = 48;
            event.totalMemoryMB = 24576;
            event.usedMemoryMB = 8192;
            event.trackedScopes = 3;
            event.virtualThreadsWithGpuWork = 12;

            assertEquals(0, event.deviceIndex);
            assertEquals("NVIDIA RTX 4090", event.deviceName);
            assertEquals(4, event.activeStreams);
            assertEquals(8, event.activeKernels);
            assertEquals(2, event.queuedKernels);
            assertEquals(75, event.estimatedTotalOccupancyPercent);
            assertEquals(3072, event.activeWarpsEstimate);
            assertEquals(64, event.estimatedActiveSMs);
            assertEquals(85, event.gpuUtilizationPercent);
            assertEquals(45, event.memoryUtilizationPercent);
            assertEquals(72, event.temperatureCelsius);
            assertEquals(350, event.powerDrawWatts);
            assertEquals(128, event.smCount);
            assertEquals(48, event.maxWarpsPerSM);
            assertEquals(24576, event.totalMemoryMB);
            assertEquals(8192, event.usedMemoryMB);
            assertEquals(3, event.trackedScopes);
            assertEquals(12, event.virtualThreadsWithGpuWork);
        }
    }

    @Nested
    @DisplayName("Computed Fields")
    class ComputedFields {

        @Test
        @DisplayName("computeDerivedFields() calculates occupancy percentage")
        void computeOccupancyPercentage() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();
            event.smCount = 128;
            event.maxWarpsPerSM = 48;
            event.activeWarpsEstimate = 3072; // 50% of 6144

            event.computeDerivedFields();

            assertEquals(50, event.estimatedTotalOccupancyPercent);
        }

        @Test
        @DisplayName("computeDerivedFields() handles 100% occupancy")
        void computeFullOccupancy() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();
            event.smCount = 128;
            event.maxWarpsPerSM = 48;
            event.activeWarpsEstimate = 6144; // 100%

            event.computeDerivedFields();

            assertEquals(100, event.estimatedTotalOccupancyPercent);
        }

        @Test
        @DisplayName("computeDerivedFields() handles zero warps")
        void computeZeroOccupancy() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();
            event.smCount = 128;
            event.maxWarpsPerSM = 48;
            event.activeWarpsEstimate = 0;

            event.computeDerivedFields();

            assertEquals(0, event.estimatedTotalOccupancyPercent);
        }

        @Test
        @DisplayName("computeDerivedFields() handles unconfigured device")
        void computeUnconfiguredDevice() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();
            event.smCount = 0; // Not configured
            event.maxWarpsPerSM = 0;
            event.activeWarpsEstimate = 100;

            // Should not throw or produce invalid values
            event.computeDerivedFields();

            assertEquals(0, event.estimatedTotalOccupancyPercent);
        }
    }

    @Nested
    @DisplayName("Memory Helpers")
    class MemoryHelpers {

        @Test
        @DisplayName("setMemoryUsage() converts bytes to MB")
        void setMemoryUsageConvertsMB() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();

            // 24 GB total, 8 GB used
            long totalBytes = 24L * 1024 * 1024 * 1024;
            long usedBytes = 8L * 1024 * 1024 * 1024;

            event.setMemoryUsage(totalBytes, usedBytes);

            assertEquals(24576, event.totalMemoryMB); // 24 GB in MB
            assertEquals(8192, event.usedMemoryMB);   // 8 GB in MB
        }

        @Test
        @DisplayName("setMemoryUsage() handles small values")
        void setMemoryUsageSmallValues() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();

            // 1 MB total, 512 KB used
            event.setMemoryUsage(1024 * 1024, 512 * 1024);

            assertEquals(1, event.totalMemoryMB);
            assertEquals(0, event.usedMemoryMB); // Truncated to 0
        }

        @Test
        @DisplayName("setMemoryUsage() handles zero values")
        void setMemoryUsageZeroValues() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();

            event.setMemoryUsage(0, 0);

            assertEquals(0, event.totalMemoryMB);
            assertEquals(0, event.usedMemoryMB);
        }
    }

    @Nested
    @DisplayName("Real-World Scenarios")
    class RealWorldScenarios {

        @Test
        @DisplayName("RTX 4090 training scenario")
        void rtx4090TrainingScenario() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();

            // RTX 4090 specs
            event.deviceIndex = 0;
            event.deviceName = "NVIDIA GeForce RTX 4090";
            event.smCount = 128;
            event.maxWarpsPerSM = 48;
            event.setMemoryUsage(24L * 1024 * 1024 * 1024, 20L * 1024 * 1024 * 1024);

            // Training workload
            event.activeStreams = 2;
            event.activeKernels = 4;
            event.queuedKernels = 1;
            event.activeWarpsEstimate = 4608; // 75% occupancy
            event.computeDerivedFields();

            event.gpuUtilizationPercent = 95;
            event.memoryUtilizationPercent = 78;
            event.temperatureCelsius = 78;
            event.powerDrawWatts = 420;

            assertEquals(75, event.estimatedTotalOccupancyPercent);
            assertTrue(event.gpuUtilizationPercent > 90);
        }

        @Test
        @DisplayName("MI300X inference scenario")
        void mi300xInferenceScenario() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();

            // MI300X specs (CDNA3, 64-wide wavefronts)
            event.deviceIndex = 0;
            event.deviceName = "AMD Instinct MI300X";
            event.smCount = 304; // 304 CUs
            event.maxWarpsPerSM = 32; // wavefronts per CU
            event.setMemoryUsage(192L * 1024 * 1024 * 1024, 64L * 1024 * 1024 * 1024);

            // Inference workload
            event.activeStreams = 8;
            event.activeKernels = 8;
            event.queuedKernels = 0;
            event.activeWarpsEstimate = 4864; // ~50% occupancy
            event.computeDerivedFields();

            event.gpuUtilizationPercent = 65;
            event.memoryUtilizationPercent = 33;
            event.temperatureCelsius = 62;
            event.powerDrawWatts = 550;

            assertEquals(50, event.estimatedTotalOccupancyPercent);
        }

        @Test
        @DisplayName("idle GPU scenario")
        void idleGpuScenario() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();

            event.deviceIndex = 0;
            event.smCount = 128;
            event.maxWarpsPerSM = 48;

            event.activeStreams = 0;
            event.activeKernels = 0;
            event.queuedKernels = 0;
            event.activeWarpsEstimate = 0;
            event.computeDerivedFields();

            event.gpuUtilizationPercent = 0;
            event.temperatureCelsius = 35;
            event.powerDrawWatts = 25; // Idle power

            assertEquals(0, event.estimatedTotalOccupancyPercent);
            assertEquals(0, event.activeKernels);
        }
    }

    @Nested
    @DisplayName("JFR Integration")
    class JfrIntegration {

        @Test
        @DisplayName("event extends jdk.jfr.Event")
        void extendsJfrEvent() {
            GpuOccupancyEvent event = new GpuOccupancyEvent();
            assertTrue(event instanceof jdk.jfr.Event);
        }

        @Test
        @DisplayName("event can be committed")
        void eventCanBeCommitted() {
            // This test verifies the event is properly structured for JFR
            GpuOccupancyEvent event = new GpuOccupancyEvent();
            event.deviceIndex = 0;
            event.activeKernels = 1;

            // Should not throw
            // Note: In production, JFR must be enabled for commit() to record
            event.commit();
        }
    }
}
