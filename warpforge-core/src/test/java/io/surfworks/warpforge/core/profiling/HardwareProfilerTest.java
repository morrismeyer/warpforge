package io.surfworks.warpforge.core.profiling;

import io.surfworks.warpforge.core.jfr.GpuProfilingEvent;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for HardwareProfiler interface and nested types.
 */
@DisplayName("HardwareProfiler Interface Tests")
class HardwareProfilerTest {

    @Nested
    @DisplayName("SessionConfig")
    class SessionConfigTests {

        @Test
        @DisplayName("ALL config enables all collection")
        void allConfigEnablesAllCollection() {
            HardwareProfiler.SessionConfig config = HardwareProfiler.SessionConfig.ALL;

            assertTrue(config.collectOccupancy());
            assertTrue(config.collectCacheMetrics());
            assertTrue(config.collectStallAnalysis());
            assertTrue(config.collectThroughput());
            assertEquals(0, config.maxKernels());
        }

        @Test
        @DisplayName("OCCUPANCY_ONLY config enables only occupancy")
        void occupancyOnlyConfig() {
            HardwareProfiler.SessionConfig config = HardwareProfiler.SessionConfig.OCCUPANCY_ONLY;

            assertTrue(config.collectOccupancy());
            assertFalse(config.collectCacheMetrics());
            assertFalse(config.collectStallAnalysis());
            assertFalse(config.collectThroughput());
            assertEquals(0, config.maxKernels());
        }

        @Test
        @DisplayName("BOTTLENECK_ANALYSIS config enables occupancy, cache, stall, throughput")
        void bottleneckAnalysisConfig() {
            HardwareProfiler.SessionConfig config = HardwareProfiler.SessionConfig.BOTTLENECK_ANALYSIS;

            assertTrue(config.collectOccupancy());
            assertTrue(config.collectCacheMetrics());
            assertTrue(config.collectStallAnalysis());
            assertTrue(config.collectThroughput());
            assertEquals(0, config.maxKernels());
        }

        @Test
        @DisplayName("custom config with limited kernels")
        void customConfigWithLimit() {
            HardwareProfiler.SessionConfig config = new HardwareProfiler.SessionConfig(
                true, false, true, false, 100
            );

            assertTrue(config.collectOccupancy());
            assertFalse(config.collectCacheMetrics());
            assertTrue(config.collectStallAnalysis());
            assertFalse(config.collectThroughput());
            assertEquals(100, config.maxKernels());
        }
    }

    @Nested
    @DisplayName("ProfilingMetrics")
    class ProfilingMetricsTests {

        /**
         * Test implementation of ProfilingMetrics.
         */
        private static class TestMetrics implements HardwareProfiler.ProfilingMetrics {
            @Override public long correlationId() { return 12345L; }
            @Override public String kernelName() { return "test_kernel"; }
            @Override public double achievedOccupancyPercent() { return 75.0; }
            @Override public double smEfficiencyPercent() { return 92.0; }
            @Override public double computeThroughputPercent() { return 85.0; }
            @Override public double memoryThroughputPercent() { return 60.0; }
            @Override public double l1CacheHitRatePercent() { return 88.0; }
            @Override public double l2CacheHitRatePercent() { return 95.0; }
            @Override public double stallMemoryDependencyPercent() { return 15.0; }
            @Override public double stallExecutionDependencyPercent() { return 10.0; }
            @Override public double stallSynchronizationPercent() { return 5.0; }
            @Override public long kernelDurationNanos() { return 500_000L; }
            @Override public long profilingOverheadNanos() { return 10_000L; }
        }

        @Test
        @DisplayName("populateEvent() fills all fields")
        void populateEventFillsAllFields() {
            TestMetrics metrics = new TestMetrics();
            GpuProfilingEvent event = new GpuProfilingEvent();

            metrics.populateEvent(event);

            assertEquals(12345L, event.correlationId);
            assertEquals("test_kernel", event.kernelName);
            assertEquals(75.0, event.achievedOccupancyPercent, 0.001);
            assertEquals(92.0, event.smEfficiencyPercent, 0.001);
            assertEquals(85.0, event.computeThroughputPercent, 0.001);
            assertEquals(60.0, event.memoryThroughputPercent, 0.001);
            assertEquals(88.0, event.l1CacheHitRatePercent, 0.001);
            assertEquals(95.0, event.l2CacheHitRatePercent, 0.001);
            assertEquals(15.0, event.stallMemoryDependencyPercent, 0.001);
            assertEquals(10.0, event.stallExecutionDependencyPercent, 0.001);
            assertEquals(5.0, event.stallSynchronizationPercent, 0.001);
            assertEquals(500_000L, event.kernelDurationNanos);
            assertEquals(10_000L, event.profilingOverheadNanos);
        }
    }

    @Nested
    @DisplayName("Mock Profiler")
    class MockProfilerTests {

        /**
         * Simple mock implementation for testing interface usage patterns.
         */
        private static class MockProfiler implements HardwareProfiler {
            private boolean sessionActive = false;
            private final int deviceIndex;

            MockProfiler(int deviceIndex) {
                this.deviceIndex = deviceIndex;
            }

            @Override
            public void startSession(SessionConfig config) {
                if (sessionActive) {
                    throw new IllegalStateException("Session already active");
                }
                sessionActive = true;
            }

            @Override
            public void stopSession() {
                if (!sessionActive) {
                    throw new IllegalStateException("No active session");
                }
                sessionActive = false;
            }

            @Override
            public boolean isSessionActive() {
                return sessionActive;
            }

            @Override
            public Optional<ProfilingMetrics> getMetrics(long correlationId) {
                return Optional.empty();
            }

            @Override
            public List<ProfilingMetrics> getAllMetrics() {
                return List.of();
            }

            @Override
            public void flush() {
                // no-op
            }

            @Override
            public void clearMetrics() {
                // no-op
            }

            @Override
            public String getBackendName() {
                return "MockProfiler";
            }

            @Override
            public int getDeviceIndex() {
                return deviceIndex;
            }

            @Override
            public boolean isSupported() {
                return true;
            }

            @Override
            public long estimatedOverheadNanos() {
                return 50_000L;
            }

            @Override
            public void close() {
                if (sessionActive) {
                    sessionActive = false;
                }
            }
        }

        @Test
        @DisplayName("session lifecycle works correctly")
        void sessionLifecycle() {
            MockProfiler profiler = new MockProfiler(0);

            assertFalse(profiler.isSessionActive());

            profiler.startSession();
            assertTrue(profiler.isSessionActive());

            profiler.stopSession();
            assertFalse(profiler.isSessionActive());
        }

        @Test
        @DisplayName("default startSession() uses ALL config")
        void defaultStartSessionUsesAllConfig() {
            MockProfiler profiler = new MockProfiler(0);

            // Should work without exception
            profiler.startSession();
            assertTrue(profiler.isSessionActive());
            profiler.stopSession();
        }

        @Test
        @DisplayName("getBackendName() returns backend")
        void getBackendNameReturnsBackend() {
            MockProfiler profiler = new MockProfiler(0);
            assertEquals("MockProfiler", profiler.getBackendName());
        }

        @Test
        @DisplayName("getDeviceIndex() returns device")
        void getDeviceIndexReturnsDevice() {
            MockProfiler profiler = new MockProfiler(2);
            assertEquals(2, profiler.getDeviceIndex());
        }

        @Test
        @DisplayName("estimatedOverheadNanos() returns positive value")
        void estimatedOverheadNanosReturnsPositive() {
            MockProfiler profiler = new MockProfiler(0);
            assertTrue(profiler.estimatedOverheadNanos() > 0);
        }

        @Test
        @DisplayName("close() stops active session")
        void closeStopsActiveSession() {
            MockProfiler profiler = new MockProfiler(0);
            profiler.startSession();
            assertTrue(profiler.isSessionActive());

            profiler.close();
            assertFalse(profiler.isSessionActive());
        }

        @Test
        @DisplayName("try-with-resources works correctly")
        void tryWithResourcesWorks() {
            boolean wasActive;
            try (MockProfiler profiler = new MockProfiler(0)) {
                profiler.startSession();
                wasActive = profiler.isSessionActive();
            }
            assertTrue(wasActive);
        }
    }
}
