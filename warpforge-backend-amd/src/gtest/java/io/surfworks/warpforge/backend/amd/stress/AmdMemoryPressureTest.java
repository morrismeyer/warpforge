package io.surfworks.warpforge.backend.amd.stress;

import io.surfworks.warpforge.backend.amd.AmdBackend;
import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.concurrency.GpuTaskScope;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Memory pressure tests for AMD GPU backend.
 *
 * <p>These tests validate behavior under memory-constrained conditions:
 * <ul>
 *   <li>Near-full GPU memory utilization</li>
 *   <li>OOM recovery and graceful degradation</li>
 *   <li>Pinned memory limits</li>
 *   <li>Memory fragmentation handling</li>
 *   <li>Async allocation under pressure</li>
 * </ul>
 *
 * <p>Tests are tagged with {@code @Tag("amd")} and {@code @Tag("stress")}.
 * Run with: {@code ./gradlew :warpforge-backend-amd:stressTest}
 */
@Tag("amd")
@Tag("stress")
@DisplayName("AMD Memory Pressure Tests")
class AmdMemoryPressureTest {

    private AmdBackend backend;

    @BeforeEach
    void setUp() {
        assumeTrue(AmdBackend.isRocmAvailable(), "ROCm not available - skipping stress test");
        backend = new AmdBackend();
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    // ==================== Memory Limits Tests ====================

    @Nested
    @DisplayName("Memory Limits")
    class MemoryLimitsTests {

        @Test
        @DisplayName("Allocate up to 80% of GPU memory")
        void allocate80PercentMemory() {
            long totalMemory = backend.totalDeviceMemory();
            long targetBytes = (long) (totalMemory * 0.8);
            long allocatedTotal = 0;
            List<Tensor> tensors = new ArrayList<>();

            // Allocate in 100MB chunks
            long chunkSize = 100 * 1024 * 1024; // 100MB
            int chunkElements = (int) (chunkSize / 4); // float32

            try {
                while (allocatedTotal < targetBytes) {
                    TensorSpec spec = TensorSpec.of(ScalarType.F32, chunkElements);
                    Tensor t = backend.allocateDevice(spec);
                    tensors.add(t);
                    allocatedTotal += chunkSize;
                }

                double percentUsed = (allocatedTotal * 100.0) / totalMemory;
                System.out.printf("Allocated %.1f%% of GPU memory (%d MB / %d MB)%n",
                    percentUsed, allocatedTotal / (1024 * 1024), totalMemory / (1024 * 1024));

                assertTrue(percentUsed >= 75, "Should be able to allocate at least 75% of GPU memory");

            } finally {
                // Tensors will be cleaned up when arena closes or by GC
                tensors.clear();
            }
        }

        @Test
        @DisplayName("OOM recovery - allocate, release, reallocate")
        void oomRecovery() {
            long totalMemory = backend.totalDeviceMemory();
            long targetBytes = (long) (totalMemory * 0.7);
            long chunkSize = 100 * 1024 * 1024; // 100MB
            int chunkElements = (int) (chunkSize / 4);

            // First allocation pass
            List<Tensor> firstPass = new ArrayList<>();
            long firstPassBytes = 0;
            while (firstPassBytes < targetBytes) {
                TensorSpec spec = TensorSpec.of(ScalarType.F32, chunkElements);
                Tensor t = backend.allocateDevice(spec);
                firstPass.add(t);
                firstPassBytes += chunkSize;
            }

            System.out.printf("First pass allocated: %d MB%n", firstPassBytes / (1024 * 1024));

            // Release all
            firstPass.clear();
            backend.synchronizeDevice();

            // Second allocation pass should succeed
            List<Tensor> secondPass = new ArrayList<>();
            long secondPassBytes = 0;
            while (secondPassBytes < targetBytes) {
                TensorSpec spec = TensorSpec.of(ScalarType.F32, chunkElements);
                Tensor t = backend.allocateDevice(spec);
                secondPass.add(t);
                secondPassBytes += chunkSize;
            }

            System.out.printf("Second pass allocated: %d MB (recovery successful)%n",
                secondPassBytes / (1024 * 1024));

            assertTrue(secondPassBytes >= targetBytes * 0.9, "Should recover at least 90% of memory");

            secondPass.clear();
        }

        @Test
        @DisplayName("Fragmentation handling - varied sizes")
        void fragmentationHandling() {
            List<Tensor> tensors = new ArrayList<>();
            int[] sizes = {1024, 4096, 16384, 65536, 262144, 1048576}; // 4KB to 4MB

            // Allocate tensors of varying sizes
            for (int round = 0; round < 10; round++) {
                for (int size : sizes) {
                    TensorSpec spec = TensorSpec.of(ScalarType.F32, size);
                    Tensor t = backend.allocateDevice(spec);
                    tensors.add(t);
                }
            }

            // Release half randomly (every other one)
            List<Tensor> remaining = new ArrayList<>();
            for (int i = 0; i < tensors.size(); i++) {
                if (i % 2 == 0) {
                    remaining.add(tensors.get(i));
                }
            }
            tensors = remaining;

            // Try to allocate a large tensor in the fragmented space
            int largeSize = 4 * 1024 * 1024; // 16MB
            TensorSpec largeSpec = TensorSpec.of(ScalarType.F32, largeSize);
            Tensor largeTensor = backend.allocateDevice(largeSpec);

            System.out.printf("Fragmentation test: allocated %d small tensors, then 16MB tensor%n",
                remaining.size());

            assertTrue(largeTensor != null, "Should handle fragmented allocation");
        }
    }

    // ==================== Pinned Memory Tests ====================

    @Nested
    @DisplayName("Pinned Memory")
    class PinnedMemoryTests {

        @Test
        @DisplayName("Allocate pinned memory up to system limits")
        void pinnedMemoryLimits() {
            List<Tensor> pinnedTensors = new ArrayList<>();
            long pinnedTotal = 0;
            long chunkSize = 64 * 1024 * 1024; // 64MB chunks
            int chunkElements = (int) (chunkSize / 4);
            int maxChunks = 32; // Cap at 2GB to avoid system issues

            try {
                for (int i = 0; i < maxChunks; i++) {
                    TensorSpec spec = TensorSpec.of(ScalarType.F32, chunkElements);
                    Tensor t = backend.allocatePinned(spec);
                    pinnedTensors.add(t);
                    pinnedTotal += chunkSize;
                }
            } catch (Exception e) {
                // Expected - hit pinned memory limit
                System.out.println("Pinned memory limit: " + e.getMessage());
            }

            System.out.printf("Pinned memory allocated: %d MB in %d chunks%n",
                pinnedTotal / (1024 * 1024), pinnedTensors.size());

            assertTrue(pinnedTotal >= 128 * 1024 * 1024, "Should allocate at least 128MB pinned");

            pinnedTensors.clear();
        }

        @Test
        @DisplayName("Pinned memory under pressure")
        void pinnedMemoryUnderPressure() throws Exception {
            int taskCount = 100;
            AtomicInteger completed = new AtomicInteger(0);
            AtomicInteger pinnedAllocations = new AtomicInteger(0);

            try (GpuTaskScope scope = GpuTaskScope.open(backend, "pinned-pressure")) {
                for (int i = 0; i < taskCount; i++) {
                    scope.forkWithStream(lease -> {
                        // Allocate pinned, transfer, release
                        TensorSpec spec = TensorSpec.of(ScalarType.F32, 65536); // 256KB
                        Tensor pinned = backend.allocatePinned(spec);
                        pinnedAllocations.incrementAndGet();

                        Tensor device = backend.copyToDeviceAsync(pinned, lease.streamHandle());
                        lease.synchronize();

                        completed.incrementAndGet();
                        return null;
                    });
                }
                scope.joinAll();
            }

            System.out.printf("Pinned pressure: %d tasks, %d pinned allocations%n",
                completed.get(), pinnedAllocations.get());

            assertTrue(completed.get() >= taskCount * 0.95, "At least 95% should complete");
        }
    }

    // ==================== Async Operations Under Pressure ====================

    @Nested
    @DisplayName("Async Under Pressure")
    class AsyncUnderPressureTests {

        @Test
        @DisplayName("Async allocations with concurrent GC")
        void asyncAllocWithGC() throws Exception {
            int rounds = 10;
            int allocsPerRound = 100;
            AtomicLong totalAllocated = new AtomicLong(0);

            for (int round = 0; round < rounds; round++) {
                List<Tensor> tensors = new ArrayList<>();

                for (int i = 0; i < allocsPerRound; i++) {
                    TensorSpec spec = TensorSpec.of(ScalarType.F32, 16384); // 64KB
                    Tensor t = backend.allocateDevice(spec);
                    tensors.add(t);
                    totalAllocated.addAndGet(16384 * 4);
                }

                // Force GC every few rounds
                if (round % 3 == 0) {
                    tensors.clear();
                    System.gc();
                    Thread.sleep(10);
                }
            }

            System.out.printf("Async alloc with GC: %d MB total allocated over %d rounds%n",
                totalAllocated.get() / (1024 * 1024), rounds);
        }

        @Test
        @DisplayName("Async free under pressure")
        void asyncFreeUnderPressure() throws Exception {
            int iterations = 50;
            AtomicInteger successfulIterations = new AtomicInteger(0);

            for (int iter = 0; iter < iterations; iter++) {
                try (GpuTaskScope scope = GpuTaskScope.open(backend, "free-pressure-" + iter)) {
                    // Allocate and immediately schedule for free
                    for (int i = 0; i < 50; i++) {
                        scope.forkWithStream(lease -> {
                            TensorSpec spec = TensorSpec.of(ScalarType.F32, 32768); // 128KB
                            Tensor device = backend.allocateDevice(spec);
                            lease.synchronize();
                            // Tensor will be collected
                            return null;
                        });
                    }
                    scope.joinAll();
                    successfulIterations.incrementAndGet();
                }
            }

            System.out.printf("Async free pressure: %d/%d iterations successful%n",
                successfulIterations.get(), iterations);

            assertTrue(successfulIterations.get() >= iterations * 0.95);
        }
    }

    // ==================== GC Interaction Tests ====================

    @Nested
    @DisplayName("GC Interaction")
    class GCInteractionTests {

        @Test
        @DisplayName("Java GC with GPU allocations")
        void javaGcWithGpuAlloc() throws Exception {
            int rounds = 20;
            AtomicInteger allocations = new AtomicInteger(0);

            for (int round = 0; round < rounds; round++) {
                // Allocate GPU memory without keeping references
                for (int i = 0; i < 100; i++) {
                    TensorSpec spec = TensorSpec.of(ScalarType.F32, 8192); // 32KB
                    backend.allocateDevice(spec); // Intentionally not stored
                    allocations.incrementAndGet();
                }

                // Let GC run
                if (round % 5 == 0) {
                    System.gc();
                    Thread.sleep(50);
                }
            }

            // Force final GC and sync
            System.gc();
            backend.synchronizeDevice();

            System.out.printf("GC interaction: %d allocations across %d rounds%n",
                allocations.get(), rounds);

            // Check that memory was recovered
            long freeMemory = backend.freeDeviceMemory();
            long totalMemory = backend.totalDeviceMemory();
            double freePercent = freeMemory * 100.0 / totalMemory;

            System.out.printf("After GC: %.1f%% memory free%n", freePercent);
            assertTrue(freePercent > 50, "Should have >50% memory free after GC");
        }

        @Test
        @DisplayName("Memory tracking accuracy")
        void memoryTrackingAccuracy() {
            long initialUsed = backend.usedDeviceMemory();

            // Allocate known amount
            int elements = 1024 * 1024; // 4MB
            List<Tensor> tensors = new ArrayList<>();
            for (int i = 0; i < 10; i++) {
                TensorSpec spec = TensorSpec.of(ScalarType.F32, elements);
                tensors.add(backend.allocateDevice(spec));
            }

            long afterAllocUsed = backend.usedDeviceMemory();
            long expectedIncrease = 10L * elements * 4; // 40MB

            System.out.printf("Memory tracking: initial=%dMB, after=%dMB, expected increase=%dMB%n",
                initialUsed / (1024 * 1024), afterAllocUsed / (1024 * 1024),
                expectedIncrease / (1024 * 1024));

            // Allow 20% margin for alignment and overhead
            long actualIncrease = afterAllocUsed - initialUsed;
            assertTrue(actualIncrease >= expectedIncrease * 0.8,
                "Tracked memory should reflect allocations");

            tensors.clear();
        }
    }
}
