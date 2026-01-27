package io.surfworks.warpforge.core.kernel;

import io.surfworks.warpforge.core.jfr.GpuKernelEvent;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for OccupancyCalculator.
 */
@DisplayName("OccupancyCalculator Unit Tests")
class OccupancyCalculatorTest {

    // RTX 4090 specs (Ada Lovelace)
    private static final int RTX4090_SM_COUNT = 128;
    private static final int RTX4090_MAX_WARPS_PER_SM = 48;
    private static final int RTX4090_REGISTERS_PER_SM = 65536;
    private static final int RTX4090_SHARED_MEM_PER_SM = 102400; // 100KB
    private static final int RTX4090_MAX_BLOCKS_PER_SM = 24;

    // MI300X specs (CDNA3)
    private static final int MI300X_CU_COUNT = 304;
    private static final int MI300X_MAX_WAVES_PER_CU = 32;
    private static final int MI300X_VGPRS_PER_CU = 65536;
    private static final int MI300X_LDS_PER_CU = 65536;
    private static final int MI300X_MAX_WORKGROUPS_PER_CU = 32;

    @Nested
    @DisplayName("Factory Methods")
    class FactoryMethods {

        @Test
        @DisplayName("forNvidiaDevice creates calculator with warp size 32")
        void forNvidiaDevice() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            assertEquals(128, calc.smCount());
            assertEquals(48, calc.maxWarpsPerSM());
            assertEquals(32, calc.warpSize());
        }

        @Test
        @DisplayName("forAmdCdnaDevice creates calculator with wavefront size 64")
        void forAmdCdnaDevice() {
            OccupancyCalculator calc = OccupancyCalculator.forAmdCdnaDevice(
                MI300X_CU_COUNT,
                MI300X_MAX_WAVES_PER_CU,
                MI300X_VGPRS_PER_CU,
                MI300X_LDS_PER_CU,
                MI300X_MAX_WORKGROUPS_PER_CU
            );

            assertEquals(304, calc.smCount());
            assertEquals(32, calc.maxWarpsPerSM());
            assertEquals(64, calc.warpSize());
        }

        @Test
        @DisplayName("forAmdRdnaDevice creates calculator with wavefront size 32")
        void forAmdRdnaDevice() {
            OccupancyCalculator calc = OccupancyCalculator.forAmdRdnaDevice(
                60, // RX 7900 XTX CU count
                32,
                65536,
                65536,
                32
            );

            assertEquals(60, calc.smCount());
            assertEquals(32, calc.maxWarpsPerSM());
            assertEquals(32, calc.warpSize());
        }
    }

    @Nested
    @DisplayName("Occupancy Calculation")
    class OccupancyCalculation {

        private OccupancyCalculator createRtx4090Calculator() {
            return OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );
        }

        @Test
        @DisplayName("100% occupancy with minimal resource usage")
        void fullOccupancyMinimalResources() {
            OccupancyCalculator calc = createRtx4090Calculator();

            // Low register usage, no shared memory
            KernelAttributes attrs = KernelAttributes.of(16, 0);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 0);

            // Should achieve maximum occupancy
            assertEquals(100, info.occupancyPercent());
            assertEquals("warps", info.limitingFactor());
        }

        @Test
        @DisplayName("Register-limited occupancy")
        void registerLimitedOccupancy() {
            OccupancyCalculator calc = createRtx4090Calculator();

            // High register usage: 128 regs/thread * 32 threads/warp = 4096 regs/warp
            // 65536 regs/SM / 4096 regs/warp = 16 warps max
            // 16 warps / 48 max warps = 33%
            KernelAttributes attrs = KernelAttributes.of(128, 0);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 0);

            assertEquals(16, info.activeWarpsPerSM());
            assertEquals("registers", info.limitingFactor());
            assertEquals(33, info.occupancyPercent());
        }

        @Test
        @DisplayName("Shared memory-limited occupancy")
        void sharedMemoryLimitedOccupancy() {
            OccupancyCalculator calc = createRtx4090Calculator();

            // 48KB shared memory per block limits to 2 blocks per SM
            // With 256 threads/block = 8 warps/block
            // 2 blocks * 8 warps = 16 warps
            // 16 / 48 = 33%
            KernelAttributes attrs = KernelAttributes.of(16, 49152);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 0);

            assertEquals(16, info.activeWarpsPerSM());
            assertEquals("shared_memory", info.limitingFactor());
        }

        @Test
        @DisplayName("Block count-limited occupancy")
        void blockCountLimitedOccupancy() {
            OccupancyCalculator calc = createRtx4090Calculator();

            // 32 threads per block = 1 warp per block
            // Max 24 blocks per SM = 24 warps
            // 24 / 48 = 50%
            KernelAttributes attrs = KernelAttributes.of(16, 0);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 32, 0);

            assertEquals(24, info.activeWarpsPerSM());
            assertEquals("blocks", info.limitingFactor());
            assertEquals(50, info.occupancyPercent());
        }

        @Test
        @DisplayName("Dynamic shared memory affects occupancy")
        void dynamicSharedMemoryAffectsOccupancy() {
            OccupancyCalculator calc = createRtx4090Calculator();

            // No static shared, 48KB dynamic per block
            KernelAttributes attrs = KernelAttributes.of(16, 0);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 49152);

            assertEquals("shared_memory", info.limitingFactor());
            assertTrue(info.occupancyPercent() < 100);
        }

        @Test
        @DisplayName("Combined static and dynamic shared memory")
        void combinedSharedMemory() {
            OccupancyCalculator calc = createRtx4090Calculator();

            // 16KB static + 32KB dynamic = 48KB total
            KernelAttributes attrs = KernelAttributes.of(16, 16384);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 32768);

            assertEquals("shared_memory", info.limitingFactor());
        }
    }

    @Nested
    @DisplayName("AMD CDNA Occupancy")
    class AmdCdnaOccupancy {

        private OccupancyCalculator createMi300xCalculator() {
            return OccupancyCalculator.forAmdCdnaDevice(
                MI300X_CU_COUNT,
                MI300X_MAX_WAVES_PER_CU,
                MI300X_VGPRS_PER_CU,
                MI300X_LDS_PER_CU,
                MI300X_MAX_WORKGROUPS_PER_CU
            );
        }

        @Test
        @DisplayName("Full occupancy on MI300X")
        void fullOccupancyMi300x() {
            OccupancyCalculator calc = createMi300xCalculator();

            // Low VGPR usage, no LDS
            KernelAttributes attrs = KernelAttributes.of(16, 0);
            // 256 threads / 64 wavefront = 4 wavefronts per workgroup
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 0);

            assertEquals(100, info.occupancyPercent());
        }

        @Test
        @DisplayName("VGPR-limited occupancy on CDNA")
        void vgprLimitedCdna() {
            OccupancyCalculator calc = createMi300xCalculator();

            // High VGPR usage: 128 regs * 64 threads = 8192 VGPRs per wavefront
            // 65536 / 8192 = 8 wavefronts max
            // 8 / 32 = 25%
            KernelAttributes attrs = KernelAttributes.of(128, 0);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 0);

            assertEquals(8, info.activeWarpsPerSM());
            assertEquals("registers", info.limitingFactor());
            assertEquals(25, info.occupancyPercent());
        }
    }

    @Nested
    @DisplayName("Event Population")
    class EventPopulation {

        @Test
        @DisplayName("populateEvent sets all occupancy fields")
        void populateEventSetsAllFields() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            KernelAttributes attrs = new KernelAttributes(8192, 0, 1024, 32, 80, 80);
            LaunchConfig config = LaunchConfig.of2D(256, 256, 16, 16);
            GpuKernelEvent event = new GpuKernelEvent();

            calc.populateEvent(event, attrs, config);

            // Resource usage
            assertEquals(32, event.registersPerThread);
            assertEquals(8192, event.staticSharedMemoryBytes);
            assertEquals(0, event.dynamicSharedMemoryBytes);
            assertEquals(0, event.localMemoryPerThread);

            // Occupancy
            assertTrue(event.theoreticalOccupancyPercent > 0);
            assertTrue(event.theoreticalOccupancyPercent <= 100);
            assertTrue(event.maxActiveBlocksPerSM > 0);
            assertTrue(event.estimatedActiveSMs > 0);
            assertTrue(event.estimatedActiveWarps > 0);
            assertTrue(event.occupancyLimitingFactor != null);
        }

        @Test
        @DisplayName("populateEvent estimates active SMs correctly")
        void populateEventEstimatesActiveSMs() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            KernelAttributes attrs = KernelAttributes.of(16, 0);

            // 64 blocks total - should use 64 SMs (less than 128 available)
            LaunchConfig config = LaunchConfig.of2D(8, 8, 16, 16);
            GpuKernelEvent event = new GpuKernelEvent();

            calc.populateEvent(event, attrs, config);

            assertEquals(64, event.estimatedActiveSMs);
        }

        @Test
        @DisplayName("populateEvent caps active SMs at device limit")
        void populateEventCapsActiveSMs() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            KernelAttributes attrs = KernelAttributes.of(16, 0);

            // 65536 blocks total - capped at 128 SMs
            LaunchConfig config = LaunchConfig.of2D(256, 256, 16, 16);
            GpuKernelEvent event = new GpuKernelEvent();

            calc.populateEvent(event, attrs, config);

            assertEquals(128, event.estimatedActiveSMs);
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("Zero registers uses max warps")
        void zeroRegistersUsesMaxWarps() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            KernelAttributes attrs = KernelAttributes.of(0, 0);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 0);

            // Not limited by registers
            assertTrue(info.activeWarpsPerSM() > 0);
        }

        @Test
        @DisplayName("Zero shared memory uses max blocks")
        void zeroSharedMemoryUsesMaxBlocks() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            KernelAttributes attrs = KernelAttributes.of(16, 0);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 0);

            // Not limited by shared memory
            assertTrue(!info.limitingFactor().equals("shared_memory") || info.occupancyPercent() == 100);
        }

        @Test
        @DisplayName("Small block size yields valid occupancy")
        void smallBlockSizeValid() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            // Just 1 thread per block
            KernelAttributes attrs = KernelAttributes.of(16, 0);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 1, 0);

            assertTrue(info.occupancyPercent() > 0);
            assertTrue(info.activeWarpsPerSM() > 0);
        }

        @Test
        @DisplayName("Large block size yields valid occupancy")
        void largeBlockSizeValid() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            // 1024 threads per block
            KernelAttributes attrs = KernelAttributes.of(16, 0);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 1024, 0);

            assertTrue(info.occupancyPercent() > 0);
        }
    }

    @Nested
    @DisplayName("KernelAttributes")
    class KernelAttributesTests {

        @Test
        @DisplayName("of() factory creates minimal attributes")
        void ofFactoryCreatesMinimal() {
            KernelAttributes attrs = KernelAttributes.of(32, 8192);

            assertEquals(32, attrs.numRegs());
            assertEquals(8192, attrs.sharedSizeBytes());
            assertEquals(0, attrs.localSizeBytes());
            assertEquals(1024, attrs.maxThreadsPerBlock());
        }

        @Test
        @DisplayName("usesSharedMemory returns true when shared > 0")
        void usesSharedMemoryTrue() {
            KernelAttributes attrs = KernelAttributes.of(16, 4096);
            assertTrue(attrs.usesSharedMemory());
        }

        @Test
        @DisplayName("usesSharedMemory returns false when shared == 0")
        void usesSharedMemoryFalse() {
            KernelAttributes attrs = KernelAttributes.of(16, 0);
            assertTrue(!attrs.usesSharedMemory());
        }

        @Test
        @DisplayName("hasRegisterSpills returns true when local > 0")
        void hasRegisterSpillsTrue() {
            KernelAttributes attrs = new KernelAttributes(0, 256, 1024, 32, 0, 0);
            assertTrue(attrs.hasRegisterSpills());
        }

        @Test
        @DisplayName("hasRegisterSpills returns false when local == 0")
        void hasRegisterSpillsFalse() {
            KernelAttributes attrs = KernelAttributes.of(32, 0);
            assertTrue(!attrs.hasRegisterSpills());
        }
    }

    @Nested
    @DisplayName("Real-World Kernels")
    class RealWorldKernels {

        @Test
        @DisplayName("GEMM kernel occupancy (typical cuBLAS)")
        void gemmKernelOccupancy() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            // Typical GEMM: 64 regs, 48KB shared, 256 threads
            KernelAttributes attrs = new KernelAttributes(49152, 0, 256, 64, 80, 80);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 0);

            // Should be limited by shared memory or registers
            assertTrue(info.limitingFactor().equals("registers") ||
                       info.limitingFactor().equals("shared_memory"));
            assertTrue(info.occupancyPercent() >= 25); // At least 25%
        }

        @Test
        @DisplayName("Elementwise kernel occupancy (high occupancy)")
        void elementwiseKernelOccupancy() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            // Elementwise: low registers, no shared memory
            KernelAttributes attrs = KernelAttributes.of(16, 0);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 0);

            // Should achieve near-maximum occupancy
            assertTrue(info.occupancyPercent() >= 75);
        }

        @Test
        @DisplayName("Flash Attention kernel (high shared memory)")
        void flashAttentionKernelOccupancy() {
            OccupancyCalculator calc = OccupancyCalculator.forNvidiaDevice(
                RTX4090_SM_COUNT,
                RTX4090_MAX_WARPS_PER_SM,
                RTX4090_REGISTERS_PER_SM,
                RTX4090_SHARED_MEM_PER_SM,
                RTX4090_MAX_BLOCKS_PER_SM
            );

            // Flash Attention: moderate registers, high shared memory (100KB)
            KernelAttributes attrs = new KernelAttributes(102400, 0, 256, 48, 80, 80);
            OccupancyCalculator.OccupancyInfo info = calc.calculate(attrs, 256, 0);

            // Limited by shared memory - only 1 block per SM
            assertEquals(1, info.maxActiveBlocksPerSM());
            assertEquals("shared_memory", info.limitingFactor());
        }
    }
}
