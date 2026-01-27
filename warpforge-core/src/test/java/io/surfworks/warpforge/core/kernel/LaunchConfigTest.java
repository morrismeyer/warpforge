package io.surfworks.warpforge.core.kernel;

import io.surfworks.warpforge.core.jfr.GpuKernelEvent;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit tests for LaunchConfig.
 */
@DisplayName("LaunchConfig Unit Tests")
class LaunchConfigTest {

    @Nested
    @DisplayName("Factory Methods")
    class FactoryMethods {

        @Test
        @DisplayName("of1D creates 1D configuration")
        void of1D() {
            LaunchConfig config = LaunchConfig.of1D(256, 128);

            assertEquals(256, config.gridDimX());
            assertEquals(1, config.gridDimY());
            assertEquals(1, config.gridDimZ());
            assertEquals(128, config.blockDimX());
            assertEquals(1, config.blockDimY());
            assertEquals(1, config.blockDimZ());
        }

        @Test
        @DisplayName("of2D creates 2D configuration")
        void of2D() {
            LaunchConfig config = LaunchConfig.of2D(16, 16, 32, 32);

            assertEquals(16, config.gridDimX());
            assertEquals(16, config.gridDimY());
            assertEquals(1, config.gridDimZ());
            assertEquals(32, config.blockDimX());
            assertEquals(32, config.blockDimY());
            assertEquals(1, config.blockDimZ());
        }

        @Test
        @DisplayName("withSharedMem creates configuration with shared memory")
        void withSharedMem() {
            LaunchConfig config = LaunchConfig.withSharedMem(8, 8, 8, 16, 16, 16, 49152);

            assertEquals(8, config.gridDimX());
            assertEquals(8, config.gridDimY());
            assertEquals(8, config.gridDimZ());
            assertEquals(16, config.blockDimX());
            assertEquals(16, config.blockDimY());
            assertEquals(16, config.blockDimZ());
            assertEquals(49152, config.sharedMemBytes());
        }
    }

    @Nested
    @DisplayName("Computed Properties")
    class ComputedProperties {

        @Test
        @DisplayName("totalBlocks calculates grid volume")
        void totalBlocks() {
            LaunchConfig config = new LaunchConfig(4, 5, 6, 8, 8, 8, 0);
            assertEquals(4 * 5 * 6, config.totalBlocks());
        }

        @Test
        @DisplayName("threadsPerBlock calculates block volume")
        void threadsPerBlock() {
            LaunchConfig config = new LaunchConfig(1, 1, 1, 16, 8, 4, 0);
            assertEquals(16 * 8 * 4, config.threadsPerBlock());
        }

        @Test
        @DisplayName("totalThreads calculates grid * block")
        void totalThreads() {
            // 256x256 blocks, 256 threads per block = 16M threads
            LaunchConfig config = LaunchConfig.of2D(256, 256, 16, 16);
            assertEquals(256L * 256 * 16 * 16, config.totalThreads());
        }

        @Test
        @DisplayName("totalWarps calculates threads / warpSize for NVIDIA")
        void totalWarpsNvidia() {
            // 1024 threads / 32 warp size = 32 warps
            LaunchConfig config = LaunchConfig.of1D(4, 256);
            assertEquals(32, config.totalWarps(LaunchConfig.NVIDIA_WARP_SIZE));
        }

        @Test
        @DisplayName("totalWarps calculates threads / wavefrontSize for AMD CDNA")
        void totalWarpsAmdCdna() {
            // 1024 threads / 64 wavefront size = 16 wavefronts
            LaunchConfig config = LaunchConfig.of1D(4, 256);
            assertEquals(16, config.totalWarps(LaunchConfig.AMD_CDNA_WAVEFRONT_SIZE));
        }

        @Test
        @DisplayName("totalWarps rounds up for non-divisible thread counts")
        void totalWarpsRoundsUp() {
            // 100 threads / 32 = 3.125 -> 4 warps
            LaunchConfig config = LaunchConfig.of1D(1, 100);
            assertEquals(4, config.totalWarps(32));
        }
    }

    @Nested
    @DisplayName("JFR Event Population")
    class JfrEventPopulation {

        @Test
        @DisplayName("populateEvent sets all launch config fields")
        void populateEvent() {
            LaunchConfig config = new LaunchConfig(64, 32, 2, 128, 4, 2, 8192);
            GpuKernelEvent event = new GpuKernelEvent();

            config.populateEvent(event, LaunchConfig.NVIDIA_WARP_SIZE);

            // Grid dimensions
            assertEquals(64, event.gridDimX);
            assertEquals(32, event.gridDimY);
            assertEquals(2, event.gridDimZ);

            // Block dimensions
            assertEquals(128, event.blockDimX);
            assertEquals(4, event.blockDimY);
            assertEquals(2, event.blockDimZ);

            // Computed fields
            assertEquals(64 * 32 * 2, event.totalBlocks);
            assertEquals(64L * 32 * 2 * 128 * 4 * 2, event.totalThreads);
            long expectedWarps = (event.totalThreads + 31) / 32;
            assertEquals(expectedWarps, event.totalWarps);
        }

        @Test
        @DisplayName("populateEvent uses AMD wavefront size correctly")
        void populateEventAmd() {
            LaunchConfig config = LaunchConfig.of1D(1024, 64);
            GpuKernelEvent event = new GpuKernelEvent();

            config.populateEvent(event, LaunchConfig.AMD_CDNA_WAVEFRONT_SIZE);

            assertEquals(1024 * 64, event.totalThreads);
            // 65536 threads / 64 = 1024 wavefronts
            assertEquals(1024, event.totalWarps);
        }
    }

    @Nested
    @DisplayName("String Representation")
    class StringRepresentation {

        @Test
        @DisplayName("toString shows 1D format for 1D config")
        void toString1D() {
            LaunchConfig config = LaunchConfig.of1D(256, 128);
            String str = config.toString();

            // Should be compact 1D format
            assertEquals("LaunchConfig[grid=256, block=128, threads=32768]", str);
        }

        @Test
        @DisplayName("toString shows 2D format for 2D config")
        void toString2D() {
            LaunchConfig config = LaunchConfig.of2D(16, 16, 32, 32);
            String str = config.toString();

            assertEquals("LaunchConfig[grid=16x16, block=32x32, threads=262144]", str);
        }

        @Test
        @DisplayName("toString shows 3D format for 3D config")
        void toString3D() {
            LaunchConfig config = new LaunchConfig(2, 3, 4, 8, 8, 8, 0);
            String str = config.toString();

            assertEquals("LaunchConfig[grid=2x3x4, block=8x8x8, threads=12288]", str);
        }
    }

    @Nested
    @DisplayName("Real-World Configurations")
    class RealWorldConfigurations {

        @Test
        @DisplayName("GEMM 4096x4096 configuration")
        void gemm4096() {
            // Typical GEMM: 256x256 blocks of 256 threads
            LaunchConfig config = LaunchConfig.of2D(256, 256, 16, 16);

            assertEquals(65536, config.totalBlocks());
            assertEquals(256, config.threadsPerBlock());
            assertEquals(16_777_216L, config.totalThreads());
            assertEquals(524_288L, config.totalWarps(32));
        }

        @Test
        @DisplayName("Elementwise 1M element operation")
        void elementwise1M() {
            // 1M elements with 256 threads per block = 4096 blocks
            int elements = 1_000_000;
            int blockSize = 256;
            int gridSize = (elements + blockSize - 1) / blockSize;

            LaunchConfig config = LaunchConfig.of1D(gridSize, blockSize);

            assertEquals(3907, config.totalBlocks());
            assertEquals(256, config.threadsPerBlock());
            assertEquals(1_000_192L, config.totalThreads()); // Slightly over due to padding
        }
    }
}
