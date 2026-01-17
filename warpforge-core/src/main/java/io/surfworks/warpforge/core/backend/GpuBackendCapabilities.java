package io.surfworks.warpforge.core.backend;

import io.surfworks.warpforge.core.tensor.ScalarType;

import java.util.Set;

/**
 * Extended capabilities for GPU backends.
 * Includes GPU-specific features like GPUDirect RDMA and device memory.
 */
public record GpuBackendCapabilities(
    BackendCapabilities base,
    long deviceMemoryBytes,
    int computeUnits,
    boolean supportsGpuDirectRdma,
    boolean supportsFp16,
    boolean supportsBf16,
    boolean supportsTensorCores,
    int maxThreadsPerBlock,
    int maxSharedMemoryPerBlock
) {

    /**
     * Create capabilities for an NVIDIA GPU.
     *
     * @param deviceMemoryBytes GPU memory in bytes
     * @param smCount Number of streaming multiprocessors
     * @param supportsGpuDirect Whether GPUDirect RDMA is supported
     * @param hasTensorCores Whether tensor cores are available
     */
    public static GpuBackendCapabilities nvidia(
            long deviceMemoryBytes,
            int smCount,
            boolean supportsGpuDirect,
            boolean hasTensorCores) {

        BackendCapabilities base = new BackendCapabilities(
            Set.of(ScalarType.F32, ScalarType.F64, ScalarType.F16, ScalarType.BF16,
                   ScalarType.I32, ScalarType.I64, ScalarType.I16, ScalarType.I8),
            true,  // Vector ops
            true,  // Async support
            8,     // Max rank
            Long.MAX_VALUE // Max elements (limited by device memory)
        );

        return new GpuBackendCapabilities(
            base,
            deviceMemoryBytes,
            smCount,
            supportsGpuDirect,
            true,  // FP16
            true,  // BF16 (Ampere+)
            hasTensorCores,
            1024,  // Max threads per block
            49152  // 48KB shared memory (typical)
        );
    }

    /**
     * Create capabilities for an AMD GPU.
     *
     * @param deviceMemoryBytes GPU memory in bytes
     * @param cuCount Number of compute units
     * @param supportsRocmRdma Whether ROCm RDMA is supported
     */
    public static GpuBackendCapabilities amd(
            long deviceMemoryBytes,
            int cuCount,
            boolean supportsRocmRdma) {

        BackendCapabilities base = new BackendCapabilities(
            Set.of(ScalarType.F32, ScalarType.F64, ScalarType.F16, ScalarType.BF16,
                   ScalarType.I32, ScalarType.I64, ScalarType.I16, ScalarType.I8),
            true,  // Vector ops
            true,  // Async support
            8,     // Max rank
            Long.MAX_VALUE
        );

        return new GpuBackendCapabilities(
            base,
            deviceMemoryBytes,
            cuCount,
            supportsRocmRdma,
            true,  // FP16
            true,  // BF16 (CDNA)
            false, // Matrix cores (would need separate flag)
            1024,  // Max threads per workgroup
            65536  // 64KB LDS (typical)
        );
    }

    /**
     * Builder for custom GPU capabilities.
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private BackendCapabilities base = BackendCapabilities.cpu();
        private long deviceMemory = 0;
        private int computeUnits = 0;
        private boolean gpuDirectRdma = false;
        private boolean fp16 = false;
        private boolean bf16 = false;
        private boolean tensorCores = false;
        private int maxThreads = 1024;
        private int maxSharedMem = 49152;

        public Builder base(BackendCapabilities base) {
            this.base = base;
            return this;
        }

        public Builder deviceMemoryBytes(long bytes) {
            this.deviceMemory = bytes;
            return this;
        }

        public Builder computeUnits(int units) {
            this.computeUnits = units;
            return this;
        }

        public Builder supportsGpuDirectRdma(boolean supports) {
            this.gpuDirectRdma = supports;
            return this;
        }

        public Builder supportsFp16(boolean supports) {
            this.fp16 = supports;
            return this;
        }

        public Builder supportsBf16(boolean supports) {
            this.bf16 = supports;
            return this;
        }

        public Builder supportsTensorCores(boolean supports) {
            this.tensorCores = supports;
            return this;
        }

        public Builder maxThreadsPerBlock(int max) {
            this.maxThreads = max;
            return this;
        }

        public Builder maxSharedMemoryPerBlock(int max) {
            this.maxSharedMem = max;
            return this;
        }

        public GpuBackendCapabilities build() {
            return new GpuBackendCapabilities(
                base, deviceMemory, computeUnits, gpuDirectRdma,
                fp16, bf16, tensorCores, maxThreads, maxSharedMem
            );
        }
    }
}
