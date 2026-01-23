package io.surfworks.warpforge.core.formats;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * NVIDIA FP4 (NVFP4) block format for Blackwell GPUs.
 *
 * <p>NVFP4 uses a dual-level scaling scheme:
 * <ul>
 *   <li><b>Block scale</b>: FP8 E4M3 per 16 elements</li>
 *   <li><b>Tensor scale</b>: FP32 global scale for the entire tensor</li>
 * </ul>
 *
 * <p>This results in an effective 4.5 bits per value when the tensor scale
 * is amortized across many elements.
 *
 * <h2>Memory Layout</h2>
 * <pre>
 * For N elements (must be multiple of 16):
 *
 * [tensor_scale: 4 bytes FP32]
 * [block_0: 8 bytes FP4 (16 values packed) | scale_0: 1 byte E4M3]
 * [block_1: 8 bytes FP4 (16 values packed) | scale_1: 1 byte E4M3]
 * ...
 * [block_k: 8 bytes FP4 (16 values packed) | scale_k: 1 byte E4M3]
 *
 * Total: 4 + (N/16) * 9 bytes
 * Effective: 4.5 bits/element for large tensors
 * </pre>
 *
 * <h2>Value Reconstruction</h2>
 * <pre>
 * actual_value = tensor_scale * block_scale[i/16] * fp4_value[i]
 * </pre>
 *
 * @see <a href="https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/">
 *      NVIDIA NVFP4 Blog Post</a>
 */
public final class NvFp4Block implements BlockFormat {

    /** Singleton instance for standard NVFP4 configuration. */
    public static final NvFp4Block INSTANCE = new NvFp4Block();

    /** Block size: 16 elements per block. */
    public static final int BLOCK_SIZE = 16;

    /** Bytes per block: 8 bytes for values + 1 byte for scale = 9 bytes. */
    private static final int BYTES_PER_BLOCK = 9;

    /** Bytes for tensor-level scale. */
    private static final int TENSOR_SCALE_BYTES = 4;

    private static final FormatParameters ELEMENT_FORMAT = FormatParameters.FP4_E2M1;
    private static final FormatParameters SCALE_FORMAT = FormatParameters.FP8_E4M3;

    private NvFp4Block() {
    }

    @Override
    public int blockSize() {
        return BLOCK_SIZE;
    }

    @Override
    public FormatParameters elementFormat() {
        return ELEMENT_FORMAT;
    }

    @Override
    public FormatParameters scaleFormat() {
        return SCALE_FORMAT;
    }

    @Override
    public long byteSize(int elementCount) {
        int numBlocks = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return TENSOR_SCALE_BYTES + (long) numBlocks * BYTES_PER_BLOCK;
    }

    @Override
    public void encode(float[] source, MemorySegment dest) {
        int count = source.length;
        int numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Step 1: Compute optimal tensor-level scale
        // Find the maximum absolute value across all elements
        float maxAbs = 0;
        for (float v : source) {
            float abs = Math.abs(v);
            if (abs > maxAbs && Float.isFinite(v)) {
                maxAbs = abs;
            }
        }

        // Tensor scale normalizes the max value to fit in the FP4 range after block scaling
        // FP4 E2M1 max value is 6.0, E4M3 max scale is ~448
        // We want: max_value / (tensor_scale * max_block_scale * 6.0) ≈ 1
        float tensorScale;
        if (maxAbs == 0) {
            tensorScale = 1.0f;
        } else {
            // Target: after tensor scale, values should be in range where block scales work well
            float fp4Max = 6.0f;
            float e4m3Max = (float) SCALE_FORMAT.maxValue();
            tensorScale = maxAbs / (fp4Max * e4m3Max);
            // Clamp to avoid very small tensor scales
            if (tensorScale < 1e-10f) tensorScale = 1e-10f;
        }

        // Store tensor scale
        dest.set(ValueLayout.JAVA_FLOAT, 0, tensorScale);

        // Step 2: Process each block
        long blockOffset = TENSOR_SCALE_BYTES;
        float invTensorScale = 1.0f / tensorScale;

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
            int start = blockIdx * BLOCK_SIZE;
            int end = Math.min(start + BLOCK_SIZE, count);

            // Find max absolute value in this block (after tensor scaling)
            float blockMaxAbs = 0;
            for (int i = start; i < end; i++) {
                float scaled = source[i] * invTensorScale;
                float abs = Math.abs(scaled);
                if (abs > blockMaxAbs && Float.isFinite(scaled)) {
                    blockMaxAbs = abs;
                }
            }

            // Compute block scale to normalize values to FP4 range
            float blockScale;
            if (blockMaxAbs == 0) {
                blockScale = 1.0f;
            } else {
                float fp4Max = 6.0f;
                blockScale = blockMaxAbs / fp4Max;
                if (blockScale < 1e-10f) blockScale = 1e-10f;
            }

            // Encode block scale as E4M3
            int blockScaleBits = MiniFloat.encode(blockScale, SCALE_FORMAT);

            // Re-decode to get actual quantized block scale
            float actualBlockScale = MiniFloat.decodeToFloat(blockScaleBits, SCALE_FORMAT);
            float invBlockScale = actualBlockScale > 0 ? 1.0f / actualBlockScale : 0;

            // Encode FP4 values (2 per byte, little-endian nibble order)
            for (int i = 0; i < BLOCK_SIZE / 2; i++) {
                int idx0 = start + i * 2;
                int idx1 = start + i * 2 + 1;

                float v0 = (idx0 < end) ? source[idx0] * invTensorScale * invBlockScale : 0;
                float v1 = (idx1 < end) ? source[idx1] * invTensorScale * invBlockScale : 0;

                int bits0 = MiniFloat.encode(v0, ELEMENT_FORMAT) & 0x0F;
                int bits1 = MiniFloat.encode(v1, ELEMENT_FORMAT) & 0x0F;

                // Pack: low nibble = first value, high nibble = second value
                byte packed = (byte) (bits0 | (bits1 << 4));
                dest.set(ValueLayout.JAVA_BYTE, blockOffset + i, packed);
            }

            // Store block scale after the 8 bytes of FP4 data
            dest.set(ValueLayout.JAVA_BYTE, blockOffset + 8, (byte) blockScaleBits);

            blockOffset += BYTES_PER_BLOCK;
        }
    }

    @Override
    public void decode(MemorySegment source, float[] dest) {
        int count = dest.length;
        int numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Read tensor scale
        float tensorScale = source.get(ValueLayout.JAVA_FLOAT, 0);

        // Process each block
        long blockOffset = TENSOR_SCALE_BYTES;

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
            int start = blockIdx * BLOCK_SIZE;
            int end = Math.min(start + BLOCK_SIZE, count);

            // Read block scale
            int blockScaleBits = source.get(ValueLayout.JAVA_BYTE, blockOffset + 8) & 0xFF;
            float blockScale = MiniFloat.decodeToFloat(blockScaleBits, SCALE_FORMAT);

            // Combined scale
            float combinedScale = tensorScale * blockScale;

            // Decode FP4 values
            for (int i = 0; i < BLOCK_SIZE / 2; i++) {
                int idx0 = start + i * 2;
                int idx1 = start + i * 2 + 1;

                byte packed = source.get(ValueLayout.JAVA_BYTE, blockOffset + i);
                int bits0 = packed & 0x0F;
                int bits1 = (packed >> 4) & 0x0F;

                if (idx0 < end) {
                    dest[idx0] = MiniFloat.decodeToFloat(bits0, ELEMENT_FORMAT) * combinedScale;
                }
                if (idx1 < end) {
                    dest[idx1] = MiniFloat.decodeToFloat(bits1, ELEMENT_FORMAT) * combinedScale;
                }
            }

            blockOffset += BYTES_PER_BLOCK;
        }
    }

    /**
     * Get the tensor-level scale from an encoded segment.
     */
    public float getTensorScale(MemorySegment encoded) {
        return encoded.get(ValueLayout.JAVA_FLOAT, 0);
    }

    /**
     * Get a block-level scale from an encoded segment.
     *
     * @param encoded The encoded data segment
     * @param blockIndex The 0-based block index
     * @return The block scale as a float
     */
    public float getBlockScale(MemorySegment encoded, int blockIndex) {
        long offset = TENSOR_SCALE_BYTES + (long) blockIndex * BYTES_PER_BLOCK + 8;
        int bits = encoded.get(ValueLayout.JAVA_BYTE, offset) & 0xFF;
        return MiniFloat.decodeToFloat(bits, SCALE_FORMAT);
    }

    /**
     * Calculate the number of blocks for a given element count.
     */
    public int numBlocks(int elementCount) {
        return (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    @Override
    public String toString() {
        return "NVFP4[blockSize=16, element=E2M1, scale=E4M3, tensorScale=FP32]";
    }
}
