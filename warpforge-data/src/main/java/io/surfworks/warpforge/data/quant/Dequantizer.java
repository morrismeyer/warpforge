package io.surfworks.warpforge.data.quant;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Dequantizes quantized tensors back to floating point.
 *
 * <p>Supports common quantization formats used in GGUF and GPTQ models.
 * Dequantization is performed block-by-block for efficiency.
 *
 * <p>Example usage:
 * <pre>{@code
 * Dequantizer dequant = Dequantizer.forType(QuantizationType.Q4_0);
 * float[] output = new float[numElements];
 * dequant.dequantize(quantizedData, output, numElements);
 * }</pre>
 */
public interface Dequantizer {

    /**
     * Dequantize data from a memory segment to a float array.
     *
     * @param input Quantized input data
     * @param output Float array to write dequantized values
     * @param numElements Number of elements to dequantize
     */
    void dequantize(MemorySegment input, float[] output, int numElements);

    /**
     * Dequantize data from a memory segment to another memory segment.
     *
     * @param input Quantized input data
     * @param output Memory segment to write dequantized float values
     * @param numElements Number of elements to dequantize
     */
    void dequantize(MemorySegment input, MemorySegment output, int numElements);

    /**
     * Get the quantization type this dequantizer handles.
     */
    QuantizationType type();

    /**
     * Get the block size for this quantization format.
     * Returns 1 for non-block formats.
     */
    int blockSize();

    /**
     * Get the number of bytes per block.
     */
    int bytesPerBlock();

    /**
     * Create a dequantizer for the given quantization type.
     */
    static Dequantizer forType(QuantizationType type) {
        return switch (type) {
            case F32 -> new F32Dequantizer();
            case F16 -> new F16Dequantizer();
            case BF16 -> new BF16Dequantizer();
            case Q8_0 -> new Q8_0Dequantizer();
            case Q4_0 -> new Q4_0Dequantizer();
            case Q4_1 -> new Q4_1Dequantizer();
            case INT8 -> new Int8Dequantizer();
            case INT4 -> new Int4Dequantizer();
            default -> throw new UnsupportedOperationException(
                    "Dequantization not yet implemented for: " + type);
        };
    }

    // ========== Implementations ==========

    /**
     * F32 passthrough (no dequantization needed).
     */
    final class F32Dequantizer implements Dequantizer {
        @Override
        public void dequantize(MemorySegment input, float[] output, int numElements) {
            for (int i = 0; i < numElements; i++) {
                output[i] = input.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            }
        }

        @Override
        public void dequantize(MemorySegment input, MemorySegment output, int numElements) {
            output.copyFrom(input.asSlice(0, (long) numElements * 4));
        }

        @Override
        public QuantizationType type() { return QuantizationType.F32; }

        @Override
        public int blockSize() { return 1; }

        @Override
        public int bytesPerBlock() { return 4; }
    }

    /**
     * F16 to F32 dequantization.
     */
    final class F16Dequantizer implements Dequantizer {
        @Override
        public void dequantize(MemorySegment input, float[] output, int numElements) {
            for (int i = 0; i < numElements; i++) {
                short bits = input.getAtIndex(ValueLayout.JAVA_SHORT, i);
                output[i] = halfToFloat(bits);
            }
        }

        @Override
        public void dequantize(MemorySegment input, MemorySegment output, int numElements) {
            for (int i = 0; i < numElements; i++) {
                short bits = input.getAtIndex(ValueLayout.JAVA_SHORT, i);
                output.setAtIndex(ValueLayout.JAVA_FLOAT, i, halfToFloat(bits));
            }
        }

        private static float halfToFloat(short bits) {
            int sign = (bits >> 15) & 0x1;
            int exp = (bits >> 10) & 0x1F;
            int mant = bits & 0x3FF;

            if (exp == 0) {
                // Denormalized or zero
                if (mant == 0) {
                    return sign == 0 ? 0.0f : -0.0f;
                }
                // Denormalized
                float val = mant / 1024.0f;
                val *= Math.pow(2, -14);
                return sign == 0 ? val : -val;
            } else if (exp == 31) {
                // Inf or NaN
                if (mant == 0) {
                    return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
                }
                return Float.NaN;
            }

            // Normalized
            int f32Exp = exp - 15 + 127;
            int f32Bits = (sign << 31) | (f32Exp << 23) | (mant << 13);
            return Float.intBitsToFloat(f32Bits);
        }

        @Override
        public QuantizationType type() { return QuantizationType.F16; }

        @Override
        public int blockSize() { return 1; }

        @Override
        public int bytesPerBlock() { return 2; }
    }

    /**
     * BF16 to F32 dequantization.
     */
    final class BF16Dequantizer implements Dequantizer {
        @Override
        public void dequantize(MemorySegment input, float[] output, int numElements) {
            for (int i = 0; i < numElements; i++) {
                short bits = input.getAtIndex(ValueLayout.JAVA_SHORT, i);
                output[i] = bfloatToFloat(bits);
            }
        }

        @Override
        public void dequantize(MemorySegment input, MemorySegment output, int numElements) {
            for (int i = 0; i < numElements; i++) {
                short bits = input.getAtIndex(ValueLayout.JAVA_SHORT, i);
                output.setAtIndex(ValueLayout.JAVA_FLOAT, i, bfloatToFloat(bits));
            }
        }

        private static float bfloatToFloat(short bits) {
            // BF16 is just the top 16 bits of F32
            int f32Bits = (bits & 0xFFFF) << 16;
            return Float.intBitsToFloat(f32Bits);
        }

        @Override
        public QuantizationType type() { return QuantizationType.BF16; }

        @Override
        public int blockSize() { return 1; }

        @Override
        public int bytesPerBlock() { return 2; }
    }

    /**
     * Q8_0 dequantization (8-bit with per-block scale).
     * Block format: 2 bytes scale (fp16) + 32 bytes data (32 x int8)
     */
    final class Q8_0Dequantizer implements Dequantizer {
        private static final int BLOCK_SIZE = 32;
        private static final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE; // scale + data

        @Override
        public void dequantize(MemorySegment input, float[] output, int numElements) {
            int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int outIdx = 0;

            for (int block = 0; block < numBlocks; block++) {
                long blockOffset = (long) block * BYTES_PER_BLOCK;

                // Read scale (fp16)
                short scaleBits = input.get(ValueLayout.JAVA_SHORT, blockOffset);
                float scale = F16Dequantizer.halfToFloat(scaleBits);

                // Read and dequantize 32 int8 values
                for (int i = 0; i < BLOCK_SIZE && outIdx < numElements; i++) {
                    byte q = input.get(ValueLayout.JAVA_BYTE, blockOffset + 2 + i);
                    output[outIdx++] = q * scale;
                }
            }
        }

        @Override
        public void dequantize(MemorySegment input, MemorySegment output, int numElements) {
            float[] temp = new float[numElements];
            dequantize(input, temp, numElements);
            for (int i = 0; i < numElements; i++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, i, temp[i]);
            }
        }

        @Override
        public QuantizationType type() { return QuantizationType.Q8_0; }

        @Override
        public int blockSize() { return BLOCK_SIZE; }

        @Override
        public int bytesPerBlock() { return BYTES_PER_BLOCK; }
    }

    /**
     * Q4_0 dequantization (4-bit with per-block scale).
     * Block format: 2 bytes scale (fp16) + 16 bytes data (32 x 4-bit)
     */
    final class Q4_0Dequantizer implements Dequantizer {
        private static final int BLOCK_SIZE = 32;
        private static final int BYTES_PER_BLOCK = 2 + 16; // scale + data

        @Override
        public void dequantize(MemorySegment input, float[] output, int numElements) {
            int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int outIdx = 0;

            for (int block = 0; block < numBlocks; block++) {
                long blockOffset = (long) block * BYTES_PER_BLOCK;

                // Read scale (fp16)
                short scaleBits = input.get(ValueLayout.JAVA_SHORT, blockOffset);
                float scale = F16Dequantizer.halfToFloat(scaleBits);

                // Read and dequantize 32 4-bit values (16 bytes)
                for (int i = 0; i < 16 && outIdx < numElements; i++) {
                    byte packed = input.get(ValueLayout.JAVA_BYTE, blockOffset + 2 + i);

                    // Lower 4 bits (sign-extended from 4-bit to int)
                    int q0 = (packed & 0x0F) - 8;
                    if (outIdx < numElements) {
                        output[outIdx++] = q0 * scale;
                    }

                    // Upper 4 bits
                    int q1 = ((packed >> 4) & 0x0F) - 8;
                    if (outIdx < numElements) {
                        output[outIdx++] = q1 * scale;
                    }
                }
            }
        }

        @Override
        public void dequantize(MemorySegment input, MemorySegment output, int numElements) {
            float[] temp = new float[numElements];
            dequantize(input, temp, numElements);
            for (int i = 0; i < numElements; i++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, i, temp[i]);
            }
        }

        @Override
        public QuantizationType type() { return QuantizationType.Q4_0; }

        @Override
        public int blockSize() { return BLOCK_SIZE; }

        @Override
        public int bytesPerBlock() { return BYTES_PER_BLOCK; }
    }

    /**
     * Q4_1 dequantization (4-bit with per-block scale and offset).
     * Block format: 2 bytes scale (fp16) + 2 bytes min (fp16) + 16 bytes data
     */
    final class Q4_1Dequantizer implements Dequantizer {
        private static final int BLOCK_SIZE = 32;
        private static final int BYTES_PER_BLOCK = 2 + 2 + 16; // scale + min + data

        @Override
        public void dequantize(MemorySegment input, float[] output, int numElements) {
            int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int outIdx = 0;

            for (int block = 0; block < numBlocks; block++) {
                long blockOffset = (long) block * BYTES_PER_BLOCK;

                // Read scale and min (both fp16)
                short scaleBits = input.get(ValueLayout.JAVA_SHORT, blockOffset);
                short minBits = input.get(ValueLayout.JAVA_SHORT, blockOffset + 2);
                float scale = F16Dequantizer.halfToFloat(scaleBits);
                float min = F16Dequantizer.halfToFloat(minBits);

                // Read and dequantize 32 4-bit values (16 bytes)
                for (int i = 0; i < 16 && outIdx < numElements; i++) {
                    byte packed = input.get(ValueLayout.JAVA_BYTE, blockOffset + 4 + i);

                    // Lower 4 bits (unsigned)
                    int q0 = packed & 0x0F;
                    if (outIdx < numElements) {
                        output[outIdx++] = q0 * scale + min;
                    }

                    // Upper 4 bits
                    int q1 = (packed >> 4) & 0x0F;
                    if (outIdx < numElements) {
                        output[outIdx++] = q1 * scale + min;
                    }
                }
            }
        }

        @Override
        public void dequantize(MemorySegment input, MemorySegment output, int numElements) {
            float[] temp = new float[numElements];
            dequantize(input, temp, numElements);
            for (int i = 0; i < numElements; i++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, i, temp[i]);
            }
        }

        @Override
        public QuantizationType type() { return QuantizationType.Q4_1; }

        @Override
        public int blockSize() { return BLOCK_SIZE; }

        @Override
        public int bytesPerBlock() { return BYTES_PER_BLOCK; }
    }

    /**
     * INT8 symmetric dequantization (single scale for entire tensor).
     */
    final class Int8Dequantizer implements Dequantizer {
        private float scale = 1.0f;

        public Int8Dequantizer() {}

        public Int8Dequantizer(float scale) {
            this.scale = scale;
        }

        public void setScale(float scale) {
            this.scale = scale;
        }

        @Override
        public void dequantize(MemorySegment input, float[] output, int numElements) {
            for (int i = 0; i < numElements; i++) {
                byte q = input.get(ValueLayout.JAVA_BYTE, i);
                output[i] = q * scale;
            }
        }

        @Override
        public void dequantize(MemorySegment input, MemorySegment output, int numElements) {
            for (int i = 0; i < numElements; i++) {
                byte q = input.get(ValueLayout.JAVA_BYTE, i);
                output.setAtIndex(ValueLayout.JAVA_FLOAT, i, q * scale);
            }
        }

        @Override
        public QuantizationType type() { return QuantizationType.INT8; }

        @Override
        public int blockSize() { return 1; }

        @Override
        public int bytesPerBlock() { return 1; }
    }

    /**
     * INT4 symmetric dequantization.
     */
    final class Int4Dequantizer implements Dequantizer {
        private float scale = 1.0f;

        public Int4Dequantizer() {}

        public Int4Dequantizer(float scale) {
            this.scale = scale;
        }

        public void setScale(float scale) {
            this.scale = scale;
        }

        @Override
        public void dequantize(MemorySegment input, float[] output, int numElements) {
            int outIdx = 0;
            int numBytes = (numElements + 1) / 2;

            for (int i = 0; i < numBytes && outIdx < numElements; i++) {
                byte packed = input.get(ValueLayout.JAVA_BYTE, i);

                // Lower 4 bits (sign-extended)
                int q0 = (packed & 0x0F);
                if (q0 > 7) q0 -= 16; // Sign extend
                if (outIdx < numElements) {
                    output[outIdx++] = q0 * scale;
                }

                // Upper 4 bits
                int q1 = (packed >> 4) & 0x0F;
                if (q1 > 7) q1 -= 16;
                if (outIdx < numElements) {
                    output[outIdx++] = q1 * scale;
                }
            }
        }

        @Override
        public void dequantize(MemorySegment input, MemorySegment output, int numElements) {
            float[] temp = new float[numElements];
            dequantize(input, temp, numElements);
            for (int i = 0; i < numElements; i++) {
                output.setAtIndex(ValueLayout.JAVA_FLOAT, i, temp[i]);
            }
        }

        @Override
        public QuantizationType type() { return QuantizationType.INT4; }

        @Override
        public int blockSize() { return 2; }

        @Override
        public int bytesPerBlock() { return 1; }
    }
}
