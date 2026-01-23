package io.surfworks.warpforge.core.formats;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * OCP Microscaling (MX) block formats.
 *
 * <p>The OCP (Open Compute Project) MX specification defines block-scaled
 * formats for AI inference. Each format uses 32-element blocks with an
 * E8M0 (power-of-2) scale factor.
 *
 * <h2>Supported Formats</h2>
 * <ul>
 *   <li><b>MXFP4</b>: 32 × E2M1 values + 1 × E8M0 scale = 17 bytes/block</li>
 *   <li><b>MXFP6 E3M2</b>: 32 × E3M2 values + 1 × E8M0 scale = 25 bytes/block</li>
 *   <li><b>MXFP6 E2M3</b>: 32 × E2M3 values + 1 × E8M0 scale = 25 bytes/block</li>
 *   <li><b>MXFP8 E4M3</b>: 32 × E4M3 values + 1 × E8M0 scale = 33 bytes/block</li>
 *   <li><b>MXFP8 E5M2</b>: 32 × E5M2 values + 1 × E8M0 scale = 33 bytes/block</li>
 * </ul>
 *
 * <h2>Memory Layout</h2>
 * <pre>
 * For N elements (must be multiple of 32):
 *
 * [block_0_values: V bytes | scale_0: 1 byte E8M0]
 * [block_1_values: V bytes | scale_1: 1 byte E8M0]
 * ...
 *
 * where V = (32 * bitWidth) / 8
 * </pre>
 *
 * <h2>Value Reconstruction</h2>
 * <pre>
 * actual_value = 2^(scale_biased - 127) * element_value
 * </pre>
 *
 * @see <a href="https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf">
 *      OCP MX Specification</a>
 */
public final class MxBlock implements BlockFormat {

    /** MXFP4: E2M1 values with E8M0 scales. */
    public static final MxBlock MXFP4 = new MxBlock(FormatParameters.FP4_E2M1, "MXFP4");

    /** MXFP6 E3M2: wider range variant. */
    public static final MxBlock MXFP6_E3M2 = new MxBlock(FormatParameters.FP6_E3M2, "MXFP6_E3M2");

    /** MXFP6 E2M3: higher precision variant. */
    public static final MxBlock MXFP6_E2M3 = new MxBlock(FormatParameters.FP6_E2M3, "MXFP6_E2M3");

    /** MXFP8 E4M3: higher precision 8-bit. */
    public static final MxBlock MXFP8_E4M3 = new MxBlock(FormatParameters.FP8_E4M3, "MXFP8_E4M3");

    /** MXFP8 E5M2: wider range 8-bit. */
    public static final MxBlock MXFP8_E5M2 = new MxBlock(FormatParameters.FP8_E5M2, "MXFP8_E5M2");

    /** Block size: 32 elements per block (OCP MX standard). */
    public static final int BLOCK_SIZE = 32;

    private static final FormatParameters SCALE_FORMAT = FormatParameters.FP8_E8M0;

    private final FormatParameters elementFormat;
    private final String name;
    private final int valueBytesPerBlock;

    private MxBlock(FormatParameters elementFormat, String name) {
        this.elementFormat = elementFormat;
        this.name = name;
        // Calculate bytes needed for 32 values
        this.valueBytesPerBlock = (int) MiniFloat.byteSize(BLOCK_SIZE, elementFormat);
    }

    @Override
    public int blockSize() {
        return BLOCK_SIZE;
    }

    @Override
    public FormatParameters elementFormat() {
        return elementFormat;
    }

    @Override
    public FormatParameters scaleFormat() {
        return SCALE_FORMAT;
    }

    @Override
    public long byteSize(int elementCount) {
        int numBlocks = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int bytesPerBlock = valueBytesPerBlock + 1; // +1 for E8M0 scale
        return (long) numBlocks * bytesPerBlock;
    }

    @Override
    public void encode(float[] source, MemorySegment dest) {
        int count = source.length;
        int numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int bytesPerBlock = valueBytesPerBlock + 1;

        long blockOffset = 0;

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
            int start = blockIdx * BLOCK_SIZE;
            int end = Math.min(start + BLOCK_SIZE, count);

            // Find max absolute value in this block
            float maxAbs = 0;
            for (int i = start; i < end; i++) {
                float abs = Math.abs(source[i]);
                if (abs > maxAbs && Float.isFinite(source[i])) {
                    maxAbs = abs;
                }
            }

            // Compute optimal E8M0 scale (power of 2)
            // We want: maxAbs / scale ≤ maxElementValue
            int scaleExponent;
            if (maxAbs == 0) {
                scaleExponent = 0;
            } else {
                double maxElementValue = elementFormat.maxValue();
                // scale = 2^exp such that maxAbs/scale ≤ maxElementValue
                // We need scale ≥ maxAbs / maxElementValue
                // So exp ≥ log2(maxAbs / maxElementValue)
                // Use ceil to ensure we don't overflow
                scaleExponent = (int) Math.ceil(Math.log(maxAbs / maxElementValue) / Math.log(2));
            }

            // Clamp to E8M0 range: biased exponent 0-255 → unbiased -127 to +128
            scaleExponent = Math.max(-127, Math.min(128, scaleExponent));
            int biasedScale = scaleExponent + 127;

            // Calculate actual scale value
            float scale = (float) Math.scalb(1.0, scaleExponent);
            float invScale = scale > 0 ? 1.0f / scale : 0;

            // Encode values for this block
            float[] blockValues = new float[BLOCK_SIZE];
            for (int i = 0; i < BLOCK_SIZE; i++) {
                int srcIdx = start + i;
                if (srcIdx < end) {
                    blockValues[i] = source[srcIdx] * invScale;
                } else {
                    blockValues[i] = 0; // Padding
                }
            }

            // Write encoded values
            MemorySegment valueSlice = dest.asSlice(blockOffset, valueBytesPerBlock);
            MiniFloat.encodeBulk(blockValues, valueSlice, elementFormat);

            // Write scale (E8M0 - biased exponent)
            dest.set(ValueLayout.JAVA_BYTE, blockOffset + valueBytesPerBlock, (byte) biasedScale);

            blockOffset += bytesPerBlock;
        }
    }

    @Override
    public void decode(MemorySegment source, float[] dest) {
        int count = dest.length;
        int numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int bytesPerBlock = valueBytesPerBlock + 1;

        long blockOffset = 0;

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
            int start = blockIdx * BLOCK_SIZE;
            int end = Math.min(start + BLOCK_SIZE, count);

            // Read scale (E8M0)
            int biasedScale = source.get(ValueLayout.JAVA_BYTE, blockOffset + valueBytesPerBlock) & 0xFF;
            int scaleExponent = biasedScale - 127;
            float scale = (float) Math.scalb(1.0, scaleExponent);

            // Decode values
            float[] blockValues = new float[BLOCK_SIZE];
            MemorySegment valueSlice = source.asSlice(blockOffset, valueBytesPerBlock);
            MiniFloat.decodeBulk(valueSlice, blockValues, elementFormat);

            // Apply scale and copy to output
            for (int i = 0; i < BLOCK_SIZE && start + i < end; i++) {
                dest[start + i] = blockValues[i] * scale;
            }

            blockOffset += bytesPerBlock;
        }
    }

    /**
     * Get the scale exponent (unbiased) for a specific block.
     *
     * @param encoded The encoded data segment
     * @param blockIndex The 0-based block index
     * @return The unbiased exponent (-127 to +128)
     */
    public int getScaleExponent(MemorySegment encoded, int blockIndex) {
        int bytesPerBlock = valueBytesPerBlock + 1;
        long offset = (long) blockIndex * bytesPerBlock + valueBytesPerBlock;
        int biased = encoded.get(ValueLayout.JAVA_BYTE, offset) & 0xFF;
        return biased - 127;
    }

    /**
     * Get the scale value (power of 2) for a specific block.
     *
     * @param encoded The encoded data segment
     * @param blockIndex The 0-based block index
     * @return The scale value as 2^exponent
     */
    public float getScale(MemorySegment encoded, int blockIndex) {
        int exp = getScaleExponent(encoded, blockIndex);
        return (float) Math.scalb(1.0, exp);
    }

    /**
     * Calculate the number of blocks for a given element count.
     */
    public int numBlocks(int elementCount) {
        return (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    /**
     * Get the name of this MX format variant.
     */
    public String name() {
        return name;
    }

    @Override
    public String toString() {
        return name + "[blockSize=32, element=" + elementFormat.shortName() + ", scale=E8M0]";
    }

    /**
     * Create a custom MX-style block format with the given element format.
     *
     * @param elementFormat The format for element values
     * @param name A descriptive name
     * @return A new MxBlock instance
     */
    public static MxBlock custom(FormatParameters elementFormat, String name) {
        return new MxBlock(elementFormat, name);
    }
}
