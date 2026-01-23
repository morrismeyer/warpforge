package io.surfworks.warpforge.core.formats;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Interface for block-scaled floating-point formats.
 *
 * <p>Block-scaled formats store low-precision values alongside shared scale
 * factors to extend their effective dynamic range. This is the key technique
 * enabling practical FP4 inference.
 *
 * <p>Two major block format families:
 * <ul>
 *   <li><b>NVFP4</b> (NVIDIA Blackwell): 16-element blocks with FP8 E4M3 scales
 *       plus a global FP32 tensor scale</li>
 *   <li><b>OCP MX</b> (Open Compute): 32-element blocks with FP8 E8M0 scales</li>
 * </ul>
 */
public interface BlockFormat {

    /**
     * Number of elements per block.
     */
    int blockSize();

    /**
     * Format parameters for the element values.
     */
    FormatParameters elementFormat();

    /**
     * Format parameters for the scale factors.
     */
    FormatParameters scaleFormat();

    /**
     * Total bytes required to store count elements including scales.
     */
    long byteSize(int elementCount);

    /**
     * Encode float values to block format in a MemorySegment.
     *
     * @param source Float values to encode
     * @param dest Destination segment (must be at least byteSize(source.length))
     */
    void encode(float[] source, MemorySegment dest);

    /**
     * Decode block format from MemorySegment to float values.
     *
     * @param source Source segment in block format
     * @param dest Destination float array
     */
    void decode(MemorySegment source, float[] dest);

    /**
     * Encode float values and return a new MemorySegment.
     */
    default MemorySegment encode(float[] source, Arena arena) {
        MemorySegment dest = arena.allocate(byteSize(source.length));
        encode(source, dest);
        return dest;
    }

    /**
     * Decode and return a new float array.
     */
    default float[] decode(MemorySegment source, int elementCount) {
        float[] dest = new float[elementCount];
        decode(source, dest);
        return dest;
    }
}
