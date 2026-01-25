package io.surfworks.warpforge.data.quant;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * View into a quantized tensor with on-demand dequantization.
 *
 * <p>Provides the same interface as TensorView but handles dequantization
 * transparently. Values are dequantized on access, or the entire tensor
 * can be dequantized at once.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Load a quantized tensor
 * QuantizedTensorView qTensor = QuantizedTensorView.of(
 *     segment, info, QuantizationType.Q4_0);
 *
 * // Access values (dequantized on the fly)
 * float value = qTensor.getFloat(0, 0);
 *
 * // Or dequantize entire tensor
 * TensorView fullTensor = qTensor.dequantize();
 * }</pre>
 */
public final class QuantizedTensorView {

    private final MemorySegment data;
    private final TensorInfo info;
    private final QuantizationType quantType;
    private final Dequantizer dequantizer;
    private final long[] shape;
    private final long[] strides;

    private QuantizedTensorView(MemorySegment data, TensorInfo info, QuantizationType quantType) {
        this.data = data;
        this.info = info;
        this.quantType = quantType;
        this.dequantizer = Dequantizer.forType(quantType);
        this.shape = info.shape();
        this.strides = computeStrides(shape);
    }

    /**
     * Create a quantized tensor view.
     *
     * @param data Raw quantized data
     * @param info Tensor metadata (shape, etc.)
     * @param quantType Quantization type
     * @return Quantized tensor view
     */
    public static QuantizedTensorView of(MemorySegment data, TensorInfo info, QuantizationType quantType) {
        return new QuantizedTensorView(data, info, quantType);
    }

    /**
     * Get tensor info.
     */
    public TensorInfo info() {
        return info;
    }

    /**
     * Get tensor shape.
     */
    public long[] shape() {
        return shape.clone();
    }

    /**
     * Get the quantization type.
     */
    public QuantizationType quantizationType() {
        return quantType;
    }

    /**
     * Get the number of elements in the tensor.
     */
    public long elementCount() {
        return info.elementCount();
    }

    /**
     * Get a single dequantized value by flat index.
     */
    public float getFloatFlat(long index) {
        // For block-based quantization, we need to dequantize the containing block
        int blockSize = dequantizer.blockSize();
        int bytesPerBlock = dequantizer.bytesPerBlock();

        if (blockSize == 1) {
            // Non-block quantization
            float[] result = new float[1];
            MemorySegment slice = data.asSlice(index * bytesPerBlock, bytesPerBlock);
            dequantizer.dequantize(slice, result, 1);
            return result[0];
        }

        // Block-based: find the block containing this index
        long blockIndex = index / blockSize;
        int offsetInBlock = (int) (index % blockSize);

        long blockOffset = blockIndex * bytesPerBlock;
        MemorySegment blockData = data.asSlice(blockOffset, bytesPerBlock);

        float[] blockValues = new float[blockSize];
        dequantizer.dequantize(blockData, blockValues, blockSize);

        return blockValues[offsetInBlock];
    }

    /**
     * Get a single dequantized value by indices.
     */
    public float getFloat(long... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                    "Expected " + shape.length + " indices, got " + indices.length);
        }

        long flatIndex = 0;
        for (int i = 0; i < indices.length; i++) {
            flatIndex += indices[i] * strides[i];
        }

        return getFloatFlat(flatIndex);
    }

    /**
     * Dequantize the entire tensor to a new TensorView.
     *
     * @param arena Arena for allocating the dequantized data
     * @return Dequantized tensor view
     */
    public TensorView dequantize(Arena arena) {
        int numElements = (int) elementCount();
        MemorySegment output = arena.allocate((long) numElements * 4);

        dequantizer.dequantize(data, output, numElements);

        TensorInfo newInfo = new TensorInfo(
                info.name(),
                DType.F32,
                shape,
                0,
                (long) numElements * 4
        );

        return new TensorView(output, newInfo);
    }

    /**
     * Dequantize the entire tensor to a float array.
     */
    public float[] dequantizeToArray() {
        int numElements = (int) elementCount();
        float[] result = new float[numElements];
        dequantizer.dequantize(data, result, numElements);
        return result;
    }

    /**
     * Get the compression ratio compared to FP32.
     */
    public double compressionRatio() {
        return quantType.compressionRatio();
    }

    /**
     * Get the raw quantized data size in bytes.
     */
    public long quantizedSizeBytes() {
        return data.byteSize();
    }

    /**
     * Get the dequantized size in bytes (FP32).
     */
    public long dequantizedSizeBytes() {
        return elementCount() * 4;
    }

    private static long[] computeStrides(long[] shape) {
        long[] strides = new long[shape.length];
        long stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
}
