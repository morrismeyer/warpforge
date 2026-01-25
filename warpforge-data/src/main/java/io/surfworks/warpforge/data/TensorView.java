package io.surfworks.warpforge.data;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * A view over tensor data backed by a MemorySegment.
 * Provides zero-copy access to tensor data from SafeTensors or other formats.
 *
 * <p>This class does not own the underlying memory - the parent container
 * (SafeTensors, GGUF, etc.) manages the Arena lifecycle.
 */
public final class TensorView {

    private final MemorySegment data;
    private final TensorInfo info;
    private final long[] strides;

    public TensorView(MemorySegment data, TensorInfo info) {
        this.data = data;
        this.info = info;
        this.strides = computeStrides(info.shape(), info.dtype().byteSize());
    }

    /**
     * Raw access to the underlying memory segment.
     * Use this for bulk operations or passing to backends.
     */
    public MemorySegment data() {
        return data;
    }

    /**
     * Tensor metadata.
     */
    public TensorInfo info() {
        return info;
    }

    /**
     * Tensor name.
     */
    public String name() {
        return info.name();
    }

    /**
     * Data type.
     */
    public DType dtype() {
        return info.dtype();
    }

    /**
     * Shape dimensions.
     */
    public long[] shape() {
        return info.shape().clone();
    }

    /**
     * Byte size of the tensor data.
     */
    public long byteSize() {
        return data.byteSize();
    }

    /**
     * Get a float value at the given indices.
     * Handles F32, F16, and BF16 types.
     */
    public float getFloat(long... indices) {
        long offset = flatOffset(indices);
        return switch (info.dtype()) {
            case F32 -> data.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
            case F16 -> DType.f16ToFloat(data.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset));
            case BF16 -> DType.bf16ToFloat(data.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset));
            default -> throw new UnsupportedOperationException("Cannot get float from " + info.dtype());
        };
    }

    /**
     * Get a double value at the given indices.
     */
    public double getDouble(long... indices) {
        long offset = flatOffset(indices);
        return switch (info.dtype()) {
            case F64 -> data.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset);
            case F32 -> data.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
            default -> throw new UnsupportedOperationException("Cannot get double from " + info.dtype());
        };
    }

    /**
     * Get an int value at the given indices.
     */
    public int getInt(long... indices) {
        long offset = flatOffset(indices);
        return switch (info.dtype()) {
            case I32, U32 -> data.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
            case I16, U16 -> data.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
            case I8, U8 -> data.get(ValueLayout.JAVA_BYTE, offset);
            default -> throw new UnsupportedOperationException("Cannot get int from " + info.dtype());
        };
    }

    /**
     * Get a long value at the given indices.
     */
    public long getLong(long... indices) {
        long offset = flatOffset(indices);
        return switch (info.dtype()) {
            case I64, U64 -> data.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
            case I32, U32 -> data.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
            default -> throw new UnsupportedOperationException("Cannot get long from " + info.dtype());
        };
    }

    /**
     * Get float at a flat index (for 1D iteration).
     */
    public float getFloatFlat(long index) {
        long offset = index * info.dtype().byteSize();
        return switch (info.dtype()) {
            case F32 -> data.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
            case F16 -> DType.f16ToFloat(data.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset));
            case BF16 -> DType.bf16ToFloat(data.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset));
            default -> throw new UnsupportedOperationException("Cannot get float from " + info.dtype());
        };
    }

    /**
     * Copy tensor data to a float array.
     * Performs conversion from F16/BF16 if needed.
     */
    public float[] toFloatArray() {
        long count = info.elementCount();
        if (count > Integer.MAX_VALUE) {
            throw new IllegalStateException("Tensor too large for array: " + count);
        }
        float[] result = new float[(int) count];
        for (int i = 0; i < result.length; i++) {
            result[i] = getFloatFlat(i);
        }
        return result;
    }

    /**
     * Create a slice of this tensor along the first dimension.
     */
    public TensorView slice(long start, long end) {
        if (info.shape().length == 0) {
            throw new IllegalStateException("Cannot slice a scalar tensor");
        }
        long dim0 = info.shape()[0];
        if (start < 0 || end > dim0 || start >= end) {
            throw new IndexOutOfBoundsException(
                    String.format("Invalid slice [%d, %d) for dimension %d", start, end, dim0));
        }

        long[] newShape = info.shape().clone();
        newShape[0] = end - start;

        long elementSize = info.dtype().byteSize();
        long sliceElements = 1;
        for (int i = 1; i < info.shape().length; i++) {
            sliceElements *= info.shape()[i];
        }

        long startOffset = start * sliceElements * elementSize;
        long sliceSize = (end - start) * sliceElements * elementSize;

        MemorySegment sliceData = data.asSlice(startOffset, sliceSize);
        TensorInfo sliceInfo = new TensorInfo(
                info.name() + "[" + start + ":" + end + "]",
                info.dtype(),
                newShape,
                0,
                sliceSize
        );

        return new TensorView(sliceData, sliceInfo);
    }

    private long flatOffset(long... indices) {
        if (indices.length != info.shape().length) {
            throw new IllegalArgumentException(
                    String.format("Expected %d indices, got %d", info.shape().length, indices.length));
        }
        long offset = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= info.shape()[i]) {
                throw new IndexOutOfBoundsException(
                        String.format("Index %d out of bounds for dimension %d (size %d)",
                                indices[i], i, info.shape()[i]));
            }
            offset += indices[i] * strides[i];
        }
        return offset;
    }

    private static long[] computeStrides(long[] shape, int elementSize) {
        if (shape.length == 0) {
            return new long[0];
        }
        long[] strides = new long[shape.length];
        strides[shape.length - 1] = elementSize;
        for (int i = shape.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    @Override
    public String toString() {
        return String.format("TensorView[%s: %s %s, %d bytes]",
                info.name(), info.dtype(), Arrays.toString(info.shape()), data.byteSize());
    }
}
