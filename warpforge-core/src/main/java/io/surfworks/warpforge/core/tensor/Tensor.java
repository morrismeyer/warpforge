package io.surfworks.warpforge.core.tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * A multi-dimensional tensor backed by a MemorySegment.
 * Provides efficient element access and bulk operations.
 */
public final class Tensor implements AutoCloseable {
    private final TensorSpec spec;
    private final MemorySegment data;
    private final Arena arena;
    private final boolean ownsArena;

    private Tensor(TensorSpec spec, MemorySegment data, Arena arena, boolean ownsArena) {
        this.spec = spec;
        this.data = data;
        this.arena = arena;
        this.ownsArena = ownsArena;
    }

    // ==================== Factory Methods ====================

    /**
     * Create a zero-initialized tensor with the given shape (defaults to F32).
     */
    public static Tensor zeros(int... shape) {
        return zeros(ScalarType.F32, shape);
    }

    /**
     * Create a zero-initialized tensor with the given dtype and shape.
     */
    public static Tensor zeros(ScalarType dtype, int... shape) {
        TensorSpec spec = TensorSpec.of(dtype, shape);
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(spec.byteSize());
        segment.fill((byte) 0);
        return new Tensor(spec, segment, arena, true);
    }

    /**
     * Create a tensor filled with a constant value.
     */
    public static Tensor full(float value, int... shape) {
        Tensor tensor = zeros(ScalarType.F32, shape);
        long count = tensor.elementCount();
        for (long i = 0; i < count; i++) {
            tensor.data.setAtIndex(ValueLayout.JAVA_FLOAT, i, value);
        }
        return tensor;
    }

    /**
     * Create a tensor from a float array (1D or flattened).
     */
    public static Tensor fromFloatArray(float[] data, int... shape) {
        TensorSpec spec = TensorSpec.of(ScalarType.F32, shape);
        if (data.length != spec.elementCount()) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " doesn't match shape " + Arrays.toString(shape) +
                " (expected " + spec.elementCount() + " elements)");
        }
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(spec.byteSize());
        MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_FLOAT, 0, data.length);
        return new Tensor(spec, segment, arena, true);
    }

    /**
     * Create a tensor from a double array.
     */
    public static Tensor fromDoubleArray(double[] data, int... shape) {
        TensorSpec spec = TensorSpec.of(ScalarType.F64, shape);
        if (data.length != spec.elementCount()) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " doesn't match shape " + Arrays.toString(shape));
        }
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(spec.byteSize());
        MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_DOUBLE, 0, data.length);
        return new Tensor(spec, segment, arena, true);
    }

    /**
     * Create a tensor from an int array.
     */
    public static Tensor fromIntArray(int[] data, int... shape) {
        TensorSpec spec = TensorSpec.of(ScalarType.I32, shape);
        if (data.length != spec.elementCount()) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " doesn't match shape " + Arrays.toString(shape));
        }
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(spec.byteSize());
        MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_INT, 0, data.length);
        return new Tensor(spec, segment, arena, true);
    }

    /**
     * Create a tensor wrapping an existing MemorySegment (does not own the arena).
     */
    public static Tensor fromMemorySegment(MemorySegment segment, TensorSpec spec) {
        if (segment.byteSize() < spec.byteSize()) {
            throw new IllegalArgumentException(
                "Segment size " + segment.byteSize() + " is smaller than tensor size " + spec.byteSize());
        }
        return new Tensor(spec, segment, null, false);
    }

    /**
     * Create a tensor wrapping an existing MemorySegment with an owned arena.
     * The arena will be closed when the tensor is closed.
     */
    public static Tensor fromMemorySegment(MemorySegment segment, TensorSpec spec, Arena arena) {
        if (segment.byteSize() < spec.byteSize()) {
            throw new IllegalArgumentException(
                "Segment size " + segment.byteSize() + " is smaller than tensor size " + spec.byteSize());
        }
        return new Tensor(spec, segment, arena, true);
    }

    /**
     * Create a tensor with a shared arena (does not own the arena).
     */
    public static Tensor allocate(TensorSpec spec, Arena arena) {
        MemorySegment segment = arena.allocate(spec.byteSize());
        segment.fill((byte) 0);
        return new Tensor(spec, segment, arena, false);
    }

    // ==================== Accessors ====================

    public TensorSpec spec() {
        return spec;
    }

    public int[] shape() {
        return spec.shape().clone();
    }

    public int rank() {
        return spec.rank();
    }

    public long elementCount() {
        return spec.elementCount();
    }

    public ScalarType dtype() {
        return spec.dtype();
    }

    public MemorySegment data() {
        return data;
    }

    // ==================== Element Access ====================

    /**
     * Get float element at multi-dimensional indices.
     */
    public float getFloat(long... indices) {
        long flatIdx = spec.flatIndex(indices);
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, flatIdx);
    }

    /**
     * Get float element at multi-dimensional int indices.
     */
    public float getFloat(int... indices) {
        long flatIdx = spec.flatIndex(indices);
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, flatIdx);
    }

    /**
     * Get float element at flat index.
     */
    public float getFloatFlat(long index) {
        return data.getAtIndex(ValueLayout.JAVA_FLOAT, index);
    }

    /**
     * Set float element at multi-dimensional indices.
     */
    public void setFloat(float value, long... indices) {
        long flatIdx = spec.flatIndex(indices);
        data.setAtIndex(ValueLayout.JAVA_FLOAT, flatIdx, value);
    }

    /**
     * Set float element at multi-dimensional int indices.
     */
    public void setFloat(float value, int... indices) {
        long flatIdx = spec.flatIndex(indices);
        data.setAtIndex(ValueLayout.JAVA_FLOAT, flatIdx, value);
    }

    /**
     * Set float element at flat index.
     */
    public void setFloatFlat(long index, float value) {
        data.setAtIndex(ValueLayout.JAVA_FLOAT, index, value);
    }

    /**
     * Get double element at multi-dimensional indices.
     */
    public double getDouble(long... indices) {
        long flatIdx = spec.flatIndex(indices);
        return data.getAtIndex(ValueLayout.JAVA_DOUBLE, flatIdx);
    }

    /**
     * Set double element at multi-dimensional indices.
     */
    public void setDouble(double value, long... indices) {
        long flatIdx = spec.flatIndex(indices);
        data.setAtIndex(ValueLayout.JAVA_DOUBLE, flatIdx, value);
    }

    /**
     * Get int element at multi-dimensional indices.
     */
    public int getInt(long... indices) {
        long flatIdx = spec.flatIndex(indices);
        return data.getAtIndex(ValueLayout.JAVA_INT, flatIdx);
    }

    /**
     * Set int element at multi-dimensional indices.
     */
    public void setInt(int value, long... indices) {
        long flatIdx = spec.flatIndex(indices);
        data.setAtIndex(ValueLayout.JAVA_INT, flatIdx, value);
    }

    // ==================== Bulk Operations ====================

    /**
     * Copy tensor data to a float array.
     */
    public float[] toFloatArray() {
        int length = (int) elementCount();
        float[] result = new float[length];
        MemorySegment.copy(data, ValueLayout.JAVA_FLOAT, 0, result, 0, length);
        return result;
    }

    /**
     * Copy tensor data to a double array.
     */
    public double[] toDoubleArray() {
        int length = (int) elementCount();
        double[] result = new double[length];
        MemorySegment.copy(data, ValueLayout.JAVA_DOUBLE, 0, result, 0, length);
        return result;
    }

    /**
     * Copy tensor data to an int array.
     */
    public int[] toIntArray() {
        int length = (int) elementCount();
        int[] result = new int[length];
        MemorySegment.copy(data, ValueLayout.JAVA_INT, 0, result, 0, length);
        return result;
    }

    /**
     * Copy data from a float array into this tensor.
     */
    public void copyFrom(float[] source) {
        if (source.length != elementCount()) {
            throw new IllegalArgumentException(
                "Source length " + source.length + " doesn't match tensor size " + elementCount());
        }
        MemorySegment.copy(source, 0, data, ValueLayout.JAVA_FLOAT, 0, source.length);
    }

    /**
     * Copy data from a double array into this tensor.
     */
    public void copyFrom(double[] source) {
        if (source.length != elementCount()) {
            throw new IllegalArgumentException(
                "Source length " + source.length + " doesn't match tensor size " + elementCount());
        }
        MemorySegment.copy(source, 0, data, ValueLayout.JAVA_DOUBLE, 0, source.length);
    }

    /**
     * Create a deep copy of this tensor.
     */
    public Tensor copy() {
        Arena newArena = Arena.ofConfined();
        MemorySegment newSegment = newArena.allocate(spec.byteSize());
        MemorySegment.copy(data, 0, newSegment, 0, spec.byteSize());
        return new Tensor(spec, newSegment, newArena, true);
    }

    /**
     * Create a view of this tensor with a new shape (must have same element count).
     */
    public Tensor reshape(int... newShape) {
        TensorSpec newSpec = TensorSpec.of(spec.dtype(), newShape);
        if (newSpec.elementCount() != spec.elementCount()) {
            throw new IllegalArgumentException(
                "Cannot reshape from " + spec.elementCount() + " to " + newSpec.elementCount() + " elements");
        }
        // Return a view (shares data, does not own arena)
        return new Tensor(newSpec, data, arena, false);
    }

    // ==================== Lifecycle ====================

    @Override
    public void close() {
        if (ownsArena && arena != null) {
            arena.close();
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor[shape=").append(Arrays.toString(spec.shape()));
        sb.append(", dtype=").append(spec.dtype());
        sb.append(", elements=").append(elementCount());
        if (elementCount() <= 10 && spec.dtype() == ScalarType.F32) {
            sb.append(", data=").append(Arrays.toString(toFloatArray()));
        }
        sb.append("]");
        return sb.toString();
    }
}
