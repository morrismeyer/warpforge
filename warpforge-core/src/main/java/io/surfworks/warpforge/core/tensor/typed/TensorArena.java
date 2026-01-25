package io.surfworks.warpforge.core.tensor.typed;

import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;

import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;

/**
 * Scoped memory arena for allocating multiple typed tensors.
 *
 * <p>TensorArena provides structured allocation where all tensors created within
 * the arena are freed together when the arena is closed. This is more efficient
 * than allocating individual tensors and provides clearer ownership semantics.
 *
 * <p>Example usage:
 * <pre>{@code
 * try (TensorArena arena = TensorArena.ofConfined()) {
 *     // All allocations share the arena's lifetime
 *     var a = arena.zeros(new Matrix(100, 100), F32.INSTANCE, Cpu.INSTANCE);
 *     var b = arena.zeros(new Matrix(100, 100), F32.INSTANCE, Cpu.INSTANCE);
 *     var c = arena.zeros(new Matrix(100, 100), F32.INSTANCE, Cpu.INSTANCE);
 *
 *     // Perform operations
 *     TypedOps.add(a, b);
 *     MatrixOps.matmul(a, b);
 *
 *     // Extract results before arena closes
 *     float[] results = c.underlying().toFloatArray();
 * } // All tensors freed here
 * }</pre>
 *
 * <p>Arena types:
 * <ul>
 *   <li>{@code ofConfined()} - Single-threaded access, fastest
 *   <li>{@code ofShared()} - Multi-threaded access allowed
 *   <li>{@code ofAuto()} - Automatically managed (GC-based cleanup)
 * </ul>
 */
public final class TensorArena implements AutoCloseable {

    private final Arena underlying;
    private final List<TypedTensor<?, ?, ?>> allocatedTensors;
    private boolean closed;

    private TensorArena(Arena underlying) {
        this.underlying = underlying;
        this.allocatedTensors = new ArrayList<>();
        this.closed = false;
    }

    // ==================== Factory Methods ====================

    /**
     * Creates a confined arena for single-threaded use.
     *
     * <p>Confined arenas are the most efficient but can only be accessed
     * from the thread that created them.
     *
     * @return a new confined tensor arena
     */
    public static TensorArena ofConfined() {
        return new TensorArena(Arena.ofConfined());
    }

    /**
     * Creates a shared arena for multi-threaded use.
     *
     * <p>Shared arenas can be accessed from any thread but have
     * slightly higher overhead.
     *
     * @return a new shared tensor arena
     */
    public static TensorArena ofShared() {
        return new TensorArena(Arena.ofShared());
    }

    /**
     * Creates an auto-managed arena.
     *
     * <p>Auto arenas are cleaned up by the garbage collector, similar
     * to regular Java objects. Use when explicit scope management
     * is inconvenient.
     *
     * @return a new auto-managed tensor arena
     */
    public static TensorArena ofAuto() {
        return new TensorArena(Arena.ofAuto());
    }

    // ==================== Allocation Methods ====================

    /**
     * Allocates a zero-initialized tensor in this arena.
     *
     * @param shape the shape phantom type
     * @param dtype the dtype phantom type
     * @param device the device phantom type
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new typed tensor owned by this arena
     * @throws IllegalStateException if the arena is closed
     */
    public <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> zeros(S shape, D dtype, V device) {
        checkNotClosed();
        validateCpuOnly(device);

        TensorSpec spec = TensorSpec.of(dtype.scalarType(), shape.dimensions());
        Tensor tensor = Tensor.allocate(spec, underlying);
        TypedTensor<S, D, V> typed = TypedTensor.from(tensor, shape, dtype, device);

        allocatedTensors.add(typed);
        return typed;
    }

    /**
     * Allocates a tensor filled with a constant value.
     *
     * @param value the fill value
     * @param shape the shape phantom type
     * @param dtype the dtype phantom type
     * @param device the device phantom type
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new typed tensor filled with the value
     */
    public <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> full(float value, S shape, D dtype, V device) {
        checkNotClosed();
        validateCpuOnly(device);

        TensorSpec spec = TensorSpec.of(dtype.scalarType(), shape.dimensions());
        Tensor tensor = Tensor.allocate(spec, underlying);

        // Fill with value
        long count = tensor.elementCount();
        for (long i = 0; i < count; i++) {
            tensor.setFloatFlat(i, value);
        }

        TypedTensor<S, D, V> typed = TypedTensor.from(tensor, shape, dtype, device);
        allocatedTensors.add(typed);
        return typed;
    }

    /**
     * Allocates a tensor and copies data from a float array.
     *
     * @param data the source data
     * @param shape the shape phantom type
     * @param dtype the dtype phantom type
     * @param device the device phantom type
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new typed tensor with copied data
     */
    public <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> fromFloatArray(float[] data, S shape, D dtype, V device) {
        checkNotClosed();
        validateCpuOnly(device);

        TensorSpec spec = TensorSpec.of(dtype.scalarType(), shape.dimensions());
        if (data.length != spec.elementCount()) {
            throw new IllegalArgumentException(
                    "Data length " + data.length + " doesn't match shape " +
                    java.util.Arrays.toString(shape.dimensions()) +
                    " (expected " + spec.elementCount() + " elements)");
        }

        Tensor tensor = Tensor.allocate(spec, underlying);
        tensor.copyFrom(data);

        TypedTensor<S, D, V> typed = TypedTensor.from(tensor, shape, dtype, device);
        allocatedTensors.add(typed);
        return typed;
    }

    /**
     * Allocates a tensor and copies data from a double array.
     */
    public <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> fromDoubleArray(double[] data, S shape, D dtype, V device) {
        checkNotClosed();
        validateCpuOnly(device);

        TensorSpec spec = TensorSpec.of(dtype.scalarType(), shape.dimensions());
        if (data.length != spec.elementCount()) {
            throw new IllegalArgumentException(
                    "Data length " + data.length + " doesn't match shape " +
                    java.util.Arrays.toString(shape.dimensions()) +
                    " (expected " + spec.elementCount() + " elements)");
        }

        Tensor tensor = Tensor.allocate(spec, underlying);
        tensor.copyFrom(data);

        TypedTensor<S, D, V> typed = TypedTensor.from(tensor, shape, dtype, device);
        allocatedTensors.add(typed);
        return typed;
    }

    /**
     * Creates a copy of an existing tensor in this arena.
     *
     * <p>The new tensor is independent of the original and will be
     * freed when this arena closes.
     *
     * @param source the tensor to copy
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a copy of the tensor owned by this arena
     */
    public <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> copy(TypedTensor<S, D, V> source) {
        checkNotClosed();
        validateCpuOnly(source.deviceType());

        TensorSpec spec = source.underlying().spec();
        Tensor tensor = Tensor.allocate(spec, underlying);

        // Copy data
        java.lang.foreign.MemorySegment.copy(
                source.underlying().data(), 0,
                tensor.data(), 0,
                spec.byteSize());

        TypedTensor<S, D, V> typed = TypedTensor.from(
                tensor, source.shapeType(), source.dtypeType(), source.deviceType());
        allocatedTensors.add(typed);
        return typed;
    }

    // ==================== Lifecycle ====================

    /**
     * Returns the underlying FFM arena.
     *
     * <p>Use with caution - direct access bypasses type safety.
     *
     * @return the underlying Arena
     */
    public Arena underlying() {
        return underlying;
    }

    /**
     * Returns the number of tensors allocated in this arena.
     *
     * @return the tensor count
     */
    public int tensorCount() {
        return allocatedTensors.size();
    }

    /**
     * Returns whether this arena has been closed.
     *
     * @return true if closed
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Closes this arena and frees all allocated tensors.
     *
     * <p>After calling close, all tensors allocated from this arena
     * become invalid and must not be used.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            allocatedTensors.clear();
            underlying.close();
        }
    }

    // ==================== Internal ====================

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("TensorArena is closed");
        }
    }

    private void validateCpuOnly(DeviceTag device) {
        if (device.isGpu()) {
            throw new UnsupportedOperationException(
                    "TensorArena currently only supports CPU allocation. " +
                    "GPU memory management will be added with backend integration.");
        }
    }

    @Override
    public String toString() {
        return "TensorArena[tensors=" + allocatedTensors.size() +
               ", closed=" + closed + "]";
    }
}
