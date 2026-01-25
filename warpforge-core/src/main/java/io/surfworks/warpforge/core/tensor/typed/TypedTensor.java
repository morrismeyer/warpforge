package io.surfworks.warpforge.core.tensor.typed;

import java.lang.foreign.Arena;
import java.util.Arrays;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;
import io.surfworks.warpforge.core.tensor.typed.device.AnyDevice;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.shape.Dynamic;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank4;
import io.surfworks.warpforge.core.tensor.typed.shape.Scalar;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Type-safe tensor wrapper with compile-time checking for shape, dtype, and device.
 *
 * <p>TypedTensor uses phantom types to provide compile-time guarantees about tensor
 * compatibility. Operations between tensors with mismatched types will fail at compile
 * time rather than runtime.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Create typed tensors
 * TypedTensor<Matrix, F32, Cpu> weights = TypedTensor.zeros(
 *     new Matrix(768, 512), F32.INSTANCE, Cpu.INSTANCE);
 * TypedTensor<Matrix, F32, Cpu> input = TypedTensor.zeros(
 *     new Matrix(32, 768), F32.INSTANCE, Cpu.INSTANCE);
 *
 * // Type-safe operations (compile-time checked)
 * TypedTensor<Matrix, F32, Cpu> output = MatrixOps.matmul(input, weights);
 *
 * // These would NOT compile:
 * // MatrixOps.matmul(input, TypedTensor.zeros(..., F64.INSTANCE, ...));  // dtype mismatch
 * // MatrixOps.matmul(input, TypedTensor.zeros(..., ..., Nvidia.DEFAULT)); // device mismatch
 *
 * // Explicit device transfer
 * TypedTensor<Matrix, F32, Nvidia> gpuWeights = weights.to(Nvidia.DEFAULT);
 * }</pre>
 *
 * @param <S> the shape phantom type (e.g., Matrix, Vector, Rank4)
 * @param <D> the data type phantom type (e.g., F32, F64)
 * @param <V> the device phantom type (e.g., Cpu, Nvidia)
 */
public final class TypedTensor<S extends Shape, D extends DTypeTag, V extends DeviceTag>
        implements AutoCloseable {

    private final Tensor underlying;
    private final S shape;
    private final D dtype;
    private final V device;

    private TypedTensor(Tensor underlying, S shape, D dtype, V device) {
        this.underlying = underlying;
        this.shape = shape;
        this.dtype = dtype;
        this.device = device;
    }

    // ==================== Factory Methods ====================

    /**
     * Creates a zero-initialized tensor with the specified shape, dtype, and device.
     *
     * @param shape the shape phantom type instance
     * @param dtype the dtype phantom type instance
     * @param device the device phantom type instance
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new zero-initialized typed tensor
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> zeros(S shape, D dtype, V device) {
        validateCpuOnly(device, "zeros");
        Tensor tensor = Tensor.zeros(dtype.scalarType(), shape.dimensions());
        return new TypedTensor<>(tensor, shape, dtype, device);
    }

    /**
     * Creates a tensor filled with a constant value.
     *
     * @param value the fill value
     * @param shape the shape phantom type instance
     * @param dtype the dtype phantom type instance
     * @param device the device phantom type instance
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new typed tensor filled with the specified value
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> full(float value, S shape, D dtype, V device) {
        validateCpuOnly(device, "full");
        Tensor tensor = Tensor.full(value, shape.dimensions());
        return new TypedTensor<>(tensor, shape, dtype, device);
    }

    /**
     * Creates a typed tensor from a float array.
     *
     * @param data the source data
     * @param shape the shape phantom type instance
     * @param dtype the dtype phantom type instance (must be F32)
     * @param device the device phantom type instance
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new typed tensor with the provided data
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> fromFloatArray(float[] data, S shape, D dtype, V device) {
        validateCpuOnly(device, "fromFloatArray");
        if (dtype.scalarType() != ScalarType.F32) {
            throw new IllegalArgumentException(
                    "fromFloatArray requires F32 dtype, got: " + dtype.scalarType());
        }
        Tensor tensor = Tensor.fromFloatArray(data, shape.dimensions());
        return new TypedTensor<>(tensor, shape, dtype, device);
    }

    /**
     * Creates a typed tensor from a double array.
     *
     * @param data the source data
     * @param shape the shape phantom type instance
     * @param dtype the dtype phantom type instance (must be F64)
     * @param device the device phantom type instance
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new typed tensor with the provided data
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> fromDoubleArray(double[] data, S shape, D dtype, V device) {
        validateCpuOnly(device, "fromDoubleArray");
        if (dtype.scalarType() != ScalarType.F64) {
            throw new IllegalArgumentException(
                    "fromDoubleArray requires F64 dtype, got: " + dtype.scalarType());
        }
        Tensor tensor = Tensor.fromDoubleArray(data, shape.dimensions());
        return new TypedTensor<>(tensor, shape, dtype, device);
    }

    /**
     * Creates a typed tensor from an int array.
     *
     * @param data the source data
     * @param shape the shape phantom type instance
     * @param dtype the dtype phantom type instance (must be I32)
     * @param device the device phantom type instance
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new typed tensor with the provided data
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> fromIntArray(int[] data, S shape, D dtype, V device) {
        validateCpuOnly(device, "fromIntArray");
        if (dtype.scalarType() != ScalarType.I32) {
            throw new IllegalArgumentException(
                    "fromIntArray requires I32 dtype, got: " + dtype.scalarType());
        }
        Tensor tensor = Tensor.fromIntArray(data, shape.dimensions());
        return new TypedTensor<>(tensor, shape, dtype, device);
    }

    /**
     * Wraps an existing untyped Tensor with compile-time type information.
     *
     * <p>This method validates that the underlying tensor matches the expected
     * shape and dtype. Use this for interop with existing untyped code.
     *
     * @param tensor the underlying untyped tensor
     * @param shape the expected shape phantom type
     * @param dtype the expected dtype phantom type
     * @param device the device phantom type
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a typed wrapper around the tensor
     * @throws IllegalArgumentException if tensor doesn't match expected shape or dtype
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> from(Tensor tensor, S shape, D dtype, V device) {
        // Validate dtype matches
        if (tensor.dtype() != dtype.scalarType()) {
            throw new IllegalArgumentException(
                    "Tensor dtype " + tensor.dtype() + " doesn't match expected " + dtype.scalarType());
        }

        // Validate shape matches
        if (!Arrays.equals(tensor.shape(), shape.dimensions())) {
            throw new IllegalArgumentException(
                    "Tensor shape " + Arrays.toString(tensor.shape()) +
                    " doesn't match expected " + Arrays.toString(shape.dimensions()));
        }

        return new TypedTensor<>(tensor, shape, dtype, device);
    }

    /**
     * Wraps an untyped tensor with dynamic shape type for maximum flexibility.
     *
     * <p>Use this when the shape is not known at compile time, such as when
     * loading tensors from files or receiving them from external systems.
     *
     * @param tensor the underlying untyped tensor
     * @param dtype the expected dtype phantom type
     * @param device the device phantom type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a typed tensor with Dynamic shape
     * @throws IllegalArgumentException if tensor dtype doesn't match
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Dynamic, D, V> fromDynamic(Tensor tensor, D dtype, V device) {
        if (tensor.dtype() != dtype.scalarType()) {
            throw new IllegalArgumentException(
                    "Tensor dtype " + tensor.dtype() + " doesn't match expected " + dtype.scalarType());
        }
        Dynamic dynamicShape = new Dynamic(tensor.shape());
        return new TypedTensor<>(tensor, dynamicShape, dtype, device);
    }

    /**
     * Creates a typed tensor allocated from an external arena (does not own the arena).
     *
     * @param spec the tensor specification
     * @param arena the arena to allocate from
     * @param shape the shape phantom type
     * @param dtype the dtype phantom type
     * @param device the device phantom type
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a typed tensor allocated from the arena
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> allocate(TensorSpec spec, Arena arena, S shape, D dtype, V device) {
        validateCpuOnly(device, "allocate");

        // Validate spec matches phantom types
        if (spec.dtype() != dtype.scalarType()) {
            throw new IllegalArgumentException(
                    "TensorSpec dtype " + spec.dtype() + " doesn't match expected " + dtype.scalarType());
        }
        if (!Arrays.equals(spec.shape(), shape.dimensions())) {
            throw new IllegalArgumentException(
                    "TensorSpec shape " + Arrays.toString(spec.shape()) +
                    " doesn't match expected " + Arrays.toString(shape.dimensions()));
        }

        Tensor tensor = Tensor.allocate(spec, arena);
        return new TypedTensor<>(tensor, shape, dtype, device);
    }

    // ==================== Accessors ====================

    /**
     * Returns the underlying untyped tensor.
     *
     * <p>Use this for interop with existing untyped APIs.
     *
     * @return the underlying Tensor
     */
    public Tensor underlying() {
        return underlying;
    }

    /**
     * Returns the shape phantom type instance.
     *
     * @return the shape type
     */
    public S shapeType() {
        return shape;
    }

    /**
     * Returns the dtype phantom type instance.
     *
     * @return the dtype type
     */
    public D dtypeType() {
        return dtype;
    }

    /**
     * Returns the device phantom type instance.
     *
     * @return the device type
     */
    public V deviceType() {
        return device;
    }

    /**
     * Returns the tensor's dimensions as an array.
     *
     * @return copy of the shape dimensions
     */
    public int[] dimensions() {
        return shape.dimensions().clone();
    }

    /**
     * Returns the tensor's rank (number of dimensions).
     *
     * @return the rank
     */
    public int rank() {
        return shape.rank();
    }

    /**
     * Returns the total number of elements in the tensor.
     *
     * @return the element count
     */
    public long elementCount() {
        return underlying.elementCount();
    }

    // ==================== Type Conversions ====================

    /**
     * Transfers this tensor to a different device.
     *
     * <p>Currently only CPU tensors are supported. GPU transfer will be
     * implemented when backend integration is complete.
     *
     * @param targetDevice the target device
     * @param <W> the target device type
     * @return a new tensor on the target device
     * @throws UnsupportedOperationException if GPU transfer is not yet implemented
     */
    public <W extends DeviceTag> TypedTensor<S, D, W> to(W targetDevice) {
        if (device instanceof Cpu && targetDevice instanceof Cpu) {
            // CPU to CPU: just copy
            Tensor copy = underlying.copy();
            return new TypedTensor<>(copy, shape, dtype, targetDevice);
        }

        if (targetDevice instanceof AnyDevice) {
            // Allow "upcast" to AnyDevice
            @SuppressWarnings("unchecked")
            TypedTensor<S, D, W> result = (TypedTensor<S, D, W>) this;
            return result;
        }

        // GPU transfer not yet implemented
        throw new UnsupportedOperationException(
                "Device transfer from " + device.deviceName() + " to " + targetDevice.deviceName() +
                " is not yet implemented. GPU backend integration required.");
    }

    /**
     * Reshapes this tensor to a new shape with the same element count.
     *
     * <p>The new shape must have the same total number of elements.
     * This operation may create a view (sharing data) or a copy depending
     * on the underlying implementation.
     *
     * @param newShape the new shape phantom type
     * @param <T> the new shape type
     * @return a reshaped tensor
     * @throws IllegalArgumentException if element counts don't match
     */
    public <T extends Shape> TypedTensor<T, D, V> reshape(T newShape) {
        if (newShape.elementCount() != shape.elementCount()) {
            throw new IllegalArgumentException(
                    "Cannot reshape from " + shape.elementCount() + " to " +
                    newShape.elementCount() + " elements");
        }
        Tensor reshaped = underlying.reshape(newShape.dimensions());
        return new TypedTensor<>(reshaped, newShape, dtype, device);
    }

    /**
     * Casts this tensor to a different dtype.
     *
     * <p>Note: This performs a conversion, creating a new tensor with the
     * new dtype. The original tensor is not modified.
     *
     * @param newDtype the target dtype
     * @param <E> the new dtype type
     * @return a new tensor with the converted dtype
     */
    public <E extends DTypeTag> TypedTensor<S, E, V> cast(E newDtype) {
        // For now, only support casting between same types (no-op) or
        // same-size types. Full casting requires backend support.
        if (dtype.scalarType() == newDtype.scalarType()) {
            @SuppressWarnings("unchecked")
            TypedTensor<S, E, V> result = (TypedTensor<S, E, V>) this;
            return result;
        }

        throw new UnsupportedOperationException(
                "Dtype casting from " + dtype.scalarType() + " to " + newDtype.scalarType() +
                " is not yet implemented. Backend integration required.");
    }

    /**
     * Creates a copy with AnyDevice type for use in generic code.
     *
     * @return this tensor with AnyDevice type
     */
    public TypedTensor<S, D, AnyDevice> asAnyDevice() {
        return new TypedTensor<>(underlying, shape, dtype, AnyDevice.INSTANCE);
    }

    /**
     * Creates a copy with Dynamic shape for use in generic code.
     *
     * @return this tensor with Dynamic shape
     */
    public TypedTensor<Dynamic, D, V> asDynamic() {
        Dynamic dynamicShape = new Dynamic(shape.dimensions());
        return new TypedTensor<>(underlying, dynamicShape, dtype, device);
    }

    // ==================== Deep Copy ====================

    /**
     * Creates a deep copy of this tensor.
     *
     * @return a new tensor with copied data
     */
    public TypedTensor<S, D, V> copy() {
        Tensor copied = underlying.copy();
        return new TypedTensor<>(copied, shape, dtype, device);
    }

    // ==================== Shape Specialization Utilities ====================

    /**
     * Attempts to specialize a Dynamic-shaped tensor to a specific shape type.
     *
     * @param tensor the dynamically-shaped tensor
     * @param targetShape the target shape
     * @param <S> the target shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return the specialized tensor
     * @throws IllegalArgumentException if shapes don't match
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> specialize(TypedTensor<Dynamic, D, V> tensor, S targetShape) {
        if (!Arrays.equals(tensor.dimensions(), targetShape.dimensions())) {
            throw new IllegalArgumentException(
                    "Cannot specialize tensor with shape " + Arrays.toString(tensor.dimensions()) +
                    " to " + Arrays.toString(targetShape.dimensions()));
        }
        return new TypedTensor<>(tensor.underlying(), targetShape, tensor.dtype, tensor.device);
    }

    /**
     * Creates a Scalar-shaped typed tensor from a Dynamic tensor.
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Scalar, D, V> toScalar(TypedTensor<Dynamic, D, V> tensor) {
        if (tensor.rank() != 0) {
            throw new IllegalArgumentException(
                    "Cannot convert rank-" + tensor.rank() + " tensor to Scalar");
        }
        return specialize(tensor, Scalar.INSTANCE);
    }

    /**
     * Creates a Vector-shaped typed tensor from a Dynamic tensor.
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Vector, D, V> toVector(TypedTensor<Dynamic, D, V> tensor) {
        if (tensor.rank() != 1) {
            throw new IllegalArgumentException(
                    "Cannot convert rank-" + tensor.rank() + " tensor to Vector");
        }
        int length = tensor.dimensions()[0];
        return specialize(tensor, new Vector(length));
    }

    /**
     * Creates a Matrix-shaped typed tensor from a Dynamic tensor.
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> toMatrix(TypedTensor<Dynamic, D, V> tensor) {
        if (tensor.rank() != 2) {
            throw new IllegalArgumentException(
                    "Cannot convert rank-" + tensor.rank() + " tensor to Matrix");
        }
        int[] dims = tensor.dimensions();
        return specialize(tensor, new Matrix(dims[0], dims[1]));
    }

    /**
     * Creates a Rank3-shaped typed tensor from a Dynamic tensor.
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Rank3, D, V> toRank3(TypedTensor<Dynamic, D, V> tensor) {
        if (tensor.rank() != 3) {
            throw new IllegalArgumentException(
                    "Cannot convert rank-" + tensor.rank() + " tensor to Rank3");
        }
        int[] dims = tensor.dimensions();
        return specialize(tensor, new Rank3(dims[0], dims[1], dims[2]));
    }

    /**
     * Creates a Rank4-shaped typed tensor from a Dynamic tensor.
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Rank4, D, V> toRank4(TypedTensor<Dynamic, D, V> tensor) {
        if (tensor.rank() != 4) {
            throw new IllegalArgumentException(
                    "Cannot convert rank-" + tensor.rank() + " tensor to Rank4");
        }
        int[] dims = tensor.dimensions();
        return specialize(tensor, new Rank4(dims[0], dims[1], dims[2], dims[3]));
    }

    // ==================== Lifecycle ====================

    @Override
    public void close() {
        underlying.close();
    }

    @Override
    public String toString() {
        return "TypedTensor[" +
                "shape=" + shape +
                ", dtype=" + dtype +
                ", device=" + device +
                ", elements=" + elementCount() +
                "]";
    }

    // ==================== Internal Utilities ====================

    private static void validateCpuOnly(DeviceTag device, String operation) {
        if (device.isGpu()) {
            throw new UnsupportedOperationException(
                    operation + " with GPU device not yet implemented. " +
                    "Use CPU device or wait for GPU backend integration.");
        }
    }
}
