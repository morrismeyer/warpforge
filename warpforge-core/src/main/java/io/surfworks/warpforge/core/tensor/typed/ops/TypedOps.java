package io.surfworks.warpforge.core.tensor.typed.ops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;

/**
 * Type-safe elementwise tensor operations.
 *
 * <p>All operations enforce compile-time compatibility of shape, dtype, and device.
 * Operations between tensors with mismatched types will fail to compile.
 *
 * <p>Example:
 * <pre>{@code
 * TypedTensor<Matrix, F32, Cpu> a = TypedTensor.zeros(new Matrix(100, 100), F32.INSTANCE, Cpu.INSTANCE);
 * TypedTensor<Matrix, F32, Cpu> b = TypedTensor.zeros(new Matrix(100, 100), F32.INSTANCE, Cpu.INSTANCE);
 *
 * // These compile and work:
 * TypedTensor<Matrix, F32, Cpu> sum = TypedOps.add(a, b);
 * TypedTensor<Matrix, F32, Cpu> prod = TypedOps.mul(a, b);
 * TypedTensor<Matrix, F32, Cpu> scaled = TypedOps.scale(a, 2.0f);
 *
 * // These would NOT compile:
 * // TypedOps.add(a, TypedTensor.zeros(..., F64.INSTANCE, ...));  // dtype mismatch
 * // TypedOps.add(a, TypedTensor.zeros(..., ..., Nvidia.DEFAULT)); // device mismatch
 * }</pre>
 */
public final class TypedOps {

    private TypedOps() {
        // Utility class
    }

    // ==================== Binary Elementwise Operations ====================

    /**
     * Elementwise addition of two tensors.
     *
     * <p>Tensors must have identical shape, dtype, and device.
     *
     * @param a first operand
     * @param b second operand
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new tensor containing a + b
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> add(TypedTensor<S, D, V> a, TypedTensor<S, D, V> b) {
        validateBroadcastableShapes(a, b);
        Tensor result = createResultTensor(a);

        ScalarType dtype = a.underlying().dtype();
        long count = a.elementCount();

        if (dtype == ScalarType.F32) {
            addF32(a.underlying().data(), b.underlying().data(), result.data(), count);
        } else if (dtype == ScalarType.F64) {
            addF64(a.underlying().data(), b.underlying().data(), result.data(), count);
        } else {
            throw new UnsupportedOperationException("add not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Elementwise subtraction of two tensors.
     *
     * @param a first operand
     * @param b second operand
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new tensor containing a - b
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> sub(TypedTensor<S, D, V> a, TypedTensor<S, D, V> b) {
        validateBroadcastableShapes(a, b);
        Tensor result = createResultTensor(a);

        ScalarType dtype = a.underlying().dtype();
        long count = a.elementCount();

        if (dtype == ScalarType.F32) {
            subF32(a.underlying().data(), b.underlying().data(), result.data(), count);
        } else if (dtype == ScalarType.F64) {
            subF64(a.underlying().data(), b.underlying().data(), result.data(), count);
        } else {
            throw new UnsupportedOperationException("sub not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Elementwise multiplication of two tensors.
     *
     * @param a first operand
     * @param b second operand
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new tensor containing a * b elementwise
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> mul(TypedTensor<S, D, V> a, TypedTensor<S, D, V> b) {
        validateBroadcastableShapes(a, b);
        Tensor result = createResultTensor(a);

        ScalarType dtype = a.underlying().dtype();
        long count = a.elementCount();

        if (dtype == ScalarType.F32) {
            mulF32(a.underlying().data(), b.underlying().data(), result.data(), count);
        } else if (dtype == ScalarType.F64) {
            mulF64(a.underlying().data(), b.underlying().data(), result.data(), count);
        } else {
            throw new UnsupportedOperationException("mul not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Elementwise division of two tensors.
     *
     * @param a numerator tensor
     * @param b denominator tensor
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new tensor containing a / b elementwise
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> div(TypedTensor<S, D, V> a, TypedTensor<S, D, V> b) {
        validateBroadcastableShapes(a, b);
        Tensor result = createResultTensor(a);

        ScalarType dtype = a.underlying().dtype();
        long count = a.elementCount();

        if (dtype == ScalarType.F32) {
            divF32(a.underlying().data(), b.underlying().data(), result.data(), count);
        } else if (dtype == ScalarType.F64) {
            divF64(a.underlying().data(), b.underlying().data(), result.data(), count);
        } else {
            throw new UnsupportedOperationException("div not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    // ==================== Scalar Operations ====================

    /**
     * Scales a tensor by a scalar value.
     *
     * @param a the tensor to scale
     * @param scalar the scaling factor
     * @param <S> the shape type
     * @param <V> the device type
     * @return a new tensor containing a * scalar
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, F32, V> scale(TypedTensor<S, F32, V> a, float scalar) {
        Tensor result = createResultTensor(a);
        long count = a.elementCount();

        MemorySegment src = a.underlying().data();
        MemorySegment dst = result.data();

        for (long i = 0; i < count; i++) {
            float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, val * scalar);
        }

        return TypedTensor.from(result, a.shapeType(), F32.INSTANCE, a.deviceType());
    }

    /**
     * Scales a tensor by a scalar value (double precision).
     *
     * @param a the tensor to scale
     * @param scalar the scaling factor
     * @param <S> the shape type
     * @param <V> the device type
     * @return a new tensor containing a * scalar
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, F64, V> scale(TypedTensor<S, F64, V> a, double scalar) {
        Tensor result = createResultTensor(a);
        long count = a.elementCount();

        MemorySegment src = a.underlying().data();
        MemorySegment dst = result.data();

        for (long i = 0; i < count; i++) {
            double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            dst.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val * scalar);
        }

        return TypedTensor.from(result, a.shapeType(), F64.INSTANCE, a.deviceType());
    }

    /**
     * Adds a scalar to all elements of a tensor.
     *
     * @param a the tensor
     * @param scalar the scalar to add
     * @param <S> the shape type
     * @param <V> the device type
     * @return a new tensor containing a + scalar
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, F32, V> addScalar(TypedTensor<S, F32, V> a, float scalar) {
        Tensor result = createResultTensor(a);
        long count = a.elementCount();

        MemorySegment src = a.underlying().data();
        MemorySegment dst = result.data();

        for (long i = 0; i < count; i++) {
            float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, val + scalar);
        }

        return TypedTensor.from(result, a.shapeType(), F32.INSTANCE, a.deviceType());
    }

    // ==================== Unary Operations ====================

    /**
     * Negates all elements of a tensor.
     *
     * @param a the tensor to negate
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new tensor containing -a
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> neg(TypedTensor<S, D, V> a) {
        Tensor result = createResultTensor(a);

        ScalarType dtype = a.underlying().dtype();
        long count = a.elementCount();

        if (dtype == ScalarType.F32) {
            negF32(a.underlying().data(), result.data(), count);
        } else if (dtype == ScalarType.F64) {
            negF64(a.underlying().data(), result.data(), count);
        } else {
            throw new UnsupportedOperationException("neg not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Computes the absolute value of all elements.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new tensor containing |a|
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> abs(TypedTensor<S, D, V> a) {
        Tensor result = createResultTensor(a);

        ScalarType dtype = a.underlying().dtype();
        long count = a.elementCount();

        if (dtype == ScalarType.F32) {
            absF32(a.underlying().data(), result.data(), count);
        } else if (dtype == ScalarType.F64) {
            absF64(a.underlying().data(), result.data(), count);
        } else {
            throw new UnsupportedOperationException("abs not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Computes the square root of all elements.
     *
     * @param a the input tensor (must be non-negative)
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new tensor containing sqrt(a)
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> sqrt(TypedTensor<S, D, V> a) {
        Tensor result = createResultTensor(a);

        ScalarType dtype = a.underlying().dtype();
        long count = a.elementCount();

        if (dtype == ScalarType.F32) {
            sqrtF32(a.underlying().data(), result.data(), count);
        } else if (dtype == ScalarType.F64) {
            sqrtF64(a.underlying().data(), result.data(), count);
        } else {
            throw new UnsupportedOperationException("sqrt not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Computes the exponential of all elements.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new tensor containing exp(a)
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> exp(TypedTensor<S, D, V> a) {
        Tensor result = createResultTensor(a);

        ScalarType dtype = a.underlying().dtype();
        long count = a.elementCount();

        if (dtype == ScalarType.F32) {
            expF32(a.underlying().data(), result.data(), count);
        } else if (dtype == ScalarType.F64) {
            expF64(a.underlying().data(), result.data(), count);
        } else {
            throw new UnsupportedOperationException("exp not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    /**
     * Computes the natural logarithm of all elements.
     *
     * @param a the input tensor (must be positive)
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new tensor containing log(a)
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    TypedTensor<S, D, V> log(TypedTensor<S, D, V> a) {
        Tensor result = createResultTensor(a);

        ScalarType dtype = a.underlying().dtype();
        long count = a.elementCount();

        if (dtype == ScalarType.F32) {
            logF32(a.underlying().data(), result.data(), count);
        } else if (dtype == ScalarType.F64) {
            logF64(a.underlying().data(), result.data(), count);
        } else {
            throw new UnsupportedOperationException("log not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, a.shapeType(), a.dtypeType(), a.deviceType());
    }

    // ==================== Internal Implementation ====================

    private static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    Tensor createResultTensor(TypedTensor<S, D, V> template) {
        return Tensor.zeros(template.underlying().dtype(), template.dimensions());
    }

    private static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    void validateBroadcastableShapes(TypedTensor<S, D, V> a, TypedTensor<S, D, V> b) {
        // For now, require exact shape match. Broadcasting would need more complex logic.
        if (a.elementCount() != b.elementCount()) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + java.util.Arrays.toString(a.dimensions()) +
                    " vs " + java.util.Arrays.toString(b.dimensions()));
        }
    }

    // F32 implementations
    private static void addF32(MemorySegment a, MemorySegment b, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            float va = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float vb = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, va + vb);
        }
    }

    private static void subF32(MemorySegment a, MemorySegment b, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            float va = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float vb = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, va - vb);
        }
    }

    private static void mulF32(MemorySegment a, MemorySegment b, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            float va = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float vb = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, va * vb);
        }
    }

    private static void divF32(MemorySegment a, MemorySegment b, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            float va = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float vb = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, va / vb);
        }
    }

    private static void negF32(MemorySegment a, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            float va = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, -va);
        }
    }

    private static void absF32(MemorySegment a, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            float va = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, Math.abs(va));
        }
    }

    private static void sqrtF32(MemorySegment a, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            float va = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.sqrt(va));
        }
    }

    private static void expF32(MemorySegment a, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            float va = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.exp(va));
        }
    }

    private static void logF32(MemorySegment a, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            float va = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.log(va));
        }
    }

    // F64 implementations
    private static void addF64(MemorySegment a, MemorySegment b, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            double va = a.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            double vb = b.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, i, va + vb);
        }
    }

    private static void subF64(MemorySegment a, MemorySegment b, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            double va = a.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            double vb = b.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, i, va - vb);
        }
    }

    private static void mulF64(MemorySegment a, MemorySegment b, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            double va = a.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            double vb = b.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, i, va * vb);
        }
    }

    private static void divF64(MemorySegment a, MemorySegment b, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            double va = a.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            double vb = b.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, i, va / vb);
        }
    }

    private static void negF64(MemorySegment a, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            double va = a.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, i, -va);
        }
    }

    private static void absF64(MemorySegment a, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            double va = a.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.abs(va));
        }
    }

    private static void sqrtF64(MemorySegment a, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            double va = a.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.sqrt(va));
        }
    }

    private static void expF64(MemorySegment a, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            double va = a.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.exp(va));
        }
    }

    private static void logF64(MemorySegment a, MemorySegment out, long count) {
        for (long i = 0; i < count; i++) {
            double va = a.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            out.setAtIndex(ValueLayout.JAVA_DOUBLE, i, Math.log(va));
        }
    }
}
