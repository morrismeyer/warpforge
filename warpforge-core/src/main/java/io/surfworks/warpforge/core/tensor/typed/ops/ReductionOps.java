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
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Scalar;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Type-safe reduction operations with compile-time checking.
 *
 * <p>Reduction operations transform tensors by collapsing dimensions.
 * Full reductions produce scalars, axis reductions reduce along specific dimensions.
 *
 * <p>Example:
 * <pre>{@code
 * TypedTensor<Matrix, F32, Cpu> matrix = TypedTensor.zeros(new Matrix(100, 200), F32.INSTANCE, Cpu.INSTANCE);
 *
 * // Full reduction to scalar
 * float total = ReductionOps.sum(matrix);
 * float average = ReductionOps.mean(matrix);
 * float maximum = ReductionOps.max(matrix);
 *
 * // Axis reduction (matrix → vector)
 * TypedTensor<Vector, F32, Cpu> rowSums = ReductionOps.sumAxis(matrix, 1);    // sum each row
 * TypedTensor<Vector, F32, Cpu> colSums = ReductionOps.sumAxis(matrix, 0);    // sum each column
 * }</pre>
 */
public final class ReductionOps {

    private ReductionOps() {
        // Utility class
    }

    // ==================== Full Reductions (to scalar) ====================

    /**
     * Sums all elements of a tensor.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return the sum of all elements
     */
    public static <S extends Shape, V extends DeviceTag>
    float sum(TypedTensor<S, F32, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();
        float sum = 0.0f;

        for (long i = 0; i < count; i++) {
            sum += data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }

        return sum;
    }

    /**
     * Sums all elements of a tensor (double precision).
     */
    public static <S extends Shape, V extends DeviceTag>
    double sumF64(TypedTensor<S, F64, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();
        double sum = 0.0;

        for (long i = 0; i < count; i++) {
            sum += data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }

        return sum;
    }

    /**
     * Computes the mean of all elements.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return the arithmetic mean
     */
    public static <S extends Shape, V extends DeviceTag>
    float mean(TypedTensor<S, F32, V> a) {
        return sum(a) / a.elementCount();
    }

    /**
     * Computes the mean of all elements (double precision).
     */
    public static <S extends Shape, V extends DeviceTag>
    double meanF64(TypedTensor<S, F64, V> a) {
        return sumF64(a) / a.elementCount();
    }

    /**
     * Finds the maximum element value.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return the maximum value
     */
    public static <S extends Shape, V extends DeviceTag>
    float max(TypedTensor<S, F32, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();

        if (count == 0) {
            throw new IllegalArgumentException("Cannot compute max of empty tensor");
        }

        float max = data.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        for (long i = 1; i < count; i++) {
            float val = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (val > max) {
                max = val;
            }
        }

        return max;
    }

    /**
     * Finds the maximum element value (double precision).
     */
    public static <S extends Shape, V extends DeviceTag>
    double maxF64(TypedTensor<S, F64, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();

        if (count == 0) {
            throw new IllegalArgumentException("Cannot compute max of empty tensor");
        }

        double max = data.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
        for (long i = 1; i < count; i++) {
            double val = data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            if (val > max) {
                max = val;
            }
        }

        return max;
    }

    /**
     * Finds the minimum element value.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return the minimum value
     */
    public static <S extends Shape, V extends DeviceTag>
    float min(TypedTensor<S, F32, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();

        if (count == 0) {
            throw new IllegalArgumentException("Cannot compute min of empty tensor");
        }

        float min = data.getAtIndex(ValueLayout.JAVA_FLOAT, 0);
        for (long i = 1; i < count; i++) {
            float val = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (val < min) {
                min = val;
            }
        }

        return min;
    }

    /**
     * Finds the minimum element value (double precision).
     */
    public static <S extends Shape, V extends DeviceTag>
    double minF64(TypedTensor<S, F64, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();

        if (count == 0) {
            throw new IllegalArgumentException("Cannot compute min of empty tensor");
        }

        double min = data.getAtIndex(ValueLayout.JAVA_DOUBLE, 0);
        for (long i = 1; i < count; i++) {
            double val = data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            if (val < min) {
                min = val;
            }
        }

        return min;
    }

    /**
     * Computes the product of all elements.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return the product of all elements
     */
    public static <S extends Shape, V extends DeviceTag>
    float prod(TypedTensor<S, F32, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();
        float prod = 1.0f;

        for (long i = 0; i < count; i++) {
            prod *= data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }

        return prod;
    }

    /**
     * Computes the product of all elements (double precision).
     */
    public static <S extends Shape, V extends DeviceTag>
    double prodF64(TypedTensor<S, F64, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();
        double prod = 1.0;

        for (long i = 0; i < count; i++) {
            prod *= data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
        }

        return prod;
    }

    /**
     * Computes the variance of all elements.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return the variance
     */
    public static <S extends Shape, V extends DeviceTag>
    float variance(TypedTensor<S, F32, V> a) {
        float m = mean(a);
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();
        float sumSq = 0.0f;

        for (long i = 0; i < count; i++) {
            float diff = data.getAtIndex(ValueLayout.JAVA_FLOAT, i) - m;
            sumSq += diff * diff;
        }

        return sumSq / count;
    }

    /**
     * Computes the variance of all elements (double precision).
     */
    public static <S extends Shape, V extends DeviceTag>
    double varianceF64(TypedTensor<S, F64, V> a) {
        double m = meanF64(a);
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();
        double sumSq = 0.0;

        for (long i = 0; i < count; i++) {
            double diff = data.getAtIndex(ValueLayout.JAVA_DOUBLE, i) - m;
            sumSq += diff * diff;
        }

        return sumSq / count;
    }

    /**
     * Computes the standard deviation of all elements.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return the standard deviation
     */
    public static <S extends Shape, V extends DeviceTag>
    float std(TypedTensor<S, F32, V> a) {
        return (float) Math.sqrt(variance(a));
    }

    /**
     * Computes the standard deviation (double precision).
     */
    public static <S extends Shape, V extends DeviceTag>
    double stdF64(TypedTensor<S, F64, V> a) {
        return Math.sqrt(varianceF64(a));
    }

    /**
     * Computes the L2 norm (Frobenius norm for matrices) of all elements.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return the L2 norm
     */
    public static <S extends Shape, V extends DeviceTag>
    float norm(TypedTensor<S, F32, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();
        float sumSq = 0.0f;

        for (long i = 0; i < count; i++) {
            float val = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            sumSq += val * val;
        }

        return (float) Math.sqrt(sumSq);
    }

    /**
     * Computes the L2 norm (double precision).
     */
    public static <S extends Shape, V extends DeviceTag>
    double normF64(TypedTensor<S, F64, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();
        double sumSq = 0.0;

        for (long i = 0; i < count; i++) {
            double val = data.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            sumSq += val * val;
        }

        return Math.sqrt(sumSq);
    }

    // ==================== Axis Reductions ====================

    /**
     * Sums along a specified axis of a matrix.
     *
     * <p>For matrix [M, N]:
     * - axis=0: sums columns, producing vector [N]
     * - axis=1: sums rows, producing vector [M]
     *
     * @param a the input matrix
     * @param axis the axis to reduce (0 or 1)
     * @param <V> the device type
     * @return the reduced vector
     */
    public static <V extends DeviceTag>
    TypedTensor<Vector, F32, V> sumAxis(TypedTensor<Matrix, F32, V> a, int axis) {
        int[] shape = a.dimensions();
        int M = shape[0];
        int N = shape[1];

        if (axis < 0 || axis > 1) {
            throw new IllegalArgumentException("Axis must be 0 or 1 for matrix, got: " + axis);
        }

        int resultSize = (axis == 0) ? N : M;
        Vector resultShape = new Vector(resultSize);
        Tensor result = Tensor.zeros(ScalarType.F32, resultSize);
        MemorySegment data = a.underlying().data();
        MemorySegment out = result.data();

        if (axis == 0) {
            // Sum along rows (collapse M dimension), result size N
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int i = 0; i < M; i++) {
                    sum += data.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j);
                }
                out.setAtIndex(ValueLayout.JAVA_FLOAT, j, sum);
            }
        } else {
            // Sum along columns (collapse N dimension), result size M
            for (int i = 0; i < M; i++) {
                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    sum += data.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j);
                }
                out.setAtIndex(ValueLayout.JAVA_FLOAT, i, sum);
            }
        }

        return TypedTensor.from(result, resultShape, F32.INSTANCE, a.deviceType());
    }

    /**
     * Computes mean along a specified axis of a matrix.
     */
    public static <V extends DeviceTag>
    TypedTensor<Vector, F32, V> meanAxis(TypedTensor<Matrix, F32, V> a, int axis) {
        int[] shape = a.dimensions();
        int M = shape[0];
        int N = shape[1];
        int divisor = (axis == 0) ? M : N;

        TypedTensor<Vector, F32, V> sums = sumAxis(a, axis);
        MemorySegment data = sums.underlying().data();
        int size = sums.dimensions()[0];

        for (int i = 0; i < size; i++) {
            float val = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            data.setAtIndex(ValueLayout.JAVA_FLOAT, i, val / divisor);
        }

        return sums;
    }

    /**
     * Finds the maximum along a specified axis of a matrix.
     */
    public static <V extends DeviceTag>
    TypedTensor<Vector, F32, V> maxAxis(TypedTensor<Matrix, F32, V> a, int axis) {
        int[] shape = a.dimensions();
        int M = shape[0];
        int N = shape[1];

        if (axis < 0 || axis > 1) {
            throw new IllegalArgumentException("Axis must be 0 or 1 for matrix, got: " + axis);
        }

        int resultSize = (axis == 0) ? N : M;
        Vector resultShape = new Vector(resultSize);
        Tensor result = Tensor.zeros(ScalarType.F32, resultSize);
        MemorySegment data = a.underlying().data();
        MemorySegment out = result.data();

        if (axis == 0) {
            // Max along rows (collapse M dimension)
            for (int j = 0; j < N; j++) {
                float max = data.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                for (int i = 1; i < M; i++) {
                    float val = data.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j);
                    if (val > max) {
                        max = val;
                    }
                }
                out.setAtIndex(ValueLayout.JAVA_FLOAT, j, max);
            }
        } else {
            // Max along columns (collapse N dimension)
            for (int i = 0; i < M; i++) {
                float max = data.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N);
                for (int j = 1; j < N; j++) {
                    float val = data.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j);
                    if (val > max) {
                        max = val;
                    }
                }
                out.setAtIndex(ValueLayout.JAVA_FLOAT, i, max);
            }
        }

        return TypedTensor.from(result, resultShape, F32.INSTANCE, a.deviceType());
    }

    /**
     * Finds the minimum along a specified axis of a matrix.
     */
    public static <V extends DeviceTag>
    TypedTensor<Vector, F32, V> minAxis(TypedTensor<Matrix, F32, V> a, int axis) {
        int[] shape = a.dimensions();
        int M = shape[0];
        int N = shape[1];

        if (axis < 0 || axis > 1) {
            throw new IllegalArgumentException("Axis must be 0 or 1 for matrix, got: " + axis);
        }

        int resultSize = (axis == 0) ? N : M;
        Vector resultShape = new Vector(resultSize);
        Tensor result = Tensor.zeros(ScalarType.F32, resultSize);
        MemorySegment data = a.underlying().data();
        MemorySegment out = result.data();

        if (axis == 0) {
            // Min along rows
            for (int j = 0; j < N; j++) {
                float min = data.getAtIndex(ValueLayout.JAVA_FLOAT, j);
                for (int i = 1; i < M; i++) {
                    float val = data.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j);
                    if (val < min) {
                        min = val;
                    }
                }
                out.setAtIndex(ValueLayout.JAVA_FLOAT, j, min);
            }
        } else {
            // Min along columns
            for (int i = 0; i < M; i++) {
                float min = data.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N);
                for (int j = 1; j < N; j++) {
                    float val = data.getAtIndex(ValueLayout.JAVA_FLOAT, (long) i * N + j);
                    if (val < min) {
                        min = val;
                    }
                }
                out.setAtIndex(ValueLayout.JAVA_FLOAT, i, min);
            }
        }

        return TypedTensor.from(result, resultShape, F32.INSTANCE, a.deviceType());
    }

    // ==================== Index Reductions ====================

    /**
     * Finds the index of the maximum element.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return the flat index of the maximum element
     */
    public static <S extends Shape, V extends DeviceTag>
    long argmax(TypedTensor<S, F32, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();

        if (count == 0) {
            throw new IllegalArgumentException("Cannot compute argmax of empty tensor");
        }

        long maxIdx = 0;
        float max = data.getAtIndex(ValueLayout.JAVA_FLOAT, 0);

        for (long i = 1; i < count; i++) {
            float val = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (val > max) {
                max = val;
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /**
     * Finds the index of the minimum element.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return the flat index of the minimum element
     */
    public static <S extends Shape, V extends DeviceTag>
    long argmin(TypedTensor<S, F32, V> a) {
        MemorySegment data = a.underlying().data();
        long count = a.elementCount();

        if (count == 0) {
            throw new IllegalArgumentException("Cannot compute argmin of empty tensor");
        }

        long minIdx = 0;
        float min = data.getAtIndex(ValueLayout.JAVA_FLOAT, 0);

        for (long i = 1; i < count; i++) {
            float val = data.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (val < min) {
                min = val;
                minIdx = i;
            }
        }

        return minIdx;
    }

    // ==================== Typed Full Reductions (returning TypedTensor<Scalar>) ====================

    /**
     * Sums all elements, returning a scalar tensor.
     *
     * @param a the input tensor
     * @param <S> the shape type
     * @param <V> the device type
     * @return a scalar tensor containing the sum
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<Scalar, F32, V> sumToScalar(TypedTensor<S, F32, V> a) {
        float result = sum(a);
        Tensor tensor = Tensor.fromFloatArray(new float[]{result});
        return TypedTensor.from(tensor, Scalar.INSTANCE, F32.INSTANCE, a.deviceType());
    }

    /**
     * Computes mean, returning a scalar tensor.
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<Scalar, F32, V> meanToScalar(TypedTensor<S, F32, V> a) {
        float result = mean(a);
        Tensor tensor = Tensor.fromFloatArray(new float[]{result});
        return TypedTensor.from(tensor, Scalar.INSTANCE, F32.INSTANCE, a.deviceType());
    }

    /**
     * Finds max, returning a scalar tensor.
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<Scalar, F32, V> maxToScalar(TypedTensor<S, F32, V> a) {
        float result = max(a);
        Tensor tensor = Tensor.fromFloatArray(new float[]{result});
        return TypedTensor.from(tensor, Scalar.INSTANCE, F32.INSTANCE, a.deviceType());
    }

    /**
     * Finds min, returning a scalar tensor.
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<Scalar, F32, V> minToScalar(TypedTensor<S, F32, V> a) {
        float result = min(a);
        Tensor tensor = Tensor.fromFloatArray(new float[]{result});
        return TypedTensor.from(tensor, Scalar.INSTANCE, F32.INSTANCE, a.deviceType());
    }
}
