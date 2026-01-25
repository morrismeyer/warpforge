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
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank4;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Type-safe activation function operations for tensors.
 *
 * <p>These operations implement common activation functions used in transformer models:
 * <ul>
 *   <li>Softmax - attention probability distribution</li>
 *   <li>GELU - Gaussian Error Linear Unit (used in BERT, GPT-2)</li>
 *   <li>SiLU - Sigmoid Linear Unit / Swish (used in GPT-NeoX, LLaMA)</li>
 *   <li>ReLU - Rectified Linear Unit (classical)</li>
 *   <li>Sigmoid - logistic function</li>
 * </ul>
 *
 * <p>All activations are decomposed into primitive operations for backend flexibility,
 * allowing backends to fuse operations as appropriate.
 *
 * <p>Example:
 * <pre>{@code
 * // Softmax for attention scores
 * TypedTensor<Rank4, F32, Cpu> scores = ...;  // [batch, heads, seq, seq]
 * TypedTensor<Rank4, F32, Cpu> probs = ActivationOps.softmax(scores, -1);
 *
 * // GELU for feed-forward network
 * TypedTensor<Rank3, F32, Cpu> hidden = ...;  // [batch, seq, ffn_dim]
 * TypedTensor<Rank3, F32, Cpu> activated = ActivationOps.gelu(hidden);
 * }</pre>
 */
public final class ActivationOps {

    private ActivationOps() {
        // Utility class
    }

    // GELU approximation constants
    private static final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);
    private static final double GELU_COEFF = 0.044715;

    // ==================== Softmax ====================

    /**
     * Applies softmax along the last dimension.
     *
     * <p>softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
     *
     * <p>Uses max-subtract for numerical stability (prevents overflow).
     *
     * @param input the input tensor
     * @param <V> device type
     * @return tensor with softmax applied along last dimension
     */
    public static <V extends DeviceTag>
    TypedTensor<Vector, F32, V> softmax(TypedTensor<Vector, F32, V> input) {
        int[] dims = input.dimensions();
        int len = dims[0];

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        softmaxVectorF32(input.underlying().data(), result.data(), len);

        return TypedTensor.from(result, new Vector(len), F32.INSTANCE, input.deviceType());
    }

    /**
     * Applies softmax along axis 1 (each row independently).
     *
     * @param input the input matrix [M, N]
     * @param <V> device type
     * @return matrix with softmax applied to each row
     */
    public static <V extends DeviceTag>
    TypedTensor<Matrix, F32, V> softmaxRows(TypedTensor<Matrix, F32, V> input) {
        int[] dims = input.dimensions();
        int rows = dims[0];
        int cols = dims[1];

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        softmaxMatrixRowsF32(input.underlying().data(), result.data(), rows, cols);

        return TypedTensor.from(result, new Matrix(rows, cols), F32.INSTANCE, input.deviceType());
    }

    /**
     * Applies softmax along the last dimension for Rank3 tensors.
     *
     * <p>For input [B, S, D], applies softmax independently to each [D] vector.
     *
     * @param input the input tensor [B, S, D]
     * @param <V> device type
     * @return tensor with softmax applied to each [D] slice
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank3, F32, V> softmaxRank3(TypedTensor<Rank3, F32, V> input) {
        int[] dims = input.dimensions();
        int batch = dims[0];
        int seq = dims[1];
        int hidden = dims[2];

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        softmaxRank3F32(input.underlying().data(), result.data(), batch, seq, hidden);

        return TypedTensor.from(result, new Rank3(batch, seq, hidden), F32.INSTANCE, input.deviceType());
    }

    /**
     * Applies softmax along the last dimension for Rank4 tensors (attention scores).
     *
     * <p>For input [B, H, S, S], applies softmax independently to each [S] vector
     * in the last dimension. This is the standard attention softmax.
     *
     * @param input the attention scores [B, H, S_q, S_k]
     * @param <V> device type
     * @return attention probabilities with softmax applied
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank4, F32, V> softmaxRank4(TypedTensor<Rank4, F32, V> input) {
        int[] dims = input.dimensions();
        int batch = dims[0];
        int heads = dims[1];
        int seqQ = dims[2];
        int seqK = dims[3];

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        softmaxRank4F32(input.underlying().data(), result.data(), batch, heads, seqQ, seqK);

        return TypedTensor.from(result, new Rank4(batch, heads, seqQ, seqK), F32.INSTANCE, input.deviceType());
    }

    // ==================== GELU ====================

    /**
     * Applies GELU (Gaussian Error Linear Unit) activation.
     *
     * <p>GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     *
     * <p>This is the tanh approximation used in BERT and GPT-2.
     *
     * @param input the input tensor
     * @param <S> shape type
     * @param <V> device type
     * @return tensor with GELU applied elementwise
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, F32, V> gelu(TypedTensor<S, F32, V> input) {
        long count = input.elementCount();
        Tensor result = Tensor.zeros(ScalarType.F32, input.dimensions());

        geluF32(input.underlying().data(), result.data(), count);

        return TypedTensor.from(result, input.shapeType(), F32.INSTANCE, input.deviceType());
    }

    /**
     * Applies GELU activation (double precision).
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, F64, V> geluF64(TypedTensor<S, F64, V> input) {
        long count = input.elementCount();
        Tensor result = Tensor.zeros(ScalarType.F64, input.dimensions());

        geluF64Impl(input.underlying().data(), result.data(), count);

        return TypedTensor.from(result, input.shapeType(), F64.INSTANCE, input.deviceType());
    }

    // ==================== SiLU (Swish) ====================

    /**
     * Applies SiLU (Sigmoid Linear Unit) activation, also known as Swish.
     *
     * <p>SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
     *
     * <p>Used in GPT-NeoX, LLaMA, and other modern transformers.
     *
     * @param input the input tensor
     * @param <S> shape type
     * @param <V> device type
     * @return tensor with SiLU applied elementwise
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, F32, V> silu(TypedTensor<S, F32, V> input) {
        long count = input.elementCount();
        Tensor result = Tensor.zeros(ScalarType.F32, input.dimensions());

        siluF32(input.underlying().data(), result.data(), count);

        return TypedTensor.from(result, input.shapeType(), F32.INSTANCE, input.deviceType());
    }

    /**
     * Applies SiLU activation (double precision).
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, F64, V> siluF64(TypedTensor<S, F64, V> input) {
        long count = input.elementCount();
        Tensor result = Tensor.zeros(ScalarType.F64, input.dimensions());

        siluF64Impl(input.underlying().data(), result.data(), count);

        return TypedTensor.from(result, input.shapeType(), F64.INSTANCE, input.deviceType());
    }

    // ==================== ReLU ====================

    /**
     * Applies ReLU (Rectified Linear Unit) activation.
     *
     * <p>ReLU(x) = max(0, x)
     *
     * @param input the input tensor
     * @param <S> shape type
     * @param <V> device type
     * @return tensor with ReLU applied elementwise
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, F32, V> relu(TypedTensor<S, F32, V> input) {
        long count = input.elementCount();
        Tensor result = Tensor.zeros(ScalarType.F32, input.dimensions());

        reluF32(input.underlying().data(), result.data(), count);

        return TypedTensor.from(result, input.shapeType(), F32.INSTANCE, input.deviceType());
    }

    // ==================== Sigmoid ====================

    /**
     * Applies sigmoid activation.
     *
     * <p>sigmoid(x) = 1 / (1 + exp(-x))
     *
     * @param input the input tensor
     * @param <S> shape type
     * @param <V> device type
     * @return tensor with sigmoid applied elementwise
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, F32, V> sigmoid(TypedTensor<S, F32, V> input) {
        long count = input.elementCount();
        Tensor result = Tensor.zeros(ScalarType.F32, input.dimensions());

        sigmoidF32(input.underlying().data(), result.data(), count);

        return TypedTensor.from(result, input.shapeType(), F32.INSTANCE, input.deviceType());
    }

    // ==================== Tanh ====================

    /**
     * Applies tanh activation.
     *
     * @param input the input tensor
     * @param <S> shape type
     * @param <V> device type
     * @return tensor with tanh applied elementwise
     */
    public static <S extends Shape, V extends DeviceTag>
    TypedTensor<S, F32, V> tanh(TypedTensor<S, F32, V> input) {
        long count = input.elementCount();
        Tensor result = Tensor.zeros(ScalarType.F32, input.dimensions());

        tanhF32(input.underlying().data(), result.data(), count);

        return TypedTensor.from(result, input.shapeType(), F32.INSTANCE, input.deviceType());
    }

    // ==================== Internal Implementation ====================

    // Softmax implementations
    private static void softmaxVectorF32(MemorySegment src, MemorySegment dst, int len) {
        // Find max for numerical stability
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < len; i++) {
            float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            if (val > max) max = val;
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float expVal = (float) Math.exp(val - max);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, expVal);
            sum += expVal;
        }

        // Normalize
        for (int i = 0; i < len; i++) {
            float val = dst.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, val / sum);
        }
    }

    private static void softmaxMatrixRowsF32(MemorySegment src, MemorySegment dst, int rows, int cols) {
        for (int r = 0; r < rows; r++) {
            long offset = (long) r * cols;

            // Find max
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) {
                float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + c);
                if (val > max) max = val;
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int c = 0; c < cols; c++) {
                float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + c);
                float expVal = (float) Math.exp(val - max);
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + c, expVal);
                sum += expVal;
            }

            // Normalize
            for (int c = 0; c < cols; c++) {
                float val = dst.getAtIndex(ValueLayout.JAVA_FLOAT, offset + c);
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + c, val / sum);
            }
        }
    }

    private static void softmaxRank3F32(MemorySegment src, MemorySegment dst,
                                        int batch, int seq, int hidden) {
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                long offset = ((long) b * seq + s) * hidden;

                // Find max
                float max = Float.NEGATIVE_INFINITY;
                for (int h = 0; h < hidden; h++) {
                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                    if (val > max) max = val;
                }

                // Compute exp and sum
                float sum = 0.0f;
                for (int h = 0; h < hidden; h++) {
                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                    float expVal = (float) Math.exp(val - max);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + h, expVal);
                    sum += expVal;
                }

                // Normalize
                for (int h = 0; h < hidden; h++) {
                    float val = dst.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + h, val / sum);
                }
            }
        }
    }

    private static void softmaxRank4F32(MemorySegment src, MemorySegment dst,
                                        int batch, int heads, int seqQ, int seqK) {
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int sq = 0; sq < seqQ; sq++) {
                    long offset = (((long) b * heads + h) * seqQ + sq) * seqK;

                    // Find max
                    float max = Float.NEGATIVE_INFINITY;
                    for (int sk = 0; sk < seqK; sk++) {
                        float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + sk);
                        if (val > max) max = val;
                    }

                    // Compute exp and sum
                    float sum = 0.0f;
                    for (int sk = 0; sk < seqK; sk++) {
                        float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + sk);
                        float expVal = (float) Math.exp(val - max);
                        dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + sk, expVal);
                        sum += expVal;
                    }

                    // Normalize
                    for (int sk = 0; sk < seqK; sk++) {
                        float val = dst.getAtIndex(ValueLayout.JAVA_FLOAT, offset + sk);
                        dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + sk, val / sum);
                    }
                }
            }
        }
    }

    // GELU implementations
    private static void geluF32(MemorySegment src, MemorySegment dst, long count) {
        float sqrt2OverPi = (float) SQRT_2_OVER_PI;
        float coeff = (float) GELU_COEFF;

        for (long i = 0; i < count; i++) {
            float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            float inner = sqrt2OverPi * (x + coeff * x * x * x);
            float result = 0.5f * x * (1.0f + (float) Math.tanh(inner));
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, result);
        }
    }

    private static void geluF64Impl(MemorySegment src, MemorySegment dst, long count) {
        for (long i = 0; i < count; i++) {
            double x = src.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            double inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
            double result = 0.5 * x * (1.0 + Math.tanh(inner));
            dst.setAtIndex(ValueLayout.JAVA_DOUBLE, i, result);
        }
    }

    // SiLU implementations
    private static void siluF32(MemorySegment src, MemorySegment dst, long count) {
        for (long i = 0; i < count; i++) {
            float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            float result = x / (1.0f + (float) Math.exp(-x));
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, result);
        }
    }

    private static void siluF64Impl(MemorySegment src, MemorySegment dst, long count) {
        for (long i = 0; i < count; i++) {
            double x = src.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
            double result = x / (1.0 + Math.exp(-x));
            dst.setAtIndex(ValueLayout.JAVA_DOUBLE, i, result);
        }
    }

    // ReLU implementation
    private static void reluF32(MemorySegment src, MemorySegment dst, long count) {
        for (long i = 0; i < count; i++) {
            float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, Math.max(0.0f, x));
        }
    }

    // Sigmoid implementation
    private static void sigmoidF32(MemorySegment src, MemorySegment dst, long count) {
        for (long i = 0; i < count; i++) {
            float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f / (1.0f + (float) Math.exp(-x)));
        }
    }

    // Tanh implementation
    private static void tanhF32(MemorySegment src, MemorySegment dst, long count) {
        for (long i = 0; i < count; i++) {
            float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.tanh(x));
        }
    }
}
