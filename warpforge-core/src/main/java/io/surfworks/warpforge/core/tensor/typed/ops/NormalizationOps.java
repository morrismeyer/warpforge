package io.surfworks.warpforge.core.tensor.typed.ops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Type-safe normalization operations for tensors.
 *
 * <p>These operations implement normalization layers used in transformer models:
 * <ul>
 *   <li>LayerNorm - Layer Normalization (BERT, GPT-2, ViT)</li>
 *   <li>RMSNorm - Root Mean Square Normalization (LLaMA, Gemma)</li>
 * </ul>
 *
 * <p>All normalizations are decomposed into primitive operations:
 * <ul>
 *   <li>LayerNorm: mean → variance → normalize → scale → shift</li>
 *   <li>RMSNorm: rms → normalize → scale</li>
 * </ul>
 *
 * <p>This decomposition allows backends to fuse operations as appropriate while
 * maintaining a clean separation of concerns.
 *
 * <p>Example:
 * <pre>{@code
 * // LayerNorm for transformer hidden states
 * TypedTensor<Rank3, F32, Cpu> hidden = ...;  // [batch, seq, hidden_dim]
 * TypedTensor<Vector, F32, Cpu> weight = ...;  // [hidden_dim] - gamma
 * TypedTensor<Vector, F32, Cpu> bias = ...;    // [hidden_dim] - beta
 *
 * TypedTensor<Rank3, F32, Cpu> normalized = NormalizationOps.layerNorm(
 *     hidden, weight, bias, 1e-5f);
 * }</pre>
 */
public final class NormalizationOps {

    private NormalizationOps() {
        // Utility class
    }

    // ==================== Layer Normalization ====================

    /**
     * Applies Layer Normalization to a Vector.
     *
     * <p>LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
     *
     * @param input the input vector [D]
     * @param weight scale parameter gamma [D]
     * @param bias shift parameter beta [D]
     * @param epsilon small constant for numerical stability
     * @param <V> device type
     * @return normalized vector
     */
    public static <V extends DeviceTag>
    TypedTensor<Vector, F32, V> layerNormVector(
            TypedTensor<Vector, F32, V> input,
            TypedTensor<Vector, F32, V> weight,
            TypedTensor<Vector, F32, V> bias,
            float epsilon) {

        int[] dims = input.dimensions();
        int dim = dims[0];

        validateNormParams(weight.dimensions()[0], bias.dimensions()[0], dim);

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        layerNormVectorF32(
                input.underlying().data(),
                weight.underlying().data(),
                bias.underlying().data(),
                result.data(),
                dim, epsilon);

        return TypedTensor.from(result, new Vector(dim), F32.INSTANCE, input.deviceType());
    }

    /**
     * Applies Layer Normalization along the last dimension of a Matrix.
     *
     * <p>For input [M, D], normalizes each [D] vector independently.
     *
     * @param input the input matrix [M, D]
     * @param weight scale parameter gamma [D]
     * @param bias shift parameter beta [D]
     * @param epsilon small constant for numerical stability
     * @param <V> device type
     * @return normalized matrix
     */
    public static <V extends DeviceTag>
    TypedTensor<Matrix, F32, V> layerNormMatrix(
            TypedTensor<Matrix, F32, V> input,
            TypedTensor<Vector, F32, V> weight,
            TypedTensor<Vector, F32, V> bias,
            float epsilon) {

        int[] dims = input.dimensions();
        int rows = dims[0];
        int cols = dims[1];

        validateNormParams(weight.dimensions()[0], bias.dimensions()[0], cols);

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        layerNormMatrixF32(
                input.underlying().data(),
                weight.underlying().data(),
                bias.underlying().data(),
                result.data(),
                rows, cols, epsilon);

        return TypedTensor.from(result, new Matrix(rows, cols), F32.INSTANCE, input.deviceType());
    }

    /**
     * Applies Layer Normalization along the last dimension of a Rank3 tensor.
     *
     * <p>For input [B, S, D], normalizes each [D] vector independently.
     * This is the standard LayerNorm used in transformers.
     *
     * @param input the input tensor [B, S, D]
     * @param weight scale parameter gamma [D]
     * @param bias shift parameter beta [D]
     * @param epsilon small constant for numerical stability (typically 1e-5 or 1e-6)
     * @param <V> device type
     * @return normalized tensor
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank3, F32, V> layerNorm(
            TypedTensor<Rank3, F32, V> input,
            TypedTensor<Vector, F32, V> weight,
            TypedTensor<Vector, F32, V> bias,
            float epsilon) {

        int[] dims = input.dimensions();
        int batch = dims[0];
        int seq = dims[1];
        int hidden = dims[2];

        validateNormParams(weight.dimensions()[0], bias.dimensions()[0], hidden);

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        layerNormRank3F32(
                input.underlying().data(),
                weight.underlying().data(),
                bias.underlying().data(),
                result.data(),
                batch, seq, hidden, epsilon);

        return TypedTensor.from(result, new Rank3(batch, seq, hidden), F32.INSTANCE, input.deviceType());
    }

    /**
     * Applies Layer Normalization without affine transformation.
     *
     * <p>LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps)
     *
     * <p>Use this variant when you want to apply weight/bias separately.
     *
     * @param input the input tensor [B, S, D]
     * @param epsilon small constant for numerical stability
     * @param <V> device type
     * @return normalized tensor (without scale/shift)
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank3, F32, V> layerNormNoAffine(
            TypedTensor<Rank3, F32, V> input,
            float epsilon) {

        int[] dims = input.dimensions();
        int batch = dims[0];
        int seq = dims[1];
        int hidden = dims[2];

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        layerNormNoAffineRank3F32(
                input.underlying().data(),
                result.data(),
                batch, seq, hidden, epsilon);

        return TypedTensor.from(result, new Rank3(batch, seq, hidden), F32.INSTANCE, input.deviceType());
    }

    // ==================== RMS Normalization ====================

    /**
     * Applies RMS (Root Mean Square) Normalization to a Vector.
     *
     * <p>RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
     *
     * <p>RMSNorm is simpler than LayerNorm (no mean subtraction, no bias)
     * and is used in LLaMA, Gemma, and other modern architectures.
     *
     * @param input the input vector [D]
     * @param weight scale parameter gamma [D]
     * @param epsilon small constant for numerical stability
     * @param <V> device type
     * @return normalized vector
     */
    public static <V extends DeviceTag>
    TypedTensor<Vector, F32, V> rmsNormVector(
            TypedTensor<Vector, F32, V> input,
            TypedTensor<Vector, F32, V> weight,
            float epsilon) {

        int[] dims = input.dimensions();
        int dim = dims[0];

        if (weight.dimensions()[0] != dim) {
            throw new IllegalArgumentException(
                    "Weight dimension must match input: expected " + dim + ", got " + weight.dimensions()[0]);
        }

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        rmsNormVectorF32(
                input.underlying().data(),
                weight.underlying().data(),
                result.data(),
                dim, epsilon);

        return TypedTensor.from(result, new Vector(dim), F32.INSTANCE, input.deviceType());
    }

    /**
     * Applies RMS Normalization along the last dimension of a Rank3 tensor.
     *
     * <p>For input [B, S, D], normalizes each [D] vector independently.
     *
     * @param input the input tensor [B, S, D]
     * @param weight scale parameter gamma [D]
     * @param epsilon small constant for numerical stability (typically 1e-5 or 1e-6)
     * @param <V> device type
     * @return normalized tensor
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank3, F32, V> rmsNorm(
            TypedTensor<Rank3, F32, V> input,
            TypedTensor<Vector, F32, V> weight,
            float epsilon) {

        int[] dims = input.dimensions();
        int batch = dims[0];
        int seq = dims[1];
        int hidden = dims[2];

        if (weight.dimensions()[0] != hidden) {
            throw new IllegalArgumentException(
                    "Weight dimension must match hidden dim: expected " + hidden +
                    ", got " + weight.dimensions()[0]);
        }

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        rmsNormRank3F32(
                input.underlying().data(),
                weight.underlying().data(),
                result.data(),
                batch, seq, hidden, epsilon);

        return TypedTensor.from(result, new Rank3(batch, seq, hidden), F32.INSTANCE, input.deviceType());
    }

    /**
     * Applies RMS Normalization without scale.
     *
     * <p>RMSNorm(x) = x / sqrt(mean(x²) + eps)
     *
     * @param input the input tensor [B, S, D]
     * @param epsilon small constant for numerical stability
     * @param <V> device type
     * @return normalized tensor (without scale)
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank3, F32, V> rmsNormNoScale(
            TypedTensor<Rank3, F32, V> input,
            float epsilon) {

        int[] dims = input.dimensions();
        int batch = dims[0];
        int seq = dims[1];
        int hidden = dims[2];

        Tensor result = Tensor.zeros(ScalarType.F32, dims);
        rmsNormNoScaleRank3F32(
                input.underlying().data(),
                result.data(),
                batch, seq, hidden, epsilon);

        return TypedTensor.from(result, new Rank3(batch, seq, hidden), F32.INSTANCE, input.deviceType());
    }

    // ==================== Internal Implementation ====================

    private static void validateNormParams(int weightDim, int biasDim, int expectedDim) {
        if (weightDim != expectedDim) {
            throw new IllegalArgumentException(
                    "Weight dimension must match normalized dimension: expected " + expectedDim +
                    ", got " + weightDim);
        }
        if (biasDim != expectedDim) {
            throw new IllegalArgumentException(
                    "Bias dimension must match normalized dimension: expected " + expectedDim +
                    ", got " + biasDim);
        }
    }

    // LayerNorm implementations
    private static void layerNormVectorF32(MemorySegment src, MemorySegment weight, MemorySegment bias,
                                           MemorySegment dst, int dim, float epsilon) {
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) {
            mean += src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        mean /= dim;

        // Compute variance
        float variance = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = src.getAtIndex(ValueLayout.JAVA_FLOAT, i) - mean;
            variance += diff * diff;
        }
        variance /= dim;

        // Normalize, scale, and shift
        float invStd = 1.0f / (float) Math.sqrt(variance + epsilon);
        for (int i = 0; i < dim; i++) {
            float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float w = weight.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float b = bias.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float normalized = (x - mean) * invStd;
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, normalized * w + b);
        }
    }

    private static void layerNormMatrixF32(MemorySegment src, MemorySegment weight, MemorySegment bias,
                                           MemorySegment dst, int rows, int cols, float epsilon) {
        for (int r = 0; r < rows; r++) {
            long offset = (long) r * cols;

            // Compute mean
            float mean = 0.0f;
            for (int c = 0; c < cols; c++) {
                mean += src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + c);
            }
            mean /= cols;

            // Compute variance
            float variance = 0.0f;
            for (int c = 0; c < cols; c++) {
                float diff = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + c) - mean;
                variance += diff * diff;
            }
            variance /= cols;

            // Normalize, scale, and shift
            float invStd = 1.0f / (float) Math.sqrt(variance + epsilon);
            for (int c = 0; c < cols; c++) {
                float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + c);
                float w = weight.getAtIndex(ValueLayout.JAVA_FLOAT, c);
                float b = bias.getAtIndex(ValueLayout.JAVA_FLOAT, c);
                float normalized = (x - mean) * invStd;
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + c, normalized * w + b);
            }
        }
    }

    private static void layerNormRank3F32(MemorySegment src, MemorySegment weight, MemorySegment bias,
                                          MemorySegment dst, int batch, int seq, int hidden, float epsilon) {
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                long offset = ((long) b * seq + s) * hidden;

                // Compute mean
                float mean = 0.0f;
                for (int h = 0; h < hidden; h++) {
                    mean += src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                }
                mean /= hidden;

                // Compute variance
                float variance = 0.0f;
                for (int h = 0; h < hidden; h++) {
                    float diff = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h) - mean;
                    variance += diff * diff;
                }
                variance /= hidden;

                // Normalize, scale, and shift
                float invStd = 1.0f / (float) Math.sqrt(variance + epsilon);
                for (int h = 0; h < hidden; h++) {
                    float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                    float w = weight.getAtIndex(ValueLayout.JAVA_FLOAT, h);
                    float bVal = bias.getAtIndex(ValueLayout.JAVA_FLOAT, h);
                    float normalized = (x - mean) * invStd;
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + h, normalized * w + bVal);
                }
            }
        }
    }

    private static void layerNormNoAffineRank3F32(MemorySegment src, MemorySegment dst,
                                                   int batch, int seq, int hidden, float epsilon) {
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                long offset = ((long) b * seq + s) * hidden;

                // Compute mean
                float mean = 0.0f;
                for (int h = 0; h < hidden; h++) {
                    mean += src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                }
                mean /= hidden;

                // Compute variance
                float variance = 0.0f;
                for (int h = 0; h < hidden; h++) {
                    float diff = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h) - mean;
                    variance += diff * diff;
                }
                variance /= hidden;

                // Normalize (no scale/shift)
                float invStd = 1.0f / (float) Math.sqrt(variance + epsilon);
                for (int h = 0; h < hidden; h++) {
                    float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + h, (x - mean) * invStd);
                }
            }
        }
    }

    // RMSNorm implementations
    private static void rmsNormVectorF32(MemorySegment src, MemorySegment weight,
                                         MemorySegment dst, int dim, float epsilon) {
        // Compute mean of squares
        float meanSq = 0.0f;
        for (int i = 0; i < dim; i++) {
            float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            meanSq += x * x;
        }
        meanSq /= dim;

        // Normalize and scale
        float invRms = 1.0f / (float) Math.sqrt(meanSq + epsilon);
        for (int i = 0; i < dim; i++) {
            float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float w = weight.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, x * invRms * w);
        }
    }

    private static void rmsNormRank3F32(MemorySegment src, MemorySegment weight,
                                        MemorySegment dst, int batch, int seq, int hidden, float epsilon) {
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                long offset = ((long) b * seq + s) * hidden;

                // Compute mean of squares
                float meanSq = 0.0f;
                for (int h = 0; h < hidden; h++) {
                    float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                    meanSq += x * x;
                }
                meanSq /= hidden;

                // Normalize and scale
                float invRms = 1.0f / (float) Math.sqrt(meanSq + epsilon);
                for (int h = 0; h < hidden; h++) {
                    float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                    float w = weight.getAtIndex(ValueLayout.JAVA_FLOAT, h);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + h, x * invRms * w);
                }
            }
        }
    }

    private static void rmsNormNoScaleRank3F32(MemorySegment src, MemorySegment dst,
                                               int batch, int seq, int hidden, float epsilon) {
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                long offset = ((long) b * seq + s) * hidden;

                // Compute mean of squares
                float meanSq = 0.0f;
                for (int h = 0; h < hidden; h++) {
                    float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                    meanSq += x * x;
                }
                meanSq /= hidden;

                // Normalize (no scale)
                float invRms = 1.0f / (float) Math.sqrt(meanSq + epsilon);
                for (int h = 0; h < hidden; h++) {
                    float x = src.getAtIndex(ValueLayout.JAVA_FLOAT, offset + h);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + h, x * invRms);
                }
            }
        }
    }
}
