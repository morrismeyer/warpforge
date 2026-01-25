package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CustomCallOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.custom_call.
 *
 * <p>Custom calls invoke external functions identified by a call_target_name.
 * This kernel maintains a registry of known custom call implementations.
 * Unknown custom calls throw an error.
 */
public final class CustomCallKernel implements OpKernel {

    private final Map<String, BiFunction<CustomCallOp, List<Tensor>, List<Tensor>>> handlers =
        new ConcurrentHashMap<>();

    public CustomCallKernel() {
        registerDefaultHandlers();
    }

    /**
     * Register a handler for a custom call target.
     */
    public void registerHandler(String targetName,
                                 BiFunction<CustomCallOp, List<Tensor>, List<Tensor>> handler) {
        handlers.put(targetName, handler);
    }

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        CustomCallOp customOp = (CustomCallOp) op;
        String targetName = customOp.callTarget();

        BiFunction<CustomCallOp, List<Tensor>, List<Tensor>> handler = handlers.get(targetName);
        if (handler != null) {
            return handler.apply(customOp, inputs);
        }

        // Check for common patterns that can be handled generically
        if (targetName.startsWith("__") && targetName.endsWith("__")) {
            // Likely an internal marker that can be ignored
            return inputs.isEmpty() ? List.of(Tensor.zeros(new int[]{1})) :
                   List.of(inputs.get(0).copy());
        }

        throw new UnsupportedOperationException(
            "Unknown custom_call target: " + targetName + ". " +
            "Register a handler using CustomCallKernel.registerHandler()");
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.CustomCallOp;
    }

    private void registerDefaultHandlers() {
        // Identity passthrough for debug markers
        handlers.put("debug_marker", (op, inputs) ->
            inputs.isEmpty() ? List.of(Tensor.zeros(new int[]{1})) :
                              List.of(inputs.get(0).copy()));

        // CUDA synchronization (no-op on CPU)
        handlers.put("cuda_stream_synchronize", (op, inputs) ->
            List.of(Tensor.zeros(new int[]{1})));

        // Memory allocation hints (no-op on CPU)
        handlers.put("memory_allocation_hint", (op, inputs) ->
            inputs.isEmpty() ? List.of(Tensor.zeros(new int[]{1})) :
                              List.of(inputs.get(0).copy()));

        // ==================== Transformer Operations ====================

        // LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
        handlers.put("layer_norm", this::executeLayerNorm);

        // Softmax: exp(x - max) / sum(exp(x - max))
        handlers.put("softmax", this::executeSoftmax);

        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        handlers.put("gelu", this::executeGelu);

        // SiLU: x * sigmoid(x)
        handlers.put("silu", this::executeSilu);

        // Embedding: lookup rows from embedding table
        handlers.put("embedding", this::executeEmbedding);

        // BatchNorm: standard batch normalization
        handlers.put("batch_norm", this::executeBatchNorm);

        // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        handlers.put("rms_norm", this::executeRmsNorm);
    }

    // ==================== Transformer Operation Implementations ====================

    private List<Tensor> executeLayerNorm(CustomCallOp op, List<Tensor> inputs) {
        // inputs[0] = input tensor
        // inputs[1] = weight (gamma) - optional
        // inputs[2] = bias (beta) - optional
        Tensor input = inputs.get(0);
        int[] shape = input.shape();
        int rank = shape.length;

        float eps = DEFAULT_EPS;

        // Normalize over the last dimension(s) - typically last dim for LayerNorm
        int normalizedSize = shape[rank - 1];
        int batchSize = 1;
        for (int i = 0; i < rank - 1; i++) {
            batchSize *= shape[i];
        }

        Tensor result = Tensor.zeros(ScalarType.F32, shape);
        MemorySegment inData = input.data();
        MemorySegment outData = result.data();

        for (int b = 0; b < batchSize; b++) {
            long offset = (long) b * normalizedSize;

            // Compute mean
            float sum = 0;
            for (int i = 0; i < normalizedSize; i++) {
                sum += inData.getAtIndex(ValueLayout.JAVA_FLOAT, offset + i);
            }
            float mean = sum / normalizedSize;

            // Compute variance
            float varSum = 0;
            for (int i = 0; i < normalizedSize; i++) {
                float diff = inData.getAtIndex(ValueLayout.JAVA_FLOAT, offset + i) - mean;
                varSum += diff * diff;
            }
            float variance = varSum / normalizedSize;
            float invStd = 1.0f / (float) Math.sqrt(variance + eps);

            // Normalize and apply scale/shift if provided
            for (int i = 0; i < normalizedSize; i++) {
                float x = inData.getAtIndex(ValueLayout.JAVA_FLOAT, offset + i);
                float normalized = (x - mean) * invStd;

                // Apply weight (gamma) if provided
                if (inputs.size() > 1) {
                    float gamma = inputs.get(1).data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    normalized *= gamma;
                }

                // Apply bias (beta) if provided
                if (inputs.size() > 2) {
                    float beta = inputs.get(2).data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    normalized += beta;
                }

                outData.setAtIndex(ValueLayout.JAVA_FLOAT, offset + i, normalized);
            }
        }

        return List.of(result);
    }

    private List<Tensor> executeSoftmax(CustomCallOp op, List<Tensor> inputs) {
        Tensor input = inputs.get(0);
        int[] shape = input.shape();
        int rank = shape.length;

        // Default: softmax over last dimension
        int dim = rank - 1;

        int outerSize = 1;
        for (int i = 0; i < dim; i++) {
            outerSize *= shape[i];
        }
        int dimSize = shape[dim];
        int innerSize = 1;
        for (int i = dim + 1; i < rank; i++) {
            innerSize *= shape[i];
        }

        Tensor result = Tensor.zeros(ScalarType.F32, shape);
        MemorySegment inData = input.data();
        MemorySegment outData = result.data();

        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                // Find max for numerical stability
                float maxVal = Float.NEGATIVE_INFINITY;
                for (int d = 0; d < dimSize; d++) {
                    long idx = ((long) outer * dimSize + d) * innerSize + inner;
                    float val = inData.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                    if (val > maxVal) maxVal = val;
                }

                // Compute exp(x - max) and sum
                float sumExp = 0;
                for (int d = 0; d < dimSize; d++) {
                    long idx = ((long) outer * dimSize + d) * innerSize + inner;
                    float val = inData.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                    sumExp += (float) Math.exp(val - maxVal);
                }

                // Normalize
                for (int d = 0; d < dimSize; d++) {
                    long idx = ((long) outer * dimSize + d) * innerSize + inner;
                    float val = inData.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                    float softmax = (float) Math.exp(val - maxVal) / sumExp;
                    outData.setAtIndex(ValueLayout.JAVA_FLOAT, idx, softmax);
                }
            }
        }

        return List.of(result);
    }

    private List<Tensor> executeGelu(CustomCallOp op, List<Tensor> inputs) {
        Tensor input = inputs.get(0);
        int[] shape = input.shape();
        int size = 1;
        for (int dim : shape) size *= dim;

        // Default to tanh approximation (faster, commonly used)
        String approximate = "tanh";

        Tensor result = Tensor.zeros(ScalarType.F32, shape);
        MemorySegment inData = input.data();
        MemorySegment outData = result.data();

        // Constants for tanh approximation
        final float SQRT_2_OVER_PI = 0.7978845608f;  // sqrt(2/pi)
        final float COEFF = 0.044715f;

        for (int i = 0; i < size; i++) {
            float x = inData.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float gelu;

            if ("tanh".equals(approximate)) {
                // Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                gelu = 0.5f * x * (1.0f + (float) Math.tanh(inner));
            } else {
                // Exact: x * 0.5 * (1 + erf(x / sqrt(2)))
                gelu = x * 0.5f * (1.0f + (float) erf(x / Math.sqrt(2)));
            }

            outData.setAtIndex(ValueLayout.JAVA_FLOAT, i, gelu);
        }

        return List.of(result);
    }

    private List<Tensor> executeSilu(CustomCallOp op, List<Tensor> inputs) {
        Tensor input = inputs.get(0);
        int[] shape = input.shape();
        int size = 1;
        for (int dim : shape) size *= dim;

        Tensor result = Tensor.zeros(ScalarType.F32, shape);
        MemorySegment inData = input.data();
        MemorySegment outData = result.data();

        for (int i = 0; i < size; i++) {
            float x = inData.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-x));
            float silu = x * sigmoid;
            outData.setAtIndex(ValueLayout.JAVA_FLOAT, i, silu);
        }

        return List.of(result);
    }

    private List<Tensor> executeEmbedding(CustomCallOp op, List<Tensor> inputs) {
        // inputs[0] = embedding table [vocab_size, embed_dim]
        // inputs[1] = indices [batch, seq] or [seq]
        Tensor embedTable = inputs.get(0);
        Tensor indices = inputs.get(1);

        int[] tableShape = embedTable.shape();
        int embedDim = tableShape[1];

        int[] indexShape = indices.shape();
        int numIndices = 1;
        for (int dim : indexShape) numIndices *= dim;

        // Output shape: indexShape + [embedDim]
        int[] outShape = new int[indexShape.length + 1];
        System.arraycopy(indexShape, 0, outShape, 0, indexShape.length);
        outShape[indexShape.length] = embedDim;

        Tensor result = Tensor.zeros(ScalarType.F32, outShape);
        MemorySegment tableData = embedTable.data();
        MemorySegment indexData = indices.data();
        MemorySegment outData = result.data();

        for (int i = 0; i < numIndices; i++) {
            int idx = (int) indexData.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            long srcOffset = (long) idx * embedDim;
            long dstOffset = (long) i * embedDim;

            for (int j = 0; j < embedDim; j++) {
                float val = tableData.getAtIndex(ValueLayout.JAVA_FLOAT, srcOffset + j);
                outData.setAtIndex(ValueLayout.JAVA_FLOAT, dstOffset + j, val);
            }
        }

        return List.of(result);
    }

    private List<Tensor> executeBatchNorm(CustomCallOp op, List<Tensor> inputs) {
        // inputs[0] = input [N, C, ...] or [N, ..., C]
        // inputs[1] = weight (gamma)
        // inputs[2] = bias (beta)
        // inputs[3] = running_mean
        // inputs[4] = running_var
        Tensor input = inputs.get(0);
        int[] shape = input.shape();

        float eps = DEFAULT_EPS;

        // Assume NCHW format, normalize over C dimension
        int N = shape[0];
        int C = shape[1];
        int spatialSize = 1;
        for (int i = 2; i < shape.length; i++) {
            spatialSize *= shape[i];
        }

        Tensor result = Tensor.zeros(ScalarType.F32, shape);
        MemorySegment inData = input.data();
        MemorySegment outData = result.data();

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                float gamma = inputs.size() > 1 ?
                    inputs.get(1).data().getAtIndex(ValueLayout.JAVA_FLOAT, c) : 1.0f;
                float beta = inputs.size() > 2 ?
                    inputs.get(2).data().getAtIndex(ValueLayout.JAVA_FLOAT, c) : 0.0f;
                float mean = inputs.size() > 3 ?
                    inputs.get(3).data().getAtIndex(ValueLayout.JAVA_FLOAT, c) : 0.0f;
                float var = inputs.size() > 4 ?
                    inputs.get(4).data().getAtIndex(ValueLayout.JAVA_FLOAT, c) : 1.0f;

                float invStd = 1.0f / (float) Math.sqrt(var + eps);

                for (int s = 0; s < spatialSize; s++) {
                    long idx = ((long) n * C + c) * spatialSize + s;
                    float x = inData.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                    float normalized = (x - mean) * invStd * gamma + beta;
                    outData.setAtIndex(ValueLayout.JAVA_FLOAT, idx, normalized);
                }
            }
        }

        return List.of(result);
    }

    private List<Tensor> executeRmsNorm(CustomCallOp op, List<Tensor> inputs) {
        Tensor input = inputs.get(0);
        int[] shape = input.shape();
        int rank = shape.length;

        float eps = DEFAULT_EPS;

        int normalizedSize = shape[rank - 1];
        int batchSize = 1;
        for (int i = 0; i < rank - 1; i++) {
            batchSize *= shape[i];
        }

        Tensor result = Tensor.zeros(ScalarType.F32, shape);
        MemorySegment inData = input.data();
        MemorySegment outData = result.data();

        for (int b = 0; b < batchSize; b++) {
            long offset = (long) b * normalizedSize;

            // Compute RMS: sqrt(mean(x^2))
            float sumSq = 0;
            for (int i = 0; i < normalizedSize; i++) {
                float x = inData.getAtIndex(ValueLayout.JAVA_FLOAT, offset + i);
                sumSq += x * x;
            }
            float rms = (float) Math.sqrt(sumSq / normalizedSize + eps);
            float invRms = 1.0f / rms;

            // Normalize and apply weight if provided
            for (int i = 0; i < normalizedSize; i++) {
                float x = inData.getAtIndex(ValueLayout.JAVA_FLOAT, offset + i);
                float normalized = x * invRms;

                // Apply weight if provided
                if (inputs.size() > 1) {
                    float weight = inputs.get(1).data().getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    normalized *= weight;
                }

                outData.setAtIndex(ValueLayout.JAVA_FLOAT, offset + i, normalized);
            }
        }

        return List.of(result);
    }

    // ==================== Attribute Helpers ====================
    // Note: CustomCallOp doesn't carry parsed attributes, so we use defaults
    // and derive parameters from input shapes where possible.

    private static final float DEFAULT_EPS = 1e-5f;
    private static final int DEFAULT_DIM = -1;  // Last dimension
    private static final String DEFAULT_APPROXIMATE = "none";

    // ==================== Math Helpers ====================

    /**
     * Error function approximation for GELU exact mode.
     */
    private static double erf(double x) {
        // Abramowitz and Stegun approximation
        double a1 =  0.254829592;
        double a2 = -0.284496736;
        double a3 =  1.421413741;
        double a4 = -1.453152027;
        double a5 =  1.061405429;
        double p  =  0.3275911;

        int sign = x < 0 ? -1 : 1;
        x = Math.abs(x);

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

        return sign * y;
    }
}
