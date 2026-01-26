package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CustomCallOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;

/**
 * CUDA kernel for stablehlo.custom_call operations.
 *
 * <p>Handles transformer-specific operations:
 * <ul>
 *   <li>gelu - Gaussian Error Linear Unit activation</li>
 *   <li>silu - Sigmoid Linear Unit (Swish) activation</li>
 *   <li>softmax - Softmax normalization over last dimension</li>
 *   <li>layer_norm - Layer normalization over last dimension</li>
 *   <li>embedding - Embedding table lookup</li>
 * </ul>
 */
public final class CudaCustomCallKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;

    // Kernel state (lazy initialized)
    private final Map<String, KernelState> kernelCache = new ConcurrentHashMap<>();

    private static class KernelState {
        long module;
        long function;
    }

    public CudaCustomCallKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        CustomCallOp customOp = (CustomCallOp) op;
        String targetName = customOp.callTarget();

        return switch (targetName) {
            case "gelu" -> executeGelu(customOp, inputs);
            case "silu" -> executeSilu(customOp, inputs);
            case "softmax" -> executeSoftmax(customOp, inputs);
            case "layer_norm" -> executeLayerNorm(customOp, inputs);
            case "embedding" -> executeEmbedding(customOp, inputs);
            case "rms_norm" -> executeRmsNorm(customOp, inputs);
            case "batch_norm" -> executeBatchNorm(customOp, inputs);
            default -> throw new UnsupportedOperationException(
                "Unknown custom_call target for CUDA: " + targetName);
        };
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof CustomCallOp;
    }

    // ==================== GELU ====================

    private List<Tensor> executeGelu(CustomCallOp op, List<Tensor> inputs) {
        Tensor input = inputs.get(0);
        int n = (int) input.elementCount();
        long byteSize = n * 4L;

        KernelState state = ensureKernel("gelu", "gelu_f32", CudaKernels::generateGeluF32);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long dIn = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            context.copyToDevice(dIn, input.data());

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            context.launchKernelWithMixedParams(
                state.function,
                new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                0,
                new long[]{dIn, dOut},
                new int[]{n},
                new float[]{}
            );

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSize);

            return List.of(output);
        } finally {
            context.free(dIn);
            context.free(dOut);
        }
    }

    // ==================== SiLU ====================

    private List<Tensor> executeSilu(CustomCallOp op, List<Tensor> inputs) {
        Tensor input = inputs.get(0);
        int n = (int) input.elementCount();
        long byteSize = n * 4L;

        KernelState state = ensureKernel("silu", "silu_f32", CudaKernels::generateSiluF32);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long dIn = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            context.copyToDevice(dIn, input.data());

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            context.launchKernelWithMixedParams(
                state.function,
                new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                0,
                new long[]{dIn, dOut},
                new int[]{n},
                new float[]{}
            );

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSize);

            return List.of(output);
        } finally {
            context.free(dIn);
            context.free(dOut);
        }
    }

    // ==================== Softmax ====================

    private List<Tensor> executeSoftmax(CustomCallOp op, List<Tensor> inputs) {
        Tensor input = inputs.get(0);
        int[] shape = input.shape();
        int rank = shape.length;

        // Softmax over last dimension
        int cols = shape[rank - 1];
        int rows = 1;
        for (int i = 0; i < rank - 1; i++) {
            rows *= shape[i];
        }

        long byteSize = (long) rows * cols * 4L;

        KernelState state = ensureKernel("softmax", "softmax_f32", CudaKernels::generateSoftmaxF32);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long dIn = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);

        try {
            context.copyToDevice(dIn, input.data());

            // One block per row, 256 threads per block
            int blockSize = 256;
            int gridSize = rows;

            context.launchKernelWithMixedParams(
                state.function,
                new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                0,
                new long[]{dIn, dOut},
                new int[]{rows, cols},
                new float[]{}
            );

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSize);

            return List.of(output);
        } finally {
            context.free(dIn);
            context.free(dOut);
        }
    }

    // ==================== LayerNorm ====================

    private List<Tensor> executeLayerNorm(CustomCallOp op, List<Tensor> inputs) {
        Tensor input = inputs.get(0);
        int[] shape = input.shape();
        int rank = shape.length;

        // LayerNorm over last dimension
        int cols = shape[rank - 1];
        int rows = 1;
        for (int i = 0; i < rank - 1; i++) {
            rows *= shape[i];
        }

        long dataBytes = (long) rows * cols * 4L;
        long paramBytes = (long) cols * 4L;

        KernelState state = ensureKernel("layer_norm", "layer_norm_f32", CudaKernels::generateLayerNormF32);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long dIn = context.allocate(dataBytes);
        long dOut = context.allocate(dataBytes);
        long dGamma = context.allocate(paramBytes);
        long dBeta = context.allocate(paramBytes);

        try {
            context.copyToDevice(dIn, input.data());

            // Gamma (weight) - input[1] if provided, else ones
            if (inputs.size() > 1) {
                context.copyToDevice(dGamma, inputs.get(1).data());
            } else {
                // Fill with 1.0
                try (Tensor ones = Tensor.full(1.0f, cols)) {
                    context.copyToDevice(dGamma, ones.data());
                }
            }

            // Beta (bias) - input[2] if provided, else zeros
            if (inputs.size() > 2) {
                context.copyToDevice(dBeta, inputs.get(2).data());
            } else {
                // Fill with 0.0
                try (Tensor zeros = Tensor.zeros(ScalarType.F32, cols)) {
                    context.copyToDevice(dBeta, zeros.data());
                }
            }

            float eps = 1e-5f;

            // One block per row
            int blockSize = 256;
            int gridSize = rows;

            context.launchKernelWithMixedParams(
                state.function,
                new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                0,
                new long[]{dIn, dOut, dGamma, dBeta},
                new int[]{rows, cols},
                new float[]{eps}
            );

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, dataBytes);

            return List.of(output);
        } finally {
            context.free(dIn);
            context.free(dOut);
            context.free(dGamma);
            context.free(dBeta);
        }
    }

    // ==================== Embedding ====================

    private List<Tensor> executeEmbedding(CustomCallOp op, List<Tensor> inputs) {
        // inputs[0] = indices (int64)
        // inputs[1] = embedding table (float32)
        Tensor indices = inputs.get(0);
        Tensor table = inputs.get(1);

        int[] indicesShape = indices.shape();
        int[] tableShape = table.shape();

        int numIndices = 1;
        for (int dim : indicesShape) numIndices *= dim;

        int embedDim = tableShape[1];

        long indicesBytes = (long) numIndices * 8L;  // int64
        long tableBytes = (long) tableShape[0] * embedDim * 4L;  // float32
        long outBytes = (long) numIndices * embedDim * 4L;

        KernelState state = ensureKernel("embedding", "embedding_f32", CudaKernels::generateEmbeddingF32);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long dIndices = context.allocate(indicesBytes);
        long dTable = context.allocate(tableBytes);
        long dOut = context.allocate(outBytes);

        try {
            context.copyToDevice(dIndices, indices.data());
            context.copyToDevice(dTable, table.data());

            // One thread per embedding lookup
            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(numIndices, blockSize);

            context.launchKernelWithMixedParams(
                state.function,
                new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                0,
                new long[]{dIndices, dTable, dOut},
                new int[]{numIndices, embedDim},
                new float[]{}
            );

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, outBytes);

            return List.of(output);
        } finally {
            context.free(dIndices);
            context.free(dTable);
            context.free(dOut);
        }
    }

    // ==================== RMSNorm ====================

    private List<Tensor> executeRmsNorm(CustomCallOp op, List<Tensor> inputs) {
        // For now, fall back to a simple CPU-style implementation on GPU
        // A proper implementation would need a dedicated PTX kernel
        throw new UnsupportedOperationException(
            "RMSNorm CUDA kernel not yet implemented. Use CPU backend.");
    }

    // ==================== BatchNorm ====================

    private List<Tensor> executeBatchNorm(CustomCallOp op, List<Tensor> inputs) {
        // BatchNorm is already implemented in BatchNormKernel
        // This is a fallback for custom_call version
        throw new UnsupportedOperationException(
            "BatchNorm custom_call should use BatchNormKernel. " +
            "Register BatchNormTrainingOp or BatchNormInferenceOp instead.");
    }

    // ==================== Kernel Management ====================

    private KernelState ensureKernel(String name, String functionName,
                                      java.util.function.IntFunction<String> ptxGenerator) {
        return kernelCache.computeIfAbsent(name, k -> {
            String ptx = ptxGenerator.apply(salt);
            String moduleName = name + "_f32_module_salt" + salt;

            KernelState state = new KernelState();
            state.module = context.loadModule(moduleName, ptx);
            state.function = context.getFunction(state.module, functionName);
            return state;
        });
    }
}
