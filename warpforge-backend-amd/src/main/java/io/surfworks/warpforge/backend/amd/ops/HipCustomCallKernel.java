package io.surfworks.warpforge.backend.amd.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CustomCallOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.amd.hip.HipContext;
import io.surfworks.warpforge.backend.amd.hip.HipKernels;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * HIP kernel for stablehlo.custom_call operations.
 *
 * <p>Handles transformer-specific operations:
 * <ul>
 *   <li>gelu - Gaussian Error Linear Unit activation</li>
 *   <li>silu - Sigmoid Linear Unit (Swish) activation</li>
 *   <li>softmax - Softmax normalization over last dimension</li>
 *   <li>layer_norm - Layer normalization over last dimension</li>
 *   <li>embedding - Embedding table lookup</li>
 * </ul>
 *
 * <p><b>Note:</b> This kernel requires HIPRTC (HIP Runtime Compilation) to compile
 * the HIP C++ source code at runtime. Until HIPRTC FFM bindings are integrated,
 * operations will throw UnsupportedOperationException with helpful messages.
 *
 * <p>The HIP C++ source for all operations is already implemented in
 * {@link HipKernels} and ready to compile once HIPRTC is available.
 *
 * @see io.surfworks.warpforge.backend.nvidia.ops.CudaCustomCallKernel NVIDIA equivalent
 */
public final class HipCustomCallKernel implements HipOpKernel {

    private final HipContext context;
    private final int salt;

    // Kernel state (lazy initialized when HIPRTC is available)
    private final Map<String, KernelState> kernelCache = new ConcurrentHashMap<>();

    private static class KernelState {
        long module;
        long function;
    }

    public HipCustomCallKernel(HipContext context, int salt) {
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
                "Unknown custom_call target for HIP: " + targetName);
        };
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof CustomCallOp;
    }

    // ==================== GELU ====================

    private List<Tensor> executeGelu(CustomCallOp op, List<Tensor> inputs) {
        // HIPRTC is required to compile the HIP C++ kernel
        // Source code is ready in HipKernels.generateGeluF32(salt)
        throw new UnsupportedOperationException(
            "GELU on AMD GPU requires HIPRTC integration. " +
            "HIP C++ source is ready in HipKernels.generateGeluF32(). " +
            "Use CPU backend until HIPRTC is integrated.");
    }

    // ==================== SiLU ====================

    private List<Tensor> executeSilu(CustomCallOp op, List<Tensor> inputs) {
        throw new UnsupportedOperationException(
            "SiLU on AMD GPU requires HIPRTC integration. " +
            "HIP C++ source is ready in HipKernels.generateSiluF32(). " +
            "Use CPU backend until HIPRTC is integrated.");
    }

    // ==================== Softmax ====================

    private List<Tensor> executeSoftmax(CustomCallOp op, List<Tensor> inputs) {
        throw new UnsupportedOperationException(
            "Softmax on AMD GPU requires HIPRTC integration. " +
            "HIP C++ source is ready in HipKernels.generateSoftmaxF32(). " +
            "Use CPU backend until HIPRTC is integrated.");
    }

    // ==================== LayerNorm ====================

    private List<Tensor> executeLayerNorm(CustomCallOp op, List<Tensor> inputs) {
        throw new UnsupportedOperationException(
            "LayerNorm on AMD GPU requires HIPRTC integration. " +
            "HIP C++ source is ready in HipKernels.generateLayerNormF32(). " +
            "Use CPU backend until HIPRTC is integrated.");
    }

    // ==================== Embedding ====================

    private List<Tensor> executeEmbedding(CustomCallOp op, List<Tensor> inputs) {
        throw new UnsupportedOperationException(
            "Embedding on AMD GPU requires HIPRTC integration. " +
            "HIP C++ source is ready in HipKernels.generateEmbeddingF32(). " +
            "Use CPU backend until HIPRTC is integrated.");
    }

    // ==================== RMSNorm ====================

    private List<Tensor> executeRmsNorm(CustomCallOp op, List<Tensor> inputs) {
        throw new UnsupportedOperationException(
            "RMSNorm on AMD GPU requires HIPRTC integration. " +
            "Use CPU backend until HIPRTC is integrated.");
    }

    // ==================== BatchNorm ====================

    private List<Tensor> executeBatchNorm(CustomCallOp op, List<Tensor> inputs) {
        throw new UnsupportedOperationException(
            "BatchNorm custom_call on AMD GPU is not implemented. " +
            "Use BatchNormTrainingOp or BatchNormInferenceOp instead.");
    }

    // ==================== Kernel Management (for future HIPRTC integration) ====================

    /**
     * Ensure kernel is compiled and cached.
     *
     * <p>This method will compile HIP C++ source via HIPRTC once integrated.
     * Currently throws UnsupportedOperationException.
     *
     * @param name Operation name for caching
     * @param functionName Kernel function name
     * @param sourceGenerator Function to generate HIP C++ source
     * @return Kernel state with module and function handles
     */
    private KernelState ensureKernel(String name, String functionName,
                                      java.util.function.IntFunction<String> sourceGenerator) {
        return kernelCache.computeIfAbsent(name, k -> {
            String source = sourceGenerator.apply(salt);
            String moduleName = name + "_f32_module_salt" + salt;

            // TODO: When HIPRTC is integrated:
            // 1. byte[] binary = HiprtcRuntime.compile(source);
            // 2. long module = context.loadModule(moduleName, binary);
            // 3. long function = context.getFunction(module, functionName);
            // 4. return new KernelState(module, function);

            throw new UnsupportedOperationException(
                "HIPRTC integration required to compile HIP C++ kernel: " + name);
        });
    }
}
