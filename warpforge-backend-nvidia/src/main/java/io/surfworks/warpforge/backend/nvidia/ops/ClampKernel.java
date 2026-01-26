package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for element-wise clamp operation.
 *
 * <p>clamp(min, operand, max) = max(min, min(operand, max))
 */
public final class ClampKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    public ClampKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        String ptxSource = CudaKernels.generateClampF32(salt);
        String moduleName = "clamp_f32_module_salt" + salt;
        String functionName = "clamp_f32";

        module = context.loadModule(moduleName, ptxSource);
        function = context.getFunction(module, functionName);
        initialized = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.ClampOp)) {
            throw new IllegalArgumentException(
                "Expected ClampOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 3) {
            throw new IllegalArgumentException(
                "clamp requires exactly 3 inputs (min, operand, max), got: " + inputs.size());
        }

        ensureInitialized();

        Tensor min = inputs.get(0);
        Tensor operand = inputs.get(1);
        Tensor max = inputs.get(2);

        if (min.elementCount() != operand.elementCount() ||
            min.elementCount() != max.elementCount()) {
            throw new IllegalArgumentException(
                "Input tensors must have same element count: " +
                min.elementCount() + ", " + operand.elementCount() + ", " + max.elementCount());
        }

        int n = (int) min.elementCount();
        long byteSize = n * 4L;

        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long dMin = context.allocate(byteSize);
        long dOperand = context.allocate(byteSize);
        long dMax = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dMin, min.data());
            context.copyToDevice(dOperand, operand.data());
            context.copyToDevice(dMax, max.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            // PTX parameter order: (min_ptr, operand_ptr, max_ptr, out_ptr, n, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dMin, dOperand, dMax, dOut},
                    new int[]{n},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after n
                );
            } else {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dMin, dOperand, dMax, dOut},
                    new int[]{n},
                    new float[]{}
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSize);

            return List.of(output);

        } finally {
            context.free(dMin);
            context.free(dOperand);
            context.free(dMax);
            context.free(dOut);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.ClampOp;
    }
}
