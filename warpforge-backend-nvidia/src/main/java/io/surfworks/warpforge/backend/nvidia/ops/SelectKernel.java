package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for element-wise select (ternary) operation.
 *
 * <p>select(pred, on_true, on_false) = pred ? on_true : on_false
 * where pred is a float tensor with 0.0 = false, non-zero = true.
 */
public final class SelectKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    public SelectKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        String ptxSource = CudaKernels.generateSelectF32(salt);
        String moduleName = "select_f32_module_salt" + salt;
        String functionName = "select_f32";

        module = context.loadModule(moduleName, ptxSource);
        function = context.getFunction(module, functionName);
        initialized = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.SelectOp)) {
            throw new IllegalArgumentException(
                "Expected SelectOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 3) {
            throw new IllegalArgumentException(
                "select requires exactly 3 inputs (pred, on_true, on_false), got: " + inputs.size());
        }

        ensureInitialized();

        Tensor pred = inputs.get(0);
        Tensor onTrue = inputs.get(1);
        Tensor onFalse = inputs.get(2);

        if (pred.elementCount() != onTrue.elementCount() ||
            pred.elementCount() != onFalse.elementCount()) {
            throw new IllegalArgumentException(
                "Input tensors must have same element count: " +
                pred.elementCount() + ", " + onTrue.elementCount() + ", " + onFalse.elementCount());
        }

        int n = (int) pred.elementCount();
        long byteSize = n * 4L;

        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long dPred = context.allocate(byteSize);
        long dOnTrue = context.allocate(byteSize);
        long dOnFalse = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dPred, pred.data());
            context.copyToDevice(dOnTrue, onTrue.data());
            context.copyToDevice(dOnFalse, onFalse.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            // PTX parameter order: (pred_ptr, on_true_ptr, on_false_ptr, out_ptr, n, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dPred, dOnTrue, dOnFalse, dOut},
                    new int[]{n},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after n
                );
            } else {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dPred, dOnTrue, dOnFalse, dOut},
                    new int[]{n},
                    new float[]{}
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSize);

            return List.of(output);

        } finally {
            context.free(dPred);
            context.free(dOnTrue);
            context.free(dOnFalse);
            context.free(dOut);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.SelectOp;
    }
}
