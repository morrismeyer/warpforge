package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * CUDA kernel for element-wise comparison operations.
 *
 * <p>Supports all comparison directions: EQ, NE, LT, LE, GT, GE.
 * Output is float32 with 1.0 for true, 0.0 for false.
 */
public final class CompareKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;
    private final Map<StableHloAst.ComparisonDirection, Long> modules = new ConcurrentHashMap<>();
    private final Map<StableHloAst.ComparisonDirection, Long> functions = new ConcurrentHashMap<>();

    public CompareKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensureInitialized(StableHloAst.ComparisonDirection direction) {
        if (functions.containsKey(direction)) {
            return;
        }

        String dirName = direction.name();
        String ptxSource = CudaKernels.generateCompareF32(dirName, salt);
        String moduleName = "compare_" + dirName.toLowerCase() + "_f32_module_salt" + salt;
        String functionName = "compare_" + dirName.toLowerCase() + "_f32";

        long module = context.loadModule(moduleName, ptxSource);
        long function = context.getFunction(module, functionName);

        modules.put(direction, module);
        functions.put(direction, function);
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.CompareOp compareOp)) {
            throw new IllegalArgumentException(
                "Expected CompareOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                "compare requires exactly 2 inputs, got: " + inputs.size());
        }

        StableHloAst.ComparisonDirection direction = compareOp.direction();
        ensureInitialized(direction);

        Tensor lhs = inputs.get(0);
        Tensor rhs = inputs.get(1);

        if (lhs.elementCount() != rhs.elementCount()) {
            throw new IllegalArgumentException(
                "Input tensors must have same element count: " +
                lhs.elementCount() + " vs " + rhs.elementCount());
        }

        int n = (int) lhs.elementCount();
        long byteSize = n * 4L;

        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long dA = context.allocate(byteSize);
        long dB = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dA, lhs.data());
            context.copyToDevice(dB, rhs.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            long function = functions.get(direction);

            // PTX parameter order: (a_ptr, b_ptr, out_ptr, n, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dA, dB, dOut},
                    new int[]{n},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after n
                );
            } else {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dA, dB, dOut},
                    new int[]{n},
                    new float[]{}
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSize);

            return List.of(output);

        } finally {
            context.free(dA);
            context.free(dB);
            context.free(dOut);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.CompareOp;
    }
}
