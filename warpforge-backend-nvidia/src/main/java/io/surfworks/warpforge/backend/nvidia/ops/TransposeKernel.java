package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for transpose operation.
 *
 * <p>Transposes tensor dimensions according to a permutation.
 * Currently optimized for 2D transpose (matrix transpose).
 *
 * @see CudaKernels#generateTranspose2DF32
 */
public final class TransposeKernel implements CudaOpKernel {

    private static final int BLOCK_SIZE = 16;

    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    public TransposeKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        String ptxSource = CudaKernels.generateTranspose2DF32(salt);
        String moduleName = "transpose_2d_f32_module_salt" + salt;
        String functionName = "transpose_2d_f32";

        module = context.loadModule(moduleName, ptxSource);
        function = context.getFunction(module, functionName);
        initialized = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.TransposeOp transposeOp)) {
            throw new IllegalArgumentException(
                "Expected TransposeOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 1) {
            throw new IllegalArgumentException(
                "Transpose requires exactly 1 input, got: " + inputs.size());
        }

        Tensor input = inputs.get(0);
        int[] inputShape = input.shape();
        List<Long> permutation = transposeOp.permutation();

        // Currently only support 2D transpose
        if (inputShape.length != 2) {
            throw new UnsupportedOperationException(
                "Only 2D transpose is currently supported, got " + inputShape.length + "D tensor");
        }

        // Validate permutation is [1, 0] for 2D
        if (permutation.size() != 2 || permutation.get(0) != 1 || permutation.get(1) != 0) {
            throw new UnsupportedOperationException(
                "Only standard 2D transpose (permutation [1, 0]) is supported, got: " + permutation);
        }

        ensureInitialized();

        int rows = inputShape[0];
        int cols = inputShape[1];
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int n = rows * cols;
        long byteSize = n * 4L;

        long dInput = context.allocate(byteSize);
        long dOutput = context.allocate(byteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dInput, input.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            // Launch 2D grid
            int gridX = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int gridY = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dInput, dOutput, dTiming},
                    rows, cols
                );
            } else {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dInput, dOutput},
                    rows, cols
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, byteSize);

            return List.of(output);

        } finally {
            context.free(dInput);
            context.free(dOutput);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.TransposeOp;
    }

    public int getSalt() {
        return salt;
    }
}
