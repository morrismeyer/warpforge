package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for reshape operation.
 *
 * <p>Reshape changes the shape of a tensor without changing its data.
 * Since the underlying data layout is unchanged, this is essentially
 * a memory copy operation.
 *
 * @see CudaKernels#generateReshapeF32
 */
public final class ReshapeKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    public ReshapeKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        String ptxSource = CudaKernels.generateReshapeF32(salt);
        String moduleName = "reshape_f32_module_salt" + salt;
        String functionName = "reshape_f32";

        module = context.loadModule(moduleName, ptxSource);
        function = context.getFunction(module, functionName);
        initialized = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.ReshapeOp)) {
            throw new IllegalArgumentException(
                "Expected ReshapeOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 1) {
            throw new IllegalArgumentException(
                "Reshape requires exactly 1 input, got: " + inputs.size());
        }

        ensureInitialized();

        Tensor input = inputs.get(0);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        // Validate element counts match
        long inputElements = input.elementCount();
        long outputElements = outputSpec.elementCount();
        if (inputElements != outputElements) {
            throw new IllegalArgumentException(
                "Reshape element count mismatch: input has " + inputElements +
                " elements, output shape requires " + outputElements);
        }

        int n = (int) inputElements;
        long byteSize = n * 4L;

        long dInput = context.allocate(byteSize);
        long dOutput = context.allocate(byteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dInput, input.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            // PTX parameter order: (in_ptr, out_ptr, n, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput},
                    new int[]{n},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after n
                );
            } else {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput},
                    new int[]{n},
                    new float[]{}
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
        return op instanceof StableHloAst.ReshapeOp;
    }

    public int getSalt() {
        return salt;
    }
}
