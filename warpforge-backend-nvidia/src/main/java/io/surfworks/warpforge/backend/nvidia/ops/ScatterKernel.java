package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for scatter operation.
 *
 * <p>Scatters updates into an output tensor using indices. Currently supports:
 * - Scatter-add: output[indices[i]] += updates[i]
 *
 * @see CudaKernels#generateScatterAddF32
 */
public final class ScatterKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;

    private long moduleAdd;
    private long functionAdd;
    private boolean initializedAdd;

    public ScatterKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensureAddInitialized() {
        if (initializedAdd) return;
        String ptx = CudaKernels.generateScatterAddF32(salt);
        moduleAdd = context.loadModule("scatter_add_f32_salt" + salt, ptx);
        functionAdd = context.getFunction(moduleAdd, "scatter_add_f32");
        initializedAdd = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.ScatterOp scatterOp)) {
            throw new IllegalArgumentException(
                "Expected ScatterOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 3) {
            throw new IllegalArgumentException(
                "Scatter requires 3 inputs (operand, indices, updates), got: " + inputs.size());
        }

        Tensor operand = inputs.get(0);
        Tensor indices = inputs.get(1);
        Tensor updates = inputs.get(2);

        // Simplified scatter: 1D scatter-add
        if (operand.shape().length != 1 || indices.shape().length != 1) {
            throw new UnsupportedOperationException(
                "Only 1D scatter-add is currently supported");
        }

        return executeScatterAdd(operand, indices, updates, scatterOp);
    }

    private List<Tensor> executeScatterAdd(Tensor operand, Tensor indices, Tensor updates,
                                            StableHloAst.ScatterOp scatterOp) {
        ensureAddInitialized();

        int operandSize = operand.shape()[0];
        int nUpdates = (int) indices.elementCount();
        TensorSpec outputSpec = TensorSpec.fromAst(scatterOp.tensorResultType());

        long operandByteSize = operandSize * 4L;
        long indicesByteSize = nUpdates * 4L;
        long updatesByteSize = nUpdates * 4L;

        long dOutput = context.allocate(operandByteSize);
        long dIndices = context.allocate(indicesByteSize);
        long dUpdates = context.allocate(updatesByteSize);
        long dTiming = 0;

        try {
            // Copy operand to output (scatter modifies in place)
            context.copyToDevice(dOutput, operand.data());
            context.copyToDevice(dIndices, indices.data());
            context.copyToDevice(dUpdates, updates.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(nUpdates, blockSize);

            // PTX parameter order: (output_ptr, indices_ptr, updates_ptr, nUpdates, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    functionAdd,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dOutput, dIndices, dUpdates},
                    new int[]{nUpdates},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after int params
                );
            } else {
                context.launchKernelWithMixedParams(
                    functionAdd,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dOutput, dIndices, dUpdates},
                    new int[]{nUpdates},
                    new float[]{}
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, operandByteSize);

            return List.of(output);

        } finally {
            context.free(dOutput);
            context.free(dIndices);
            context.free(dUpdates);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.ScatterOp;
    }

    public int getSalt() {
        return salt;
    }
}
