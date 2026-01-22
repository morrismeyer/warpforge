package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for gather operation.
 *
 * <p>Gathers elements from a tensor using indices. Currently supports:
 * - 1D gather: output[i] = input[indices[i]]
 * - 2D embedding lookup: output[i,:] = input[indices[i],:]
 *
 * @see CudaKernels#generateGather1DF32
 * @see CudaKernels#generateGather2DF32
 */
public final class GatherKernel implements CudaOpKernel {

    private static final int BLOCK_SIZE_2D = 16;

    private final CudaContext context;
    private final int salt;

    private long module1D;
    private long function1D;
    private boolean initialized1D;

    private long module2D;
    private long function2D;
    private boolean initialized2D;

    public GatherKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensure1DInitialized() {
        if (initialized1D) return;
        String ptx = CudaKernels.generateGather1DF32(salt);
        module1D = context.loadModule("gather_1d_f32_salt" + salt, ptx);
        function1D = context.getFunction(module1D, "gather_1d_f32");
        initialized1D = true;
    }

    private synchronized void ensure2DInitialized() {
        if (initialized2D) return;
        String ptx = CudaKernels.generateGather2DF32(salt);
        module2D = context.loadModule("gather_2d_f32_salt" + salt, ptx);
        function2D = context.getFunction(module2D, "gather_2d_f32");
        initialized2D = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.GatherOp gatherOp)) {
            throw new IllegalArgumentException(
                "Expected GatherOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                "Gather requires 2 inputs (operand and indices), got: " + inputs.size());
        }

        Tensor operand = inputs.get(0);
        Tensor startIndices = inputs.get(1);
        int operandDim = operand.shape().length;

        // Simplified gather: handle common embedding lookup case
        if (operandDim == 1) {
            return execute1DGather(operand, startIndices, gatherOp);
        } else if (operandDim == 2 && startIndices.shape().length == 1) {
            return execute2DGather(operand, startIndices, gatherOp);
        } else {
            throw new UnsupportedOperationException(
                "Only 1D gather and 2D embedding lookup are currently supported");
        }
    }

    private List<Tensor> execute1DGather(Tensor operand, Tensor indices,
                                          StableHloAst.GatherOp gatherOp) {
        ensure1DInitialized();

        int nIndices = (int) indices.elementCount();
        TensorSpec outputSpec = TensorSpec.fromAst(gatherOp.tensorResultType());

        long operandByteSize = operand.elementCount() * 4L;
        long indicesByteSize = nIndices * 4L;
        long outputByteSize = nIndices * 4L;

        long dOperand = context.allocate(operandByteSize);
        long dIndices = context.allocate(indicesByteSize);
        long dOutput = context.allocate(outputByteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dOperand, operand.data());
            context.copyToDevice(dIndices, indices.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(nIndices, blockSize);

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function1D,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dOperand, dIndices, dOutput, dTiming},
                    nIndices
                );
            } else {
                context.launchKernelWithIntParams(
                    function1D,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dOperand, dIndices, dOutput},
                    nIndices
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, outputByteSize);

            return List.of(output);

        } finally {
            context.free(dOperand);
            context.free(dIndices);
            context.free(dOutput);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    private List<Tensor> execute2DGather(Tensor operand, Tensor indices,
                                          StableHloAst.GatherOp gatherOp) {
        ensure2DInitialized();

        int nIndices = (int) indices.elementCount();
        int embeddingDim = operand.shape()[1];
        TensorSpec outputSpec = TensorSpec.fromAst(gatherOp.tensorResultType());

        long operandByteSize = operand.elementCount() * 4L;
        long indicesByteSize = nIndices * 4L;
        long outputByteSize = (long) nIndices * embeddingDim * 4L;

        long dOperand = context.allocate(operandByteSize);
        long dIndices = context.allocate(indicesByteSize);
        long dOutput = context.allocate(outputByteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dOperand, operand.data());
            context.copyToDevice(dIndices, indices.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int gridX = (embeddingDim + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (nIndices + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function2D,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                    0,
                    new long[]{dOperand, dIndices, dOutput, dTiming},
                    nIndices, embeddingDim
                );
            } else {
                context.launchKernelWithIntParams(
                    function2D,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                    0,
                    new long[]{dOperand, dIndices, dOutput},
                    nIndices, embeddingDim
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, outputByteSize);

            return List.of(output);

        } finally {
            context.free(dOperand);
            context.free(dIndices);
            context.free(dOutput);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.GatherOp;
    }

    public int getSalt() {
        return salt;
    }
}
