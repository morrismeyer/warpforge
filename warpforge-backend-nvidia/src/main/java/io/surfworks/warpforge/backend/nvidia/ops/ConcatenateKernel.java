package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for concatenate operation.
 *
 * <p>Concatenates multiple tensors along a specified dimension.
 * Currently optimized for 1D concatenation of two tensors.
 *
 * @see CudaKernels#generateConcatenate2F32
 */
public final class ConcatenateKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    public ConcatenateKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        String ptxSource = CudaKernels.generateConcatenate2F32(salt);
        String moduleName = "concatenate_2_f32_module_salt" + salt;
        String functionName = "concatenate_2_f32";

        module = context.loadModule(moduleName, ptxSource);
        function = context.getFunction(module, functionName);
        initialized = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.ConcatenateOp concatOp)) {
            throw new IllegalArgumentException(
                "Expected ConcatenateOp, got: " + op.getClass().getSimpleName());
        }

        // Currently only support 2 inputs
        if (inputs.size() != 2) {
            throw new UnsupportedOperationException(
                "Only concatenation of 2 tensors is currently supported, got: " + inputs.size());
        }

        Tensor inputA = inputs.get(0);
        Tensor inputB = inputs.get(1);
        long dimension = concatOp.dimension();

        // Currently only support 1D concatenation (dimension 0)
        if (inputA.shape().length != 1 || inputB.shape().length != 1) {
            throw new UnsupportedOperationException(
                "Only 1D tensor concatenation is currently supported");
        }

        if (dimension != 0) {
            throw new UnsupportedOperationException(
                "Only concatenation along dimension 0 is supported for 1D tensors");
        }

        ensureInitialized();

        int nA = inputA.shape()[0];
        int nB = inputB.shape()[0];
        int nTotal = nA + nB;

        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long byteSizeA = nA * 4L;
        long byteSizeB = nB * 4L;
        long byteSizeOut = nTotal * 4L;

        long dA = context.allocate(byteSizeA);
        long dB = context.allocate(byteSizeB);
        long dOut = context.allocate(byteSizeOut);
        long dTiming = 0;

        try {
            context.copyToDevice(dA, inputA.data());
            context.copyToDevice(dB, inputB.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(nTotal, blockSize);

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dA, dB, dOut, dTiming},
                    nA, nTotal
                );
            } else {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dA, dB, dOut},
                    nA, nTotal
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSizeOut);

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
        return op instanceof StableHloAst.ConcatenateOp;
    }

    public int getSalt() {
        return salt;
    }
}
