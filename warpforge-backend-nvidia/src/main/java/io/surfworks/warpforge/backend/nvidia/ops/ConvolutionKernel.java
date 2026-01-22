package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for convolution operation.
 *
 * <p>Currently supports:
 * - 2D convolution (single channel)
 *
 * <p>More complex convolutions (batch, multi-channel) can be built on top.
 *
 * @see CudaKernels#generateConv2DF32
 */
public final class ConvolutionKernel implements CudaOpKernel {

    private static final int BLOCK_SIZE = 16;

    private final CudaContext context;
    private final int salt;

    private long module2D;
    private long function2D;
    private boolean initialized2D;

    public ConvolutionKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensure2DInitialized() {
        if (initialized2D) return;
        String ptx = CudaKernels.generateConv2DF32(salt);
        module2D = context.loadModule("conv2d_f32_salt" + salt, ptx);
        function2D = context.getFunction(module2D, "conv2d_f32");
        initialized2D = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.ConvolutionOp convOp)) {
            throw new IllegalArgumentException(
                "Expected ConvolutionOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                "Convolution requires 2 inputs (lhs/input, rhs/kernel), got: " + inputs.size());
        }

        Tensor input = inputs.get(0);
        Tensor kernel = inputs.get(1);

        int inputDim = input.shape().length;
        int kernelDim = kernel.shape().length;

        // Handle 2D convolution
        if (inputDim == 2 && kernelDim == 2) {
            return executeConv2D(input, kernel, convOp);
        }

        throw new UnsupportedOperationException(
            "Only 2D convolution is currently supported. Got input dim=" + inputDim + ", kernel dim=" + kernelDim);
    }

    private List<Tensor> executeConv2D(Tensor input, Tensor kernel, StableHloAst.ConvolutionOp convOp) {
        ensure2DInitialized();

        int[] inputShape = input.shape();  // [H, W]
        int[] kernelShape = kernel.shape();  // [kH, kW]

        int inHeight = inputShape[0];
        int inWidth = inputShape[1];
        int kernelH = kernelShape[0];
        int kernelW = kernelShape[1];

        // Extract strides
        List<Long> strides = convOp.windowStrides();
        int strideH = strides.size() > 0 ? strides.get(0).intValue() : 1;
        int strideW = strides.size() > 1 ? strides.get(1).intValue() : 1;

        // Extract padding
        List<Long> paddingLow = convOp.paddingLow();
        int padH = paddingLow.size() > 0 ? paddingLow.get(0).intValue() : 0;
        int padW = paddingLow.size() > 1 ? paddingLow.get(1).intValue() : 0;

        TensorSpec outputSpec = TensorSpec.fromAst(convOp.tensorResultType());
        int outHeight = outputSpec.shape()[0];
        int outWidth = outputSpec.shape()[1];

        long inByteSize = (long) inHeight * inWidth * 4L;
        long kernelByteSize = (long) kernelH * kernelW * 4L;
        long outByteSize = (long) outHeight * outWidth * 4L;

        long dIn = context.allocate(inByteSize);
        long dKernel = context.allocate(kernelByteSize);
        long dOut = context.allocate(outByteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dIn, input.data());
            context.copyToDevice(dKernel, kernel.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int gridX = (outWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int gridY = (outHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function2D,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dIn, dKernel, dOut, dTiming},
                    inHeight, inWidth, kernelH, kernelW, outHeight, outWidth, strideH, strideW, padH, padW
                );
            } else {
                context.launchKernelWithIntParams(
                    function2D,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dIn, dKernel, dOut},
                    inHeight, inWidth, kernelH, kernelW, outHeight, outWidth, strideH, strideW, padH, padW
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, outByteSize);

            return List.of(output);

        } finally {
            context.free(dIn);
            context.free(dKernel);
            context.free(dOut);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.ConvolutionOp;
    }

    public int getSalt() {
        return salt;
    }
}
