package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for reduce_window operation (pooling).
 *
 * <p>Supports:
 * - 2D max pooling
 * - 2D average pooling
 *
 * <p>Window dimensions and strides are extracted from the operation.
 *
 * @see CudaKernels#generateMaxPool2DF32
 * @see CudaKernels#generateAvgPool2DF32
 */
public final class ReduceWindowKernel implements CudaOpKernel {

    private static final int BLOCK_SIZE = 16;

    private final CudaContext context;
    private final int salt;

    private long moduleMax;
    private long functionMax;
    private boolean initializedMax;

    private long moduleAvg;
    private long functionAvg;
    private boolean initializedAvg;

    public ReduceWindowKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensureMaxInitialized() {
        if (initializedMax) return;
        String ptx = CudaKernels.generateMaxPool2DF32(salt);
        moduleMax = context.loadModule("maxpool_2d_f32_salt" + salt, ptx);
        functionMax = context.getFunction(moduleMax, "maxpool_2d_f32");
        initializedMax = true;
    }

    private synchronized void ensureAvgInitialized() {
        if (initializedAvg) return;
        String ptx = CudaKernels.generateAvgPool2DF32(salt);
        moduleAvg = context.loadModule("avgpool_2d_f32_salt" + salt, ptx);
        functionAvg = context.getFunction(moduleAvg, "avgpool_2d_f32");
        initializedAvg = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.ReduceWindowOp reduceWindowOp)) {
            throw new IllegalArgumentException(
                "Expected ReduceWindowOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() < 1) {
            throw new IllegalArgumentException(
                "ReduceWindow requires at least 1 input, got: " + inputs.size());
        }

        Tensor input = inputs.get(0);
        int[] shape = input.shape();

        // Currently only support 2D pooling
        if (shape.length != 2) {
            throw new UnsupportedOperationException(
                "Only 2D pooling is currently supported, got " + shape.length + "D input");
        }

        // Determine reducer type
        String reducer = reduceWindowOp.reducer();
        boolean isMax = reducer.contains("max") || reducer.contains("maximum");
        boolean isAdd = reducer.contains("add") || reducer.contains("sum");

        if (isMax) {
            return executeMaxPool2D(input, reduceWindowOp);
        } else if (isAdd) {
            return executeAvgPool2D(input, reduceWindowOp);
        } else {
            throw new UnsupportedOperationException(
                "Unsupported reducer: " + reducer + ". Only max and add/avg are supported.");
        }
    }

    private List<Tensor> executeMaxPool2D(Tensor input, StableHloAst.ReduceWindowOp reduceWindowOp) {
        ensureMaxInitialized();

        int[] shape = input.shape();
        int inHeight = shape[0];
        int inWidth = shape[1];

        List<Long> windowDims = reduceWindowOp.windowDimensions();
        List<Long> strides = reduceWindowOp.windowStrides();

        int windowH = windowDims.size() > 0 ? windowDims.get(0).intValue() : 1;
        int windowW = windowDims.size() > 1 ? windowDims.get(1).intValue() : 1;
        int strideH = strides.size() > 0 ? strides.get(0).intValue() : 1;
        int strideW = strides.size() > 1 ? strides.get(1).intValue() : 1;

        TensorSpec outputSpec = TensorSpec.fromAst(reduceWindowOp.tensorResultType());
        int outHeight = outputSpec.shape()[0];
        int outWidth = outputSpec.shape()[1];

        long inByteSize = (long) inHeight * inWidth * 4L;
        long outByteSize = (long) outHeight * outWidth * 4L;

        long dIn = context.allocate(inByteSize);
        long dOut = context.allocate(outByteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dIn, input.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int gridX = (outWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int gridY = (outHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // PTX parameter order: (in_ptr, out_ptr, inHeight, inWidth, outHeight, outWidth, windowH, windowW, strideH, strideW, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    functionMax,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dIn, dOut},
                    new int[]{inHeight, inWidth, outHeight, outWidth, windowH, windowW, strideH, strideW},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after int params
                );
            } else {
                context.launchKernelWithMixedParams(
                    functionMax,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dIn, dOut},
                    new int[]{inHeight, inWidth, outHeight, outWidth, windowH, windowW, strideH, strideW},
                    new float[]{}
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, outByteSize);

            return List.of(output);

        } finally {
            context.free(dIn);
            context.free(dOut);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    private List<Tensor> executeAvgPool2D(Tensor input, StableHloAst.ReduceWindowOp reduceWindowOp) {
        ensureAvgInitialized();

        int[] shape = input.shape();
        int inHeight = shape[0];
        int inWidth = shape[1];

        List<Long> windowDims = reduceWindowOp.windowDimensions();
        List<Long> strides = reduceWindowOp.windowStrides();

        int windowH = windowDims.size() > 0 ? windowDims.get(0).intValue() : 1;
        int windowW = windowDims.size() > 1 ? windowDims.get(1).intValue() : 1;
        int strideH = strides.size() > 0 ? strides.get(0).intValue() : 1;
        int strideW = strides.size() > 1 ? strides.get(1).intValue() : 1;

        TensorSpec outputSpec = TensorSpec.fromAst(reduceWindowOp.tensorResultType());
        int outHeight = outputSpec.shape()[0];
        int outWidth = outputSpec.shape()[1];

        long inByteSize = (long) inHeight * inWidth * 4L;
        long outByteSize = (long) outHeight * outWidth * 4L;

        long dIn = context.allocate(inByteSize);
        long dOut = context.allocate(outByteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dIn, input.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int gridX = (outWidth + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int gridY = (outHeight + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // PTX parameter order: (in_ptr, out_ptr, inHeight, inWidth, outHeight, outWidth, windowH, windowW, strideH, strideW, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    functionAvg,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dIn, dOut},
                    new int[]{inHeight, inWidth, outHeight, outWidth, windowH, windowW, strideH, strideW},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after int params
                );
            } else {
                context.launchKernelWithMixedParams(
                    functionAvg,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dIn, dOut},
                    new int[]{inHeight, inWidth, outHeight, outWidth, windowH, windowW, strideH, strideW},
                    new float[]{}
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, outByteSize);

            return List.of(output);

        } finally {
            context.free(dIn);
            context.free(dOut);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.ReduceWindowOp;
    }

    public int getSalt() {
        return salt;
    }
}
