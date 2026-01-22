package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for slice operation.
 *
 * <p>Extracts a slice from a tensor with start, limit, and stride for each dimension.
 * Supports 1D and 2D slicing.
 *
 * @see CudaKernels#generateSlice1DF32
 * @see CudaKernels#generateSlice2DF32
 */
public final class SliceKernel implements CudaOpKernel {

    private static final int BLOCK_SIZE_2D = 16;

    private final CudaContext context;
    private final int salt;

    private long module1D;
    private long function1D;
    private boolean initialized1D;

    private long module2D;
    private long function2D;
    private boolean initialized2D;

    public SliceKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensure1DInitialized() {
        if (initialized1D) return;
        String ptx = CudaKernels.generateSlice1DF32(salt);
        module1D = context.loadModule("slice_1d_f32_salt" + salt, ptx);
        function1D = context.getFunction(module1D, "slice_1d_f32");
        initialized1D = true;
    }

    private synchronized void ensure2DInitialized() {
        if (initialized2D) return;
        String ptx = CudaKernels.generateSlice2DF32(salt);
        module2D = context.loadModule("slice_2d_f32_salt" + salt, ptx);
        function2D = context.getFunction(module2D, "slice_2d_f32");
        initialized2D = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.SliceOp sliceOp)) {
            throw new IllegalArgumentException(
                "Expected SliceOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 1) {
            throw new IllegalArgumentException(
                "Slice requires exactly 1 input, got: " + inputs.size());
        }

        Tensor input = inputs.get(0);
        int[] inputShape = input.shape();
        int ndim = inputShape.length;

        List<Long> startIndices = sliceOp.startIndices();
        List<Long> limitIndices = sliceOp.limitIndices();
        List<Long> strides = sliceOp.strides();

        if (ndim == 1) {
            return execute1DSlice(input, sliceOp, startIndices, limitIndices, strides);
        } else if (ndim == 2) {
            return execute2DSlice(input, sliceOp, startIndices, limitIndices, strides);
        } else {
            throw new UnsupportedOperationException(
                "Only 1D and 2D slicing is currently supported, got " + ndim + "D tensor");
        }
    }

    private List<Tensor> execute1DSlice(Tensor input, StableHloAst.SliceOp sliceOp,
                                         List<Long> startIndices, List<Long> limitIndices,
                                         List<Long> strides) {
        ensure1DInitialized();

        int start = startIndices.get(0).intValue();
        int limit = limitIndices.get(0).intValue();
        int stride = strides.get(0).intValue();

        int nOut = (limit - start + stride - 1) / stride;
        TensorSpec outputSpec = TensorSpec.fromAst(sliceOp.tensorResultType());

        int nIn = input.shape()[0];
        long inputByteSize = nIn * 4L;
        long outputByteSize = nOut * 4L;

        long dInput = context.allocate(inputByteSize);
        long dOutput = context.allocate(outputByteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dInput, input.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(nOut, blockSize);

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function1D,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput, dTiming},
                    start, stride, nOut
                );
            } else {
                context.launchKernelWithIntParams(
                    function1D,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput},
                    start, stride, nOut
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, outputByteSize);

            return List.of(output);

        } finally {
            context.free(dInput);
            context.free(dOutput);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    private List<Tensor> execute2DSlice(Tensor input, StableHloAst.SliceOp sliceOp,
                                         List<Long> startIndices, List<Long> limitIndices,
                                         List<Long> strides) {
        ensure2DInitialized();

        int start0 = startIndices.get(0).intValue();
        int start1 = startIndices.get(1).intValue();
        int limit0 = limitIndices.get(0).intValue();
        int limit1 = limitIndices.get(1).intValue();
        int stride0 = strides.get(0).intValue();
        int stride1 = strides.get(1).intValue();

        int outRows = (limit0 - start0 + stride0 - 1) / stride0;
        int outCols = (limit1 - start1 + stride1 - 1) / stride1;
        int inCols = input.shape()[1];

        TensorSpec outputSpec = TensorSpec.fromAst(sliceOp.tensorResultType());

        int nIn = (int) input.elementCount();
        int nOut = outRows * outCols;
        long inputByteSize = nIn * 4L;
        long outputByteSize = nOut * 4L;

        long dInput = context.allocate(inputByteSize);
        long dOutput = context.allocate(outputByteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dInput, input.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int gridX = (outCols + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (outRows + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function2D,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                    0,
                    new long[]{dInput, dOutput, dTiming},
                    inCols, outRows, outCols, start0, start1, stride0, stride1
                );
            } else {
                context.launchKernelWithIntParams(
                    function2D,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                    0,
                    new long[]{dInput, dOutput},
                    inCols, outRows, outCols, start0, start1, stride0, stride1
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, outputByteSize);

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
        return op instanceof StableHloAst.SliceOp;
    }

    public int getSalt() {
        return salt;
    }
}
