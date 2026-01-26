package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for reverse operation.
 *
 * <p>Reverses elements along specified dimensions.
 * For 1D tensors: output[i] = input[n - 1 - i]
 * For 2D tensors: reverses along dimension 0, 1, or both
 *
 * @see CudaKernels#generateReverse1DF32
 * @see CudaKernels#generateReverse2DDim0F32
 * @see CudaKernels#generateReverse2DDim1F32
 */
public final class ReverseKernel implements CudaOpKernel {

    private static final int BLOCK_SIZE_2D = 16;

    private final CudaContext context;
    private final int salt;

    private long module1D;
    private long function1D;
    private boolean initialized1D;

    private long module2DDim0;
    private long function2DDim0;
    private boolean initialized2DDim0;

    private long module2DDim1;
    private long function2DDim1;
    private boolean initialized2DDim1;

    public ReverseKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensure1DInitialized() {
        if (initialized1D) return;
        String ptx = CudaKernels.generateReverse1DF32(salt);
        module1D = context.loadModule("reverse_1d_f32_salt" + salt, ptx);
        function1D = context.getFunction(module1D, "reverse_1d_f32");
        initialized1D = true;
    }

    private synchronized void ensure2DDim0Initialized() {
        if (initialized2DDim0) return;
        String ptx = CudaKernels.generateReverse2DDim0F32(salt);
        module2DDim0 = context.loadModule("reverse_2d_dim0_f32_salt" + salt, ptx);
        function2DDim0 = context.getFunction(module2DDim0, "reverse_2d_dim0_f32");
        initialized2DDim0 = true;
    }

    private synchronized void ensure2DDim1Initialized() {
        if (initialized2DDim1) return;
        String ptx = CudaKernels.generateReverse2DDim1F32(salt);
        module2DDim1 = context.loadModule("reverse_2d_dim1_f32_salt" + salt, ptx);
        function2DDim1 = context.getFunction(module2DDim1, "reverse_2d_dim1_f32");
        initialized2DDim1 = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.ReverseOp reverseOp)) {
            throw new IllegalArgumentException(
                "Expected ReverseOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 1) {
            throw new IllegalArgumentException(
                "Reverse requires exactly 1 input, got: " + inputs.size());
        }

        Tensor input = inputs.get(0);
        int[] shape = input.shape();
        int ndim = shape.length;
        List<Long> dimensions = reverseOp.dimensions();

        if (ndim == 1) {
            if (dimensions.isEmpty() || !dimensions.contains(0L)) {
                // No dimensions to reverse, just copy
                return List.of(input.copy());
            }
            return execute1DReverse(input, reverseOp);
        } else if (ndim == 2) {
            return execute2DReverse(input, reverseOp, dimensions);
        } else {
            throw new UnsupportedOperationException(
                "Only 1D and 2D reverse is currently supported, got " + ndim + "D tensor");
        }
    }

    private List<Tensor> execute1DReverse(Tensor input, StableHloAst.ReverseOp reverseOp) {
        ensure1DInitialized();

        int n = input.shape()[0];
        TensorSpec outputSpec = TensorSpec.fromAst(reverseOp.tensorResultType());

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
                    function1D,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput},
                    new int[]{n},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after n
                );
            } else {
                context.launchKernelWithMixedParams(
                    function1D,
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

    private List<Tensor> execute2DReverse(Tensor input, StableHloAst.ReverseOp reverseOp,
                                           List<Long> dimensions) {
        boolean reverseDim0 = dimensions.contains(0L);
        boolean reverseDim1 = dimensions.contains(1L);

        if (!reverseDim0 && !reverseDim1) {
            // No dimensions to reverse, just copy
            return List.of(input.copy());
        }

        // For both dimensions, we need to do two passes
        // For single dimension, one pass is enough
        if (reverseDim0 && reverseDim1) {
            // Reverse dim0, then dim1
            List<Tensor> intermediate = execute2DReverseSingleDim(input, reverseOp, 0);
            return execute2DReverseSingleDim(intermediate.get(0), reverseOp, 1);
        } else if (reverseDim0) {
            return execute2DReverseSingleDim(input, reverseOp, 0);
        } else {
            return execute2DReverseSingleDim(input, reverseOp, 1);
        }
    }

    private List<Tensor> execute2DReverseSingleDim(Tensor input, StableHloAst.ReverseOp reverseOp,
                                                    int dim) {
        int rows = input.shape()[0];
        int cols = input.shape()[1];
        int nTotal = rows * cols;
        TensorSpec outputSpec = TensorSpec.fromAst(reverseOp.tensorResultType());

        long byteSize = nTotal * 4L;

        long dInput = context.allocate(byteSize);
        long dOutput = context.allocate(byteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dInput, input.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int gridX = (cols + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (rows + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            long function;
            if (dim == 0) {
                ensure2DDim0Initialized();
                function = function2DDim0;
            } else {
                ensure2DDim1Initialized();
                function = function2DDim1;
            }

            // PTX parameter order: (in_ptr, out_ptr, rows, cols, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                    0,
                    new long[]{dInput, dOutput},
                    new int[]{rows, cols},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after int params
                );
            } else {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                    0,
                    new long[]{dInput, dOutput},
                    new int[]{rows, cols},
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
        return op instanceof StableHloAst.ReverseOp;
    }

    public int getSalt() {
        return salt;
    }
}
