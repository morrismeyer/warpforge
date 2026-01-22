package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for iota operation.
 *
 * <p>Generates a tensor filled with values 0, 1, 2, ..., n-1 along the specified dimension.
 * For 1D tensors: output[i] = i
 * For 2D tensors with dim=0: output[i,j] = i
 * For 2D tensors with dim=1: output[i,j] = j
 *
 * @see CudaKernels#generateIota1DF32
 * @see CudaKernels#generateIota2DDim0F32
 * @see CudaKernels#generateIota2DDim1F32
 */
public final class IotaKernel implements CudaOpKernel {

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

    public IotaKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensure1DInitialized() {
        if (initialized1D) return;
        String ptx = CudaKernels.generateIota1DF32(salt);
        module1D = context.loadModule("iota_1d_f32_salt" + salt, ptx);
        function1D = context.getFunction(module1D, "iota_1d_f32");
        initialized1D = true;
    }

    private synchronized void ensure2DDim0Initialized() {
        if (initialized2DDim0) return;
        String ptx = CudaKernels.generateIota2DDim0F32(salt);
        module2DDim0 = context.loadModule("iota_2d_dim0_f32_salt" + salt, ptx);
        function2DDim0 = context.getFunction(module2DDim0, "iota_2d_dim0_f32");
        initialized2DDim0 = true;
    }

    private synchronized void ensure2DDim1Initialized() {
        if (initialized2DDim1) return;
        String ptx = CudaKernels.generateIota2DDim1F32(salt);
        module2DDim1 = context.loadModule("iota_2d_dim1_f32_salt" + salt, ptx);
        function2DDim1 = context.getFunction(module2DDim1, "iota_2d_dim1_f32");
        initialized2DDim1 = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.IotaOp iotaOp)) {
            throw new IllegalArgumentException(
                "Expected IotaOp, got: " + op.getClass().getSimpleName());
        }

        // Iota has no inputs, it just generates values based on the output shape
        TensorSpec outputSpec = TensorSpec.fromAst(iotaOp.tensorResultType());
        int[] shape = outputSpec.shape();
        int ndim = shape.length;
        long dimension = iotaOp.iotaDimension();

        if (ndim == 1) {
            return execute1DIota(outputSpec);
        } else if (ndim == 2) {
            return execute2DIota(outputSpec, (int) dimension);
        } else {
            throw new UnsupportedOperationException(
                "Only 1D and 2D iota is currently supported, got " + ndim + "D tensor");
        }
    }

    private List<Tensor> execute1DIota(TensorSpec outputSpec) {
        ensure1DInitialized();

        int n = outputSpec.shape()[0];
        long outputByteSize = n * 4L;

        long dOutput = context.allocate(outputByteSize);
        long dTiming = 0;

        try {
            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function1D,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dOutput, dTiming},
                    n
                );
            } else {
                context.launchKernelWithIntParams(
                    function1D,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dOutput},
                    n
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, outputByteSize);

            return List.of(output);

        } finally {
            context.free(dOutput);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    private List<Tensor> execute2DIota(TensorSpec outputSpec, int dimension) {
        int rows = outputSpec.shape()[0];
        int cols = outputSpec.shape()[1];
        int nTotal = rows * cols;
        long outputByteSize = nTotal * 4L;

        long dOutput = context.allocate(outputByteSize);
        long dTiming = 0;

        try {
            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int gridX = (cols + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (rows + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            long function;
            if (dimension == 0) {
                ensure2DDim0Initialized();
                function = function2DDim0;
            } else if (dimension == 1) {
                ensure2DDim1Initialized();
                function = function2DDim1;
            } else {
                throw new UnsupportedOperationException(
                    "Iota dimension must be 0 or 1 for 2D tensors, got: " + dimension);
            }

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                    0,
                    new long[]{dOutput, dTiming},
                    rows, cols
                );
            } else {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                    0,
                    new long[]{dOutput},
                    rows, cols
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, outputByteSize);

            return List.of(output);

        } finally {
            context.free(dOutput);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.IotaOp;
    }

    public int getSalt() {
        return salt;
    }
}
