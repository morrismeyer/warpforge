package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for broadcast_in_dim operation.
 *
 * <p>Broadcasts a tensor to a larger shape by replicating elements
 * along specified dimensions. Supports:
 * <ul>
 *   <li>Scalar to any shape</li>
 *   <li>1D to 2D broadcasts (row or column)</li>
 * </ul>
 *
 * @see CudaKernels#generateBroadcastScalarF32
 * @see CudaKernels#generateBroadcast1Dto2DRowF32
 * @see CudaKernels#generateBroadcast1Dto2DColF32
 */
public final class BroadcastInDimKernel implements CudaOpKernel {

    private static final int BLOCK_SIZE = 16;

    private final CudaContext context;
    private final int salt;

    // Lazy-initialized modules for different broadcast patterns
    private long moduleScalar;
    private long functionScalar;
    private boolean initializedScalar;

    private long moduleRow;
    private long functionRow;
    private boolean initializedRow;

    private long moduleCol;
    private long functionCol;
    private boolean initializedCol;

    public BroadcastInDimKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensureScalarInitialized() {
        if (initializedScalar) return;
        String ptx = CudaKernels.generateBroadcastScalarF32(salt);
        moduleScalar = context.loadModule("broadcast_scalar_f32_salt" + salt, ptx);
        functionScalar = context.getFunction(moduleScalar, "broadcast_scalar_f32");
        initializedScalar = true;
    }

    private synchronized void ensureRowInitialized() {
        if (initializedRow) return;
        String ptx = CudaKernels.generateBroadcast1Dto2DRowF32(salt);
        moduleRow = context.loadModule("broadcast_1d_to_2d_row_f32_salt" + salt, ptx);
        functionRow = context.getFunction(moduleRow, "broadcast_1d_to_2d_row_f32");
        initializedRow = true;
    }

    private synchronized void ensureColInitialized() {
        if (initializedCol) return;
        String ptx = CudaKernels.generateBroadcast1Dto2DColF32(salt);
        moduleCol = context.loadModule("broadcast_1d_to_2d_col_f32_salt" + salt, ptx);
        functionCol = context.getFunction(moduleCol, "broadcast_1d_to_2d_col_f32");
        initializedCol = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.BroadcastInDimOp broadcastOp)) {
            throw new IllegalArgumentException(
                "Expected BroadcastInDimOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 1) {
            throw new IllegalArgumentException(
                "BroadcastInDim requires exactly 1 input, got: " + inputs.size());
        }

        Tensor input = inputs.get(0);
        int[] inputShape = input.shape();
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());
        int[] outputShape = outputSpec.shape();
        List<Long> broadcastDims = broadcastOp.broadcastDimensions();

        // Determine broadcast pattern and dispatch
        if (inputShape.length == 0 || (inputShape.length == 1 && inputShape[0] == 1)) {
            // Scalar broadcast
            return executeScalarBroadcast(input, outputSpec);
        } else if (inputShape.length == 1 && outputShape.length == 2) {
            // 1D to 2D broadcast
            return execute1Dto2DBroadcast(input, outputSpec, broadcastDims);
        } else {
            throw new UnsupportedOperationException(
                "Unsupported broadcast pattern: " + java.util.Arrays.toString(inputShape) +
                " -> " + java.util.Arrays.toString(outputShape) +
                " with dimensions " + broadcastDims);
        }
    }

    private List<Tensor> executeScalarBroadcast(Tensor input, TensorSpec outputSpec) {
        ensureScalarInitialized();

        int n = (int) outputSpec.elementCount();
        long inputByteSize = 4L;
        long outputByteSize = n * 4L;

        long dInput = context.allocate(inputByteSize);
        long dOutput = context.allocate(outputByteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dInput, input.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    functionScalar,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput, dTiming},
                    n
                );
            } else {
                context.launchKernelWithIntParams(
                    functionScalar,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput},
                    n
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

    private List<Tensor> execute1Dto2DBroadcast(Tensor input, TensorSpec outputSpec,
                                                  List<Long> broadcastDims) {
        int[] outputShape = outputSpec.shape();
        int rows = outputShape[0];
        int cols = outputShape[1];

        // broadcastDims tells us which output dimension corresponds to the input
        // [0] means input maps to rows (broadcast along columns)
        // [1] means input maps to cols (broadcast along rows)
        if (broadcastDims.size() != 1) {
            throw new UnsupportedOperationException(
                "Expected single broadcast dimension for 1D->2D, got: " + broadcastDims);
        }

        long broadcastDim = broadcastDims.get(0);
        boolean broadcastAlongRows = (broadcastDim == 1);  // input[j] -> output[i,j]
        boolean broadcastAlongCols = (broadcastDim == 0);  // input[i] -> output[i,j]

        if (broadcastAlongRows) {
            ensureRowInitialized();
        } else {
            ensureColInitialized();
        }

        long function = broadcastAlongRows ? functionRow : functionCol;

        int n = rows * cols;
        long inputByteSize = input.elementCount() * 4L;
        long outputByteSize = n * 4L;

        long dInput = context.allocate(inputByteSize);
        long dOutput = context.allocate(outputByteSize);
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
        return op instanceof StableHloAst.BroadcastInDimOp;
    }

    public int getSalt() {
        return salt;
    }
}
