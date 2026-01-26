package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for matrix multiplication (dot product).
 *
 * <p>Computes C[M,N] = A[M,K] * B[K,N]
 *
 * @see CudaKernels#generateDotF32
 */
public final class DotKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    public DotKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        String ptxSource = CudaKernels.generateDotF32(salt);
        String moduleName = "dot_f32_module_salt" + salt;
        String functionName = "dot_f32";

        module = context.loadModule(moduleName, ptxSource);
        function = context.getFunction(module, functionName);
        initialized = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.DotOp)) {
            throw new IllegalArgumentException(
                "Expected DotOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                "Dot requires exactly 2 inputs, got: " + inputs.size());
        }

        ensureInitialized();

        Tensor lhs = inputs.get(0);  // A[M, K]
        Tensor rhs = inputs.get(1);  // B[K, N]

        // Get dimensions
        int[] lhsShape = lhs.shape();
        int[] rhsShape = rhs.shape();

        if (lhsShape.length != 2 || rhsShape.length != 2) {
            throw new IllegalArgumentException(
                "Dot requires 2D tensors, got shapes: " +
                java.util.Arrays.toString(lhsShape) + " and " + java.util.Arrays.toString(rhsShape));
        }

        int M = lhsShape[0];
        int K = lhsShape[1];
        int K2 = rhsShape[0];
        int N = rhsShape[1];

        if (K != K2) {
            throw new IllegalArgumentException(
                "Inner dimensions must match: A[" + M + "," + K + "] * B[" + K2 + "," + N + "]");
        }

        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long byteSizeA = M * K * 4L;
        long byteSizeB = K * N * 4L;
        long byteSizeC = M * N * 4L;

        long dA = context.allocate(byteSizeA);
        long dB = context.allocate(byteSizeB);
        long dC = context.allocate(byteSizeC);
        long dTiming = 0;

        try {
            context.copyToDevice(dA, lhs.data());
            context.copyToDevice(dB, rhs.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            // Launch with 2D grid
            int blockSize = CudaKernels.DOT_BLOCK_SIZE;
            int gridX = (N + blockSize - 1) / blockSize;
            int gridY = (M + blockSize - 1) / blockSize;

            // PTX parameter order: (a_ptr, b_ptr, c_ptr, M, N, K, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridX, gridY, 1}, new int[]{blockSize, blockSize, 1},
                    0,
                    new long[]{dA, dB, dC},
                    new int[]{M, N, K},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after int params
                );
            } else {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridX, gridY, 1}, new int[]{blockSize, blockSize, 1},
                    0,
                    new long[]{dA, dB, dC},
                    new int[]{M, N, K},
                    new float[]{}
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dC, byteSizeC);

            return List.of(output);

        } finally {
            context.free(dA);
            context.free(dB);
            context.free(dC);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.DotOp;
    }

    public int getSalt() {
        return salt;
    }
}
