package io.surfworks.warpforge.backend.amd.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.amd.hip.HipContext;
import io.surfworks.warpforge.backend.amd.hip.HipKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * HIP kernel for matrix multiplication (dot product) with salt instrumentation.
 *
 * <p>This is the OPTIMIZED_OBSERVABLE / CORRECTNESS tier implementation.
 * It uses custom HIP C++ kernels that can be instrumented with salt for
 * internal kernel visibility.
 *
 * <p>Computes C[M,N] = A[M,K] * B[K,N]
 *
 * <p>For PRODUCTION tier (maximum performance), use {@link RocblasDotKernel}
 * which leverages AMD's highly optimized rocBLAS library.
 *
 * <h2>Implementation Status</h2>
 * <p>Requires HIPRTC FFM bindings to compile HIP C++ at runtime.
 *
 * @see HipKernels#generateDotF32(int)
 * @see RocblasDotKernel PRODUCTION tier implementation
 */
public final class DotKernel implements HipOpKernel {

    private final HipContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    public DotKernel(HipContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        String hipSource = HipKernels.generateDotF32(salt);

        throw new UnsupportedOperationException(
            "DotKernel requires HIPRTC integration to compile HIP C++ at runtime. " +
            "Generated source (" + hipSource.length() + " chars) is ready for compilation. " +
            "Use RocblasDotKernel for PRODUCTION tier, or CPU backend for correctness testing.");
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

            if (salt >= HipKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            // Launch with 2D grid
            int blockSize = HipKernels.DOT_BLOCK_SIZE;
            int[] gridDim = HipKernels.calculateGridSize2D(M, N);

            if (salt >= HipKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function,
                    gridDim, new int[]{blockSize, blockSize, 1},
                    0,
                    new long[]{dA, dB, dC, dTiming},
                    M, N, K
                );
            } else {
                context.launchKernelWithIntParams(
                    function,
                    gridDim, new int[]{blockSize, blockSize, 1},
                    0,
                    new long[]{dA, dB, dC},
                    M, N, K
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
