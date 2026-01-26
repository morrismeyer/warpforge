package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for generalized dot product (batched matrix multiplication).
 *
 * <p>Supports batched matrix multiplication: C[b,M,N] = A[b,M,K] * B[b,K,N]
 *
 * <p>Currently supports:
 * - 3D batched matmul with batch dimension 0
 * - 2D regular matmul (delegated to standard dot)
 *
 * @see CudaKernels#generateBatchMatMulF32
 */
public final class DotGeneralKernel implements CudaOpKernel {

    private static final int BLOCK_SIZE = 16;

    private final CudaContext context;
    private final int salt;

    private long moduleBatch;
    private long functionBatch;
    private boolean initializedBatch;

    public DotGeneralKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensureBatchInitialized() {
        if (initializedBatch) return;
        String ptx = CudaKernels.generateBatchMatMulF32(salt);
        moduleBatch = context.loadModule("batch_matmul_f32_salt" + salt, ptx);
        functionBatch = context.getFunction(moduleBatch, "batch_matmul_f32");
        initializedBatch = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.DotGeneralOp dotGeneralOp)) {
            throw new IllegalArgumentException(
                "Expected DotGeneralOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                "DotGeneral requires 2 inputs (lhs, rhs), got: " + inputs.size());
        }

        Tensor lhs = inputs.get(0);
        Tensor rhs = inputs.get(1);

        int lhsDim = lhs.shape().length;
        int rhsDim = rhs.shape().length;

        // Handle batched matmul: [B, M, K] x [B, K, N] -> [B, M, N]
        if (lhsDim == 3 && rhsDim == 3) {
            return executeBatchMatMul(lhs, rhs, dotGeneralOp);
        }

        // Handle 2D case: [M, K] x [K, N] -> [M, N]
        if (lhsDim == 2 && rhsDim == 2) {
            return execute2DMatMul(lhs, rhs, dotGeneralOp);
        }

        throw new UnsupportedOperationException(
            "Only 2D and 3D (batched) matmul are currently supported. Got lhs dim=" + lhsDim + ", rhs dim=" + rhsDim);
    }

    private List<Tensor> executeBatchMatMul(Tensor lhs, Tensor rhs, StableHloAst.DotGeneralOp dotGeneralOp) {
        ensureBatchInitialized();

        int[] lhsShape = lhs.shape();  // [B, M, K]
        int[] rhsShape = rhs.shape();  // [B, K, N]

        int batchSize = lhsShape[0];
        int M = lhsShape[1];
        int K = lhsShape[2];
        int K2 = rhsShape[1];
        int N = rhsShape[2];

        if (lhsShape[0] != rhsShape[0]) {
            throw new IllegalArgumentException(
                "Batch dimensions must match: " + lhsShape[0] + " vs " + rhsShape[0]);
        }

        if (K != K2) {
            throw new IllegalArgumentException(
                "Inner dimensions must match: K=" + K + " vs K2=" + K2);
        }

        TensorSpec outputSpec = TensorSpec.fromAst(dotGeneralOp.tensorResultType());

        long byteSizeA = (long) batchSize * M * K * 4L;
        long byteSizeB = (long) batchSize * K * N * 4L;
        long byteSizeC = (long) batchSize * M * N * 4L;

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

            // 3D grid: (N, M, batch)
            int gridX = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int gridY = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int gridZ = batchSize;

            // PTX parameter order: (a_ptr, b_ptr, c_ptr, batchSize, M, N, K, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    functionBatch,
                    new int[]{gridX, gridY, gridZ}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dA, dB, dC},
                    new int[]{batchSize, M, N, K},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after int params
                );
            } else {
                context.launchKernelWithMixedParams(
                    functionBatch,
                    new int[]{gridX, gridY, gridZ}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dA, dB, dC},
                    new int[]{batchSize, M, N, K},
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

    private List<Tensor> execute2DMatMul(Tensor lhs, Tensor rhs, StableHloAst.DotGeneralOp dotGeneralOp) {
        // For 2D case, use batch size of 1
        ensureBatchInitialized();

        int[] lhsShape = lhs.shape();  // [M, K]
        int[] rhsShape = rhs.shape();  // [K, N]

        int M = lhsShape[0];
        int K = lhsShape[1];
        int K2 = rhsShape[0];
        int N = rhsShape[1];

        if (K != K2) {
            throw new IllegalArgumentException(
                "Inner dimensions must match: K=" + K + " vs K2=" + K2);
        }

        TensorSpec outputSpec = TensorSpec.fromAst(dotGeneralOp.tensorResultType());

        long byteSizeA = (long) M * K * 4L;
        long byteSizeB = (long) K * N * 4L;
        long byteSizeC = (long) M * N * 4L;

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

            // 3D grid with batch=1: (N, M, 1)
            int gridX = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int gridY = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // PTX parameter order: (a_ptr, b_ptr, c_ptr, batchSize=1, M, N, K, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    functionBatch,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dA, dB, dC},
                    new int[]{1, M, N, K},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after int params
                );
            } else {
                context.launchKernelWithMixedParams(
                    functionBatch,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE, BLOCK_SIZE, 1},
                    0,
                    new long[]{dA, dB, dC},
                    new int[]{1, M, N, K},
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
        return op instanceof StableHloAst.DotGeneralOp;
    }

    public int getSalt() {
        return salt;
    }
}
