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
        // Read dimension numbers to handle different contracting dimensions
        StableHloAst.DotDimensionNumbers dimNums = dotGeneralOp.dimensionNumbers();
        java.util.List<Long> lhsContractDims = dimNums.lhsContractingDimensions();
        java.util.List<Long> rhsContractDims = dimNums.rhsContractingDimensions();

        if (lhsContractDims.size() != 1 || rhsContractDims.size() != 1) {
            throw new UnsupportedOperationException(
                "Only single contracting dimension supported for 2D matmul");
        }

        int lhsContractDim = lhsContractDims.get(0).intValue();
        int rhsContractDim = rhsContractDims.get(0).intValue();

        int[] lhsShape = lhs.shape();
        int[] rhsShape = rhs.shape();

        // Determine M, N, K based on contracting dimensions
        // Standard case: lhs[M,K] @ rhs[K,N] with lhsContract=1, rhsContract=0
        // Transposed B case: lhs[M,K] @ rhs[N,K] with lhsContract=1, rhsContract=1
        // Transposed A case: lhs[K,M] @ rhs[K,N] with lhsContract=0, rhsContract=0

        boolean transposeA = (lhsContractDim == 0);  // A contracts on first dim
        boolean transposeB = (rhsContractDim == 1);  // B contracts on second dim

        int M, K, N;
        if (transposeA) {
            // lhs is [K, M], contracting on dim 0
            K = lhsShape[0];
            M = lhsShape[1];
        } else {
            // lhs is [M, K], contracting on dim 1 (standard)
            M = lhsShape[0];
            K = lhsShape[1];
        }

        int K2;
        if (transposeB) {
            // rhs is [N, K], contracting on dim 1
            N = rhsShape[0];
            K2 = rhsShape[1];
        } else {
            // rhs is [K, N], contracting on dim 0 (standard)
            K2 = rhsShape[0];
            N = rhsShape[1];
        }

        if (K != K2) {
            throw new IllegalArgumentException(
                "Contracting dimensions must match: K=" + K + " vs K2=" + K2 +
                " (transposeA=" + transposeA + ", transposeB=" + transposeB + ")");
        }

        TensorSpec outputSpec = TensorSpec.fromAst(dotGeneralOp.tensorResultType());

        long byteSizeA = (long) lhsShape[0] * lhsShape[1] * 4L;
        long byteSizeB = (long) rhsShape[0] * rhsShape[1] * 4L;
        long byteSizeC = (long) M * N * 4L;

        long dA = context.allocate(byteSizeA);
        long dB = context.allocate(byteSizeB);
        long dC = context.allocate(byteSizeC);

        try {
            context.copyToDevice(dA, lhs.data());
            context.copyToDevice(dB, rhs.data());

            // Use cuBLAS with transpose flags for proper handling
            context.sgemmTranspose(dA, dB, dC, M, N, K, transposeA, transposeB);

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dC, byteSizeC);

            return List.of(output);

        } finally {
            context.free(dA);
            context.free(dB);
            context.free(dC);
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
