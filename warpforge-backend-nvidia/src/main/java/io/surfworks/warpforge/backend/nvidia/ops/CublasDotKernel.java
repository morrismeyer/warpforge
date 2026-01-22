package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * cuBLAS-based kernel for matrix multiplication (dot product).
 *
 * <p>This is the PRODUCTION tier implementation that uses NVIDIA's highly
 * optimized cuBLAS library. It provides maximum performance but limited
 * internal observability (timing only via external CUDA events).
 *
 * <p>Computes C[M,N] = A[M,K] * B[K,N]
 *
 * <p>For internal kernel observability, use {@link DotKernel} with salt
 * instrumentation (OPTIMIZED_OBSERVABLE tier).
 *
 * @see DotKernel PTX-based implementation with salt instrumentation
 */
public final class CublasDotKernel implements CudaOpKernel {

    private final CudaContext context;

    public CublasDotKernel(CudaContext context) {
        this.context = context;
        if (!context.isCublasAvailable()) {
            throw new UnsupportedOperationException(
                "cuBLAS is not available - use DotKernel for PTX-based matrix multiplication");
        }
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

        // Verify dtype is float32 (sgemm)
        if (lhs.dtype() != io.surfworks.warpforge.core.tensor.ScalarType.F32 ||
            rhs.dtype() != io.surfworks.warpforge.core.tensor.ScalarType.F32) {
            throw new UnsupportedOperationException(
                "CublasDotKernel currently only supports F32. Got: " + lhs.dtype() + " and " + rhs.dtype());
        }

        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long byteSizeA = M * K * 4L;
        long byteSizeB = K * N * 4L;
        long byteSizeC = M * N * 4L;

        long dA = context.allocate(byteSizeA);
        long dB = context.allocate(byteSizeB);
        long dC = context.allocate(byteSizeC);

        try {
            context.copyToDevice(dA, lhs.data());
            context.copyToDevice(dB, rhs.data());

            // Perform matrix multiplication via cuBLAS
            context.sgemm(dA, dB, dC, M, N, K);

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
        return op instanceof StableHloAst.DotOp;
    }
}
