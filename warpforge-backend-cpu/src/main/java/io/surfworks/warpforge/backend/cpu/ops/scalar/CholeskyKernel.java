package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CholeskyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.cholesky.
 *
 * <p>Computes the Cholesky decomposition of a symmetric positive-definite matrix.
 * Returns L such that A = L * L^T (lower triangular) or A = U^T * U (upper triangular).
 */
public final class CholeskyKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        CholeskyOp cholOp = (CholeskyOp) op;

        if (inputs.size() != 1) {
            throw new IllegalArgumentException("cholesky requires exactly 1 input");
        }

        Tensor input = inputs.get(0);
        int[] shape = input.shape();

        if (shape.length < 2) {
            throw new IllegalArgumentException("cholesky requires at least 2D input");
        }

        int n = shape[shape.length - 1];
        int m = shape[shape.length - 2];
        if (n != m) {
            throw new IllegalArgumentException("cholesky requires square matrices");
        }

        boolean lower = cholOp.lower();

        float[] inputData = input.toFloatArray();
        float[] outputData = new float[inputData.length];

        // Handle batched inputs
        int batchSize = 1;
        for (int i = 0; i < shape.length - 2; i++) {
            batchSize *= shape[i];
        }

        int matrixSize = n * n;

        for (int batch = 0; batch < batchSize; batch++) {
            int offset = batch * matrixSize;
            choleskyDecomposition(inputData, outputData, offset, n, lower);
        }

        Tensor output = Tensor.zeros(shape);
        output.copyFrom(outputData);
        return List.of(output);
    }

    private void choleskyDecomposition(float[] input, float[] output, int offset, int n, boolean lower) {
        // Initialize output matrix (zeros above/below diagonal)
        for (int i = 0; i < n * n; i++) {
            output[offset + i] = 0;
        }

        // Cholesky-Crout algorithm
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = 0;

                if (i == j) {
                    // Diagonal elements
                    for (int k = 0; k < j; k++) {
                        float lji = lower ? output[offset + j * n + k] : output[offset + k * n + j];
                        sum += lji * lji;
                    }
                    float diag = input[offset + j * n + j] - sum;
                    if (diag <= 0) {
                        throw new IllegalArgumentException(
                            "Matrix is not positive definite (diagonal element " + j + " is " + diag + ")");
                    }
                    float val = (float) Math.sqrt(diag);
                    if (lower) {
                        output[offset + j * n + j] = val;
                    } else {
                        output[offset + j * n + j] = val;
                    }
                } else {
                    // Off-diagonal elements
                    for (int k = 0; k < j; k++) {
                        float lik = lower ? output[offset + i * n + k] : output[offset + k * n + i];
                        float ljk = lower ? output[offset + j * n + k] : output[offset + k * n + j];
                        sum += lik * ljk;
                    }
                    float ljj = lower ? output[offset + j * n + j] : output[offset + j * n + j];
                    float val = (input[offset + i * n + j] - sum) / ljj;
                    if (lower) {
                        output[offset + i * n + j] = val;
                    } else {
                        output[offset + j * n + i] = val;
                    }
                }
            }
        }
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.CholeskyOp;
    }
}
