package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TriangularSolveOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.triangular_solve.
 *
 * <p>Solves systems of linear equations with a triangular coefficient matrix.
 * op(A) * X = B where op is transpose or identity, and A is triangular.
 */
public final class TriangularSolveKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        TriangularSolveOp solveOp = (TriangularSolveOp) op;

        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                "triangular_solve requires 2 inputs (a, b)");
        }

        Tensor a = inputs.get(0);
        Tensor b = inputs.get(1);

        boolean leftSide = solveOp.leftSide();
        boolean lower = solveOp.lower();
        boolean transpose = solveOp.transposeA();
        boolean unitDiagonal = false;  // Default: non-unit diagonal

        int[] aShape = a.shape();
        int[] bShape = b.shape();

        int n = aShape[aShape.length - 1];
        int m = aShape[aShape.length - 2];

        if (n != m) {
            throw new IllegalArgumentException("triangular_solve requires square A matrix");
        }

        float[] aData = a.toFloatArray();
        float[] bData = b.toFloatArray();
        float[] xData = new float[bData.length];

        // Handle batched inputs
        int aBatchSize = 1;
        for (int i = 0; i < aShape.length - 2; i++) {
            aBatchSize *= aShape[i];
        }

        int bBatchSize = 1;
        for (int i = 0; i < bShape.length - 2; i++) {
            bBatchSize *= bShape[i];
        }

        int bCols = bShape[bShape.length - 1];
        int bRows = bShape[bShape.length - 2];

        int matrixSizeA = n * n;
        int matrixSizeB = bRows * bCols;

        for (int batch = 0; batch < Math.max(aBatchSize, bBatchSize); batch++) {
            int aOffset = (batch % aBatchSize) * matrixSizeA;
            int bOffset = batch * matrixSizeB;

            solveTriangular(aData, aOffset, bData, bOffset, xData, bOffset,
                           n, bCols, leftSide, lower, transpose, unitDiagonal);
        }

        Tensor output = Tensor.zeros(bShape);
        output.copyFrom(xData);
        return List.of(output);
    }

    private void solveTriangular(float[] a, int aOffset, float[] b, int bOffset,
                                  float[] x, int xOffset, int n, int nrhs,
                                  boolean leftSide, boolean lower, boolean transpose,
                                  boolean unitDiagonal) {
        // Copy b to x as starting point
        System.arraycopy(b, bOffset, x, xOffset, n * nrhs);

        if (leftSide) {
            // Solve A * X = B for X
            if ((lower && !transpose) || (!lower && transpose)) {
                // Forward substitution
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < nrhs; j++) {
                        float sum = x[xOffset + i * nrhs + j];
                        for (int k = 0; k < i; k++) {
                            int aIdx = transpose ? (k * n + i) : (i * n + k);
                            sum -= a[aOffset + aIdx] * x[xOffset + k * nrhs + j];
                        }
                        float diag = unitDiagonal ? 1.0f : a[aOffset + i * n + i];
                        x[xOffset + i * nrhs + j] = sum / diag;
                    }
                }
            } else {
                // Back substitution
                for (int i = n - 1; i >= 0; i--) {
                    for (int j = 0; j < nrhs; j++) {
                        float sum = x[xOffset + i * nrhs + j];
                        for (int k = i + 1; k < n; k++) {
                            int aIdx = transpose ? (k * n + i) : (i * n + k);
                            sum -= a[aOffset + aIdx] * x[xOffset + k * nrhs + j];
                        }
                        float diag = unitDiagonal ? 1.0f : a[aOffset + i * n + i];
                        x[xOffset + i * nrhs + j] = sum / diag;
                    }
                }
            }
        } else {
            // Solve X * A = B for X (treat as solving A^T * X^T = B^T)
            // Simplified implementation
            throw new UnsupportedOperationException(
                "Right-side triangular solve not yet implemented");
        }
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.TriangularSolveOp;
    }
}
