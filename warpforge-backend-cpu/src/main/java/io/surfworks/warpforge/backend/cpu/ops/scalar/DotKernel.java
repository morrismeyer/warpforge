package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.dot - Simple matrix multiplication.
 * For 2D inputs: standard matrix multiplication.
 * For 1D inputs: dot product.
 */
public class DotKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Dot requires exactly 2 inputs, got " + inputs.size());
        }

        Tensor lhs = inputs.get(0);
        Tensor rhs = inputs.get(1);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] lhsShape = lhs.shape();
        int[] rhsShape = rhs.shape();
        int[] outputShape = outputSpec.shape();

        float[] lhsData = lhs.toFloatArray();
        float[] rhsData = rhs.toFloatArray();

        // Handle different input ranks
        if (lhsShape.length == 1 && rhsShape.length == 1) {
            // Vector dot product
            float result = 0;
            for (int i = 0; i < lhsData.length; i++) {
                result += lhsData[i] * rhsData[i];
            }
            Tensor output = Tensor.fromFloatArray(new float[]{result});
            return List.of(output);
        } else if (lhsShape.length == 2 && rhsShape.length == 2) {
            // Matrix multiplication: [M, K] x [K, N] = [M, N]
            int M = lhsShape[0];
            int K = lhsShape[1];
            int N = rhsShape[1];

            float[] outputData = new float[M * N];

            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += lhsData[m * K + k] * rhsData[k * N + n];
                    }
                    outputData[m * N + n] = sum;
                }
            }

            Tensor output = Tensor.fromFloatArray(outputData, outputShape);
            return List.of(output);
        } else if (lhsShape.length == 2 && rhsShape.length == 1) {
            // Matrix-vector: [M, K] x [K] = [M]
            int M = lhsShape[0];
            int K = lhsShape[1];

            float[] outputData = new float[M];

            for (int m = 0; m < M; m++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += lhsData[m * K + k] * rhsData[k];
                }
                outputData[m] = sum;
            }

            Tensor output = Tensor.fromFloatArray(outputData, outputShape);
            return List.of(output);
        } else if (lhsShape.length == 1 && rhsShape.length == 2) {
            // Vector-matrix: [K] x [K, N] = [N]
            int K = lhsShape[0];
            int N = rhsShape[1];

            float[] outputData = new float[N];

            for (int n = 0; n < N; n++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += lhsData[k] * rhsData[k * N + n];
                }
                outputData[n] = sum;
            }

            Tensor output = Tensor.fromFloatArray(outputData, outputShape);
            return List.of(output);
        } else {
            throw new UnsupportedOperationException(
                "Dot operation not supported for shapes: " +
                java.util.Arrays.toString(lhsShape) + " x " + java.util.Arrays.toString(rhsShape));
        }
    }
}
