package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BinaryOperator;

/**
 * stablehlo.reduce - Reduction operation along specified dimensions.
 */
public class ReduceKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Reduce requires exactly 2 inputs, got " + inputs.size());
        }

        StableHloAst.ReduceOp reduceOp = (StableHloAst.ReduceOp) op;
        Tensor operand = inputs.get(0);
        Tensor initValue = inputs.get(1);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] inputShape = operand.shape();
        int[] outputShape = outputSpec.shape();
        int inputRank = inputShape.length;

        Set<Integer> reduceDims = new HashSet<>();
        for (Long dim : reduceOp.dimensions()) {
            reduceDims.add(dim.intValue());
        }

        float[] inputData = operand.toFloatArray();
        float initVal = initValue.toFloatArray()[0];
        float[] outputData = new float[(int) outputSpec.elementCount()];

        // Initialize output with init value
        java.util.Arrays.fill(outputData, initVal);

        // Get reducer function
        BinaryOperator<Float> reducer = getReducer(reduceOp.reducer());

        long[] inputStrides = computeStrides(inputShape);
        long[] outputStrides = computeStrides(outputShape);

        int[] inputIdx = new int[inputRank];
        for (int inFlatIdx = 0; inFlatIdx < inputData.length; inFlatIdx++) {
            unflattenIndex(inFlatIdx, inputStrides, inputIdx);

            // Map input index to output index (skip reduced dimensions)
            int[] outputIdx = new int[outputShape.length];
            int outDim = 0;
            for (int d = 0; d < inputRank; d++) {
                if (!reduceDims.contains(d)) {
                    outputIdx[outDim++] = inputIdx[d];
                }
            }

            int outFlatIdx = outputShape.length > 0 ? flattenIndex(outputIdx, outputStrides) : 0;
            outputData[outFlatIdx] = reducer.apply(outputData[outFlatIdx], inputData[inFlatIdx]);
        }

        Tensor output = Tensor.fromFloatArray(outputData, outputShape);
        return List.of(output);
    }

    private BinaryOperator<Float> getReducer(String reducerName) {
        return switch (reducerName.toLowerCase()) {
            case "add", "sum" -> Float::sum;
            case "mul", "multiply", "prod" -> (a, b) -> a * b;
            case "max", "maximum" -> Float::max;
            case "min", "minimum" -> Float::min;
            case "and" -> (a, b) -> (a != 0 && b != 0) ? 1.0f : 0.0f;
            case "or" -> (a, b) -> (a != 0 || b != 0) ? 1.0f : 0.0f;
            default -> throw new UnsupportedOperationException("Unknown reducer: " + reducerName);
        };
    }

    private long[] computeStrides(int[] shape) {
        if (shape.length == 0) return new long[0];
        long[] strides = new long[shape.length];
        long stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private void unflattenIndex(long flatIdx, long[] strides, int[] result) {
        for (int i = 0; i < strides.length; i++) {
            result[i] = (int) (flatIdx / strides[i]);
            flatIdx %= strides[i];
        }
    }

    private int flattenIndex(int[] indices, long[] strides) {
        if (strides.length == 0) return 0;
        long result = 0;
        for (int i = 0; i < indices.length; i++) {
            result += indices[i] * strides[i];
        }
        return (int) result;
    }
}
