package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;
import java.util.function.BinaryOperator;

/**
 * stablehlo.reduce_window - Windowed reduction (pooling).
 */
public class ReduceWindowKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("ReduceWindow requires exactly 2 inputs, got " + inputs.size());
        }

        StableHloAst.ReduceWindowOp reduceWindowOp = (StableHloAst.ReduceWindowOp) op;
        Tensor operand = inputs.get(0);
        Tensor initValue = inputs.get(1);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] inputShape = operand.shape();
        int[] outputShape = outputSpec.shape();
        int rank = inputShape.length;

        List<Long> windowDims = reduceWindowOp.windowDimensions();
        List<Long> windowStrides = reduceWindowOp.windowStrides();
        List<Long> baseDilations = reduceWindowOp.baseDilations();
        List<Long> windowDilations = reduceWindowOp.windowDilations();
        List<Long> paddingLow = reduceWindowOp.paddingLow();
        List<Long> paddingHigh = reduceWindowOp.paddingHigh();

        float[] inputData = operand.toFloatArray();
        float initVal = initValue.toFloatArray()[0];
        float[] outputData = new float[(int) outputSpec.elementCount()];

        BinaryOperator<Float> reducer = getReducer(reduceWindowOp.reducer());

        long[] inputStrides = computeStrides(inputShape);
        long[] outputStrides = computeStrides(outputShape);

        int[] outputIdx = new int[rank];
        for (int outFlatIdx = 0; outFlatIdx < outputData.length; outFlatIdx++) {
            unflattenIndex(outFlatIdx, outputStrides, outputIdx);

            float result = initVal;

            // Iterate over window
            int[] windowIdx = new int[rank];
            boolean done = false;
            while (!done) {
                // Compute input index for this window position
                int[] inputIdx = new int[rank];
                boolean valid = true;
                for (int d = 0; d < rank; d++) {
                    int stride = windowStrides.isEmpty() ? 1 : windowStrides.get(d).intValue();
                    int baseDilation = baseDilations.isEmpty() ? 1 : baseDilations.get(d).intValue();
                    int windowDilation = windowDilations.isEmpty() ? 1 : windowDilations.get(d).intValue();
                    int padLow = paddingLow.isEmpty() ? 0 : paddingLow.get(d).intValue();

                    int dilatedBase = outputIdx[d] * stride - padLow;
                    int dilatedWindow = windowIdx[d] * windowDilation;
                    int idx = dilatedBase + dilatedWindow;

                    // Check if within dilated input bounds
                    if (baseDilation > 1 && idx % baseDilation != 0) {
                        valid = false;
                        break;
                    }
                    idx = baseDilation > 1 ? idx / baseDilation : idx;

                    if (idx < 0 || idx >= inputShape[d]) {
                        valid = false;
                        break;
                    }
                    inputIdx[d] = idx;
                }

                if (valid) {
                    int inFlatIdx = flattenIndex(inputIdx, inputStrides);
                    result = reducer.apply(result, inputData[inFlatIdx]);
                }

                // Advance window index
                for (int d = rank - 1; d >= 0; d--) {
                    windowIdx[d]++;
                    if (windowIdx[d] < windowDims.get(d).intValue()) {
                        break;
                    }
                    windowIdx[d] = 0;
                    if (d == 0) {
                        done = true;
                    }
                }
            }

            outputData[outFlatIdx] = result;
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
            default -> throw new UnsupportedOperationException("Unknown reducer: " + reducerName);
        };
    }

    private long[] computeStrides(int[] shape) {
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
        long result = 0;
        for (int i = 0; i < indices.length; i++) {
            result += indices[i] * strides[i];
        }
        return (int) result;
    }
}
