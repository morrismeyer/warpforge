package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.Arrays;
import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SelectAndScatterOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.select_and_scatter.
 *
 * <p>Composite operation that selects elements from a source based on
 * a window-based selection function, then scatters updates to those positions.
 * Commonly used in max-pooling gradient computations.
 */
public final class SelectAndScatterKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        SelectAndScatterOp sasOp = (SelectAndScatterOp) op;

        if (inputs.size() != 3) {
            throw new IllegalArgumentException(
                "select_and_scatter requires 3 inputs (operand, source, init_value)");
        }

        Tensor operand = inputs.get(0);
        Tensor source = inputs.get(1);
        Tensor initValue = inputs.get(2);

        List<Long> windowDimensions = sasOp.windowDimensions();
        List<Long> windowStrides = sasOp.windowStrides();
        List<Long> padding = sasOp.padding();

        int[] operandShape = operand.shape();
        int[] sourceShape = source.shape();
        float[] operandData = operand.toFloatArray();
        float[] sourceData = source.toFloatArray();
        float initVal = initValue.toFloatArray()[0];

        // Initialize output with init value
        float[] outputData = new float[operandData.length];
        Arrays.fill(outputData, initVal);

        int rank = operandShape.length;

        // Calculate strides
        int[] operandStrides = new int[rank];
        int[] sourceStrides = new int[rank];
        operandStrides[rank - 1] = 1;
        sourceStrides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) {
            operandStrides[i] = operandStrides[i + 1] * operandShape[i + 1];
            sourceStrides[i] = sourceStrides[i + 1] * sourceShape[i + 1];
        }

        // Convert window params
        int[] winDims = new int[rank];
        int[] winStrides = new int[rank];
        int[] padLow = new int[rank];
        for (int i = 0; i < rank; i++) {
            winDims[i] = i < windowDimensions.size() ? windowDimensions.get(i).intValue() : 1;
            winStrides[i] = i < windowStrides.size() ? windowStrides.get(i).intValue() : 1;
            padLow[i] = (i * 2) < padding.size() ? padding.get(i * 2).intValue() : 0;
        }

        // Iterate over source positions
        int sourceSize = (int) source.elementCount();
        for (int srcFlatIdx = 0; srcFlatIdx < sourceSize; srcFlatIdx++) {
            // Convert to source coordinates
            int[] srcCoords = new int[rank];
            int remaining = srcFlatIdx;
            for (int d = 0; d < rank; d++) {
                srcCoords[d] = remaining / sourceStrides[d];
                remaining = remaining % sourceStrides[d];
            }

            // Find selected position in window (max selection)
            int selectedIdx = -1;
            float selectedVal = Float.NEGATIVE_INFINITY;

            // Window start position in operand
            int[] winStart = new int[rank];
            for (int d = 0; d < rank; d++) {
                winStart[d] = srcCoords[d] * winStrides[d] - padLow[d];
            }

            // Iterate over window
            int[] winCoords = new int[rank];
            boolean done = false;
            while (!done) {
                // Check if within bounds
                boolean inBounds = true;
                int operandIdx = 0;
                for (int d = 0; d < rank; d++) {
                    int pos = winStart[d] + winCoords[d];
                    if (pos < 0 || pos >= operandShape[d]) {
                        inBounds = false;
                        break;
                    }
                    operandIdx += pos * operandStrides[d];
                }

                if (inBounds) {
                    float val = operandData[operandIdx];
                    if (val > selectedVal) {
                        selectedVal = val;
                        selectedIdx = operandIdx;
                    }
                }

                // Advance window coordinates
                for (int d = rank - 1; d >= 0; d--) {
                    winCoords[d]++;
                    if (winCoords[d] < winDims[d]) {
                        break;
                    }
                    winCoords[d] = 0;
                    if (d == 0) {
                        done = true;
                    }
                }
            }

            // Scatter source value to selected position (add scatter)
            if (selectedIdx >= 0) {
                outputData[selectedIdx] += sourceData[srcFlatIdx];
            }
        }

        Tensor output = Tensor.zeros(operandShape);
        output.copyFrom(outputData);
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.SelectAndScatterOp;
    }
}
