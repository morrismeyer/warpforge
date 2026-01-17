package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.Arrays;
import java.util.List;

/**
 * stablehlo.scatter - Scatter updates into operand at specified indices.
 */
public class ScatterKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 3) {
            throw new IllegalArgumentException("Scatter requires exactly 3 inputs, got " + inputs.size());
        }

        StableHloAst.ScatterOp scatterOp = (StableHloAst.ScatterOp) op;
        Tensor operand = inputs.get(0);
        Tensor scatterIndices = inputs.get(1);
        Tensor updates = inputs.get(2);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] operandShape = operand.shape();
        int[] scatterIndicesShape = scatterIndices.shape();
        int[] updatesShape = updates.shape();
        int[] outputShape = outputSpec.shape();

        float[] operandData = operand.toFloatArray();
        int[] scatterIndicesData = scatterIndices.toIntArray();
        float[] updatesData = updates.toFloatArray();

        // Start with a copy of the operand
        float[] outputData = Arrays.copyOf(operandData, operandData.length);

        List<Long> updateWindowDims = scatterOp.updateWindowDims();
        List<Long> insertedWindowDims = scatterOp.insertedWindowDims();
        List<Long> scatterDimsToOperandDims = scatterOp.scatterDimsToOperandDims();
        int indexVectorDim = (int) scatterOp.indexVectorDim();

        long[] operandStrides = computeStrides(operandShape);
        long[] scatterIndicesStrides = computeStrides(scatterIndicesShape);
        long[] updatesStrides = computeStrides(updatesShape);

        int operandRank = operandShape.length;
        int updatesRank = updatesShape.length;
        int scatterIndicesRank = scatterIndicesShape.length;

        // Number of scatter dimensions (batch dims in updates not in updateWindowDims)
        int numScatterDims = updatesRank - updateWindowDims.size();

        int[] updatesIdx = new int[updatesRank];
        for (int updateFlatIdx = 0; updateFlatIdx < updatesData.length; updateFlatIdx++) {
            unflattenIndex(updateFlatIdx, updatesStrides, updatesIdx);

            // Extract scatter indices (dimensions not in updateWindowDims)
            int[] scatterIdx = new int[numScatterDims];
            int scatterCounter = 0;
            for (int d = 0; d < updatesRank; d++) {
                if (!updateWindowDims.contains((long) d)) {
                    scatterIdx[scatterCounter++] = updatesIdx[d];
                }
            }

            // Extract window indices (dimensions in updateWindowDims)
            int[] windowIdx = new int[updateWindowDims.size()];
            for (int i = 0; i < updateWindowDims.size(); i++) {
                windowIdx[i] = updatesIdx[updateWindowDims.get(i).intValue()];
            }

            // Build index into scatterIndices tensor to get start indices
            int[] scatterIndicesIdx = new int[scatterIndicesRank];
            int si = 0;
            for (int d = 0; d < scatterIndicesRank; d++) {
                if (d == indexVectorDim) {
                    scatterIndicesIdx[d] = 0; // Will be iterated
                } else if (si < scatterIdx.length) {
                    scatterIndicesIdx[d] = scatterIdx[si++];
                }
            }

            // Build operand index
            int[] operandIdx = new int[operandRank];
            int windowCounter = 0;
            for (int d = 0; d < operandRank; d++) {
                int scatterDimIdx = scatterDimsToOperandDims.indexOf((long) d);
                if (scatterDimIdx >= 0 && scatterIndicesRank > 0) {
                    // This dimension gets its index from scatter_indices
                    scatterIndicesIdx[indexVectorDim] = scatterDimIdx;
                    int idxFlat = flattenIndex(scatterIndicesIdx, scatterIndicesStrides);
                    operandIdx[d] = scatterIndicesData[idxFlat];
                }

                if (!insertedWindowDims.contains((long) d)) {
                    // Add window offset
                    if (windowCounter < windowIdx.length) {
                        operandIdx[d] += windowIdx[windowCounter++];
                    }
                }
            }

            // Check bounds and scatter
            boolean inBounds = true;
            for (int d = 0; d < operandRank; d++) {
                if (operandIdx[d] < 0 || operandIdx[d] >= operandShape[d]) {
                    inBounds = false;
                    break;
                }
            }

            if (inBounds) {
                int operandFlatIdx = flattenIndex(operandIdx, operandStrides);
                // Default scatter semantics: replace (can be extended to support add, etc.)
                outputData[operandFlatIdx] = updatesData[updateFlatIdx];
            }
        }

        Tensor output = Tensor.fromFloatArray(outputData, outputShape);
        return List.of(output);
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
