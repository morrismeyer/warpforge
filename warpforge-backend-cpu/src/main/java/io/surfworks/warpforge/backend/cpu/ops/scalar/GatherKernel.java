package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.gather - Gather slices from operand based on start indices.
 */
public class GatherKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Gather requires exactly 2 inputs, got " + inputs.size());
        }

        StableHloAst.GatherOp gatherOp = (StableHloAst.GatherOp) op;
        Tensor operand = inputs.get(0);
        Tensor startIndices = inputs.get(1);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] operandShape = operand.shape();
        int[] startIndicesShape = startIndices.shape();
        int[] outputShape = outputSpec.shape();

        float[] operandData = operand.toFloatArray();
        int[] startIndicesData = startIndices.toIntArray();
        float[] outputData = new float[(int) outputSpec.elementCount()];

        List<Long> offsetDims = gatherOp.offsetDims();
        List<Long> collapsedSliceDims = gatherOp.collapsedSliceDims();
        List<Long> startIndexMap = gatherOp.startIndexMap();
        int indexVectorDim = (int) gatherOp.indexVectorDim();
        List<Long> sliceSizes = gatherOp.sliceSizes();

        long[] operandStrides = computeStrides(operandShape);
        long[] outputStrides = computeStrides(outputShape);
        long[] startIndicesStrides = computeStrides(startIndicesShape);

        int outputRank = outputShape.length;
        int operandRank = operandShape.length;
        int startIndicesRank = startIndicesShape.length;

        // Compute batch dimensions in output (non-offset dims)
        int numBatchDims = outputRank - offsetDims.size();

        int[] outputIdx = new int[outputRank];
        for (int outFlatIdx = 0; outFlatIdx < outputData.length; outFlatIdx++) {
            unflattenIndex(outFlatIdx, outputStrides, outputIdx);

            // Extract batch indices from output (dimensions not in offsetDims)
            int[] batchIdx = new int[numBatchDims];
            int batchDimCounter = 0;
            for (int d = 0; d < outputRank; d++) {
                if (!offsetDims.contains((long) d)) {
                    batchIdx[batchDimCounter++] = outputIdx[d];
                }
            }

            // Extract offset indices (dimensions in offsetDims)
            int[] offsetIdx = new int[offsetDims.size()];
            for (int i = 0; i < offsetDims.size(); i++) {
                offsetIdx[i] = outputIdx[offsetDims.get(i).intValue()];
            }

            // Build index into startIndices tensor
            int[] startIdxIdx = new int[startIndicesRank];
            int batchCounter = 0;
            for (int d = 0; d < startIndicesRank; d++) {
                if (d == indexVectorDim) {
                    startIdxIdx[d] = 0; // Will be iterated below
                } else {
                    startIdxIdx[d] = batchIdx[batchCounter++];
                }
            }

            // Get the start indices for each operand dimension
            int[] operandIdx = new int[operandRank];
            int offsetCounter = 0;
            for (int d = 0; d < operandRank; d++) {
                if (collapsedSliceDims.contains((long) d)) {
                    // Collapsed dimension: start index from startIndices
                    int startMapIdx = startIndexMap.indexOf((long) d);
                    if (startMapIdx >= 0) {
                        startIdxIdx[indexVectorDim] = startMapIdx;
                        int startIdxFlat = flattenIndex(startIdxIdx, startIndicesStrides);
                        operandIdx[d] = startIndicesData[startIdxFlat];
                    } else {
                        operandIdx[d] = 0;
                    }
                } else {
                    // Non-collapsed dimension: combine start index + offset
                    int startMapIdx = startIndexMap.indexOf((long) d);
                    int startVal = 0;
                    if (startMapIdx >= 0) {
                        startIdxIdx[indexVectorDim] = startMapIdx;
                        int startIdxFlat = flattenIndex(startIdxIdx, startIndicesStrides);
                        startVal = startIndicesData[startIdxFlat];
                    }
                    operandIdx[d] = startVal + offsetIdx[offsetCounter++];
                }
            }

            // Clamp indices to valid range
            for (int d = 0; d < operandRank; d++) {
                operandIdx[d] = Math.max(0, Math.min(operandIdx[d], operandShape[d] - 1));
            }

            int operandFlatIdx = flattenIndex(operandIdx, operandStrides);
            outputData[outFlatIdx] = operandData[operandFlatIdx];
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
