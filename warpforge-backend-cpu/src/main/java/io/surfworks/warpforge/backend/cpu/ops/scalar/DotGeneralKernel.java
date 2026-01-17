package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.Arrays;
import java.util.List;

/**
 * stablehlo.dot_general - General matrix multiplication with batching and contracting dimensions.
 */
public class DotGeneralKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("DotGeneral requires exactly 2 inputs, got " + inputs.size());
        }

        StableHloAst.DotGeneralOp dotOp = (StableHloAst.DotGeneralOp) op;
        Tensor lhs = inputs.get(0);
        Tensor rhs = inputs.get(1);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] lhsShape = lhs.shape();
        int[] rhsShape = rhs.shape();
        int[] outputShape = outputSpec.shape();

        float[] lhsData = lhs.toFloatArray();
        float[] rhsData = rhs.toFloatArray();
        float[] outputData = new float[(int) outputSpec.elementCount()];

        StableHloAst.DotDimensionNumbers dimNums = dotOp.dimensionNumbers();
        List<Long> lhsBatchDims = dimNums.lhsBatchingDimensions();
        List<Long> rhsBatchDims = dimNums.rhsBatchingDimensions();
        List<Long> lhsContractDims = dimNums.lhsContractingDimensions();
        List<Long> rhsContractDims = dimNums.rhsContractingDimensions();

        long[] lhsStrides = computeStrides(lhsShape);
        long[] rhsStrides = computeStrides(rhsShape);
        long[] outputStrides = computeStrides(outputShape);

        int lhsRank = lhsShape.length;
        int rhsRank = rhsShape.length;
        int outputRank = outputShape.length;

        // Identify free dimensions (not batch, not contracting)
        int[] lhsFreeDims = getFreeDims(lhsRank, lhsBatchDims, lhsContractDims);
        int[] rhsFreeDims = getFreeDims(rhsRank, rhsBatchDims, rhsContractDims);

        // Compute contracting dimension product
        int contractingSize = 1;
        for (Long dim : lhsContractDims) {
            contractingSize *= lhsShape[dim.intValue()];
        }

        int[] outputIdx = new int[outputRank];
        for (int outFlatIdx = 0; outFlatIdx < outputData.length; outFlatIdx++) {
            unflattenIndex(outFlatIdx, outputStrides, outputIdx);

            // Map output indices to batch and free indices
            int outIdxPos = 0;

            // Extract batch indices
            int[] batchIdx = new int[lhsBatchDims.size()];
            for (int b = 0; b < lhsBatchDims.size(); b++) {
                batchIdx[b] = outputIdx[outIdxPos++];
            }

            // Extract lhs free indices
            int[] lhsFreeIdx = new int[lhsFreeDims.length];
            for (int f = 0; f < lhsFreeDims.length; f++) {
                lhsFreeIdx[f] = outputIdx[outIdxPos++];
            }

            // Extract rhs free indices
            int[] rhsFreeIdx = new int[rhsFreeDims.length];
            for (int f = 0; f < rhsFreeDims.length; f++) {
                rhsFreeIdx[f] = outputIdx[outIdxPos++];
            }

            // Compute dot product over contracting dimensions
            float sum = 0;

            // Iterate over all contracting dimension combinations
            int[] contractIdx = new int[lhsContractDims.size()];
            for (int c = 0; c < contractingSize; c++) {
                // Build lhs index
                int[] lhsIdx = new int[lhsRank];
                for (int b = 0; b < lhsBatchDims.size(); b++) {
                    lhsIdx[lhsBatchDims.get(b).intValue()] = batchIdx[b];
                }
                for (int f = 0; f < lhsFreeDims.length; f++) {
                    lhsIdx[lhsFreeDims[f]] = lhsFreeIdx[f];
                }
                for (int cd = 0; cd < lhsContractDims.size(); cd++) {
                    lhsIdx[lhsContractDims.get(cd).intValue()] = contractIdx[cd];
                }

                // Build rhs index
                int[] rhsIdx = new int[rhsRank];
                for (int b = 0; b < rhsBatchDims.size(); b++) {
                    rhsIdx[rhsBatchDims.get(b).intValue()] = batchIdx[b];
                }
                for (int f = 0; f < rhsFreeDims.length; f++) {
                    rhsIdx[rhsFreeDims[f]] = rhsFreeIdx[f];
                }
                for (int cd = 0; cd < rhsContractDims.size(); cd++) {
                    rhsIdx[rhsContractDims.get(cd).intValue()] = contractIdx[cd];
                }

                int lhsFlatIdx = flattenIndex(lhsIdx, lhsStrides);
                int rhsFlatIdx = flattenIndex(rhsIdx, rhsStrides);
                sum += lhsData[lhsFlatIdx] * rhsData[rhsFlatIdx];

                // Advance contracting indices
                for (int cd = lhsContractDims.size() - 1; cd >= 0; cd--) {
                    contractIdx[cd]++;
                    if (contractIdx[cd] < lhsShape[lhsContractDims.get(cd).intValue()]) {
                        break;
                    }
                    contractIdx[cd] = 0;
                }
            }

            outputData[outFlatIdx] = sum;
        }

        Tensor output = Tensor.fromFloatArray(outputData, outputShape);
        return List.of(output);
    }

    private int[] getFreeDims(int rank, List<Long> batchDims, List<Long> contractDims) {
        int[] free = new int[rank - batchDims.size() - contractDims.size()];
        int freeIdx = 0;
        for (int d = 0; d < rank; d++) {
            if (!batchDims.contains((long) d) && !contractDims.contains((long) d)) {
                free[freeIdx++] = d;
            }
        }
        return free;
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
