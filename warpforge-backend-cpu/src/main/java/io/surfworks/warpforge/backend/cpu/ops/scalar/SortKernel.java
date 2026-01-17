package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.Arrays;
import java.util.List;

/**
 * stablehlo.sort - Sort tensor along a dimension.
 */
public class SortKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("Sort requires at least 1 input");
        }

        StableHloAst.SortOp sortOp = (StableHloAst.SortOp) op;
        Tensor input = inputs.get(0);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] shape = input.shape();
        int[] outputShape = outputSpec.shape();
        int rank = shape.length;
        int sortDim = (int) sortOp.dimension();
        boolean isStable = sortOp.isStable();

        float[] inputData = input.toFloatArray();
        float[] outputData = inputData.clone();

        if (rank == 0) {
            // Scalar, nothing to sort
            return List.of(Tensor.fromFloatArray(outputData, outputShape));
        }

        int sortDimSize = shape[sortDim];
        long[] strides = computeStrides(shape);

        // Number of sort operations needed
        int totalElements = inputData.length;
        int numSortOps = totalElements / sortDimSize;

        // For each 1D slice along sort dimension
        for (int sortOpIdx = 0; sortOpIdx < numSortOps; sortOpIdx++) {
            // Extract the 1D slice
            float[] slice = new float[sortDimSize];
            int baseIdx = getBaseIndex(sortOpIdx, sortDim, shape, strides);

            for (int i = 0; i < sortDimSize; i++) {
                int flatIdx = baseIdx + (int) (i * strides[sortDim]);
                slice[i] = inputData[flatIdx];
            }

            // Sort the slice
            if (isStable) {
                // For stable sort, use a sorting algorithm that maintains order
                // Java's Arrays.sort is stable for objects
                Float[] boxed = new Float[sortDimSize];
                for (int i = 0; i < sortDimSize; i++) boxed[i] = slice[i];
                Arrays.sort(boxed);
                for (int i = 0; i < sortDimSize; i++) slice[i] = boxed[i];
            } else {
                Arrays.sort(slice);
            }

            // Write back the sorted slice
            for (int i = 0; i < sortDimSize; i++) {
                int flatIdx = baseIdx + (int) (i * strides[sortDim]);
                outputData[flatIdx] = slice[i];
            }
        }

        Tensor output = Tensor.fromFloatArray(outputData, outputShape);
        return List.of(output);
    }

    private int getBaseIndex(int sortOpIdx, int sortDim, int[] shape, long[] strides) {
        int rank = shape.length;
        int[] idx = new int[rank];

        // Compute the multi-dimensional index for this sort operation
        int remaining = sortOpIdx;
        for (int d = 0; d < rank; d++) {
            if (d == sortDim) continue;
            int dimSize = shape[d];
            int dimStride = 1;
            for (int dd = d + 1; dd < rank; dd++) {
                if (dd != sortDim) dimStride *= shape[dd];
            }
            idx[d] = remaining / dimStride;
            remaining %= dimStride;
        }

        // Compute flat base index
        int baseIdx = 0;
        for (int d = 0; d < rank; d++) {
            baseIdx += idx[d] * strides[d];
        }
        return baseIdx;
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
}
