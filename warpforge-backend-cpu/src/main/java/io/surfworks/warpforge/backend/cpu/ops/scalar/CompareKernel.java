package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.compare - Element-wise comparison.
 * Returns a boolean tensor.
 */
public class CompareKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.CompareOp compareOp)) {
            throw new IllegalArgumentException("Expected CompareOp");
        }
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Compare requires exactly 2 inputs");
        }

        Tensor lhs = inputs.get(0);
        Tensor rhs = inputs.get(1);

        float[] lhsData = lhs.toFloatArray();
        float[] rhsData = rhs.toFloatArray();
        float[] outputData = new float[lhsData.length];

        StableHloAst.ComparisonDirection direction = compareOp.direction();
        for (int i = 0; i < lhsData.length; i++) {
            boolean result = compare(lhsData[i], rhsData[i], direction);
            outputData[i] = result ? 1.0f : 0.0f;
        }

        // Compare returns a boolean tensor with same shape
        TensorSpec spec = TensorSpec.fromAst(op.tensorResultType());
        Tensor output = Tensor.fromFloatArray(outputData, spec.shape());
        return List.of(output);
    }

    private boolean compare(float a, float b, StableHloAst.ComparisonDirection direction) {
        return switch (direction) {
            case EQ -> a == b;
            case NE -> a != b;
            case LT -> a < b;
            case LE -> a <= b;
            case GT -> a > b;
            case GE -> a >= b;
        };
    }
}
