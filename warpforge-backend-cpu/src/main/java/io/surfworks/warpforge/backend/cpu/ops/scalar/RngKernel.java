package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;
import java.util.Random;

/**
 * stablehlo.rng - Random number generation.
 */
public class RngKernel implements OpKernel {

    private final Random random = new Random();

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Rng requires exactly 2 inputs (a, b), got " + inputs.size());
        }

        StableHloAst.RngOp rngOp = (StableHloAst.RngOp) op;
        Tensor a = inputs.get(0);
        Tensor b = inputs.get(1);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        float aVal = a.toFloatArray()[0];
        float bVal = b.toFloatArray()[0];
        String distribution = rngOp.distribution();

        float[] outputData = new float[(int) outputSpec.elementCount()];

        switch (distribution.toLowerCase()) {
            case "uniform" -> {
                // Uniform distribution in [a, b)
                for (int i = 0; i < outputData.length; i++) {
                    outputData[i] = aVal + random.nextFloat() * (bVal - aVal);
                }
            }
            case "normal" -> {
                // Normal distribution with mean=a, stddev=b
                for (int i = 0; i < outputData.length; i++) {
                    outputData[i] = (float) (aVal + random.nextGaussian() * bVal);
                }
            }
            default -> throw new UnsupportedOperationException("Unknown RNG distribution: " + distribution);
        }

        Tensor output = Tensor.fromFloatArray(outputData, outputSpec.shape());
        return List.of(output);
    }
}
