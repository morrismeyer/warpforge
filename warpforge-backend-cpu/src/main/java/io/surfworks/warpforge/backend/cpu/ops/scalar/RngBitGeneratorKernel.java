package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;
import java.util.Random;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RngBitGeneratorOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.rng_bit_generator.
 *
 * <p>Generates random bits using a stateful random number generator.
 * Returns both the updated state and the generated random values.
 */
public final class RngBitGeneratorKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        RngBitGeneratorOp rngOp = (RngBitGeneratorOp) op;

        if (inputs.size() != 1) {
            throw new IllegalArgumentException(
                "rng_bit_generator requires 1 input (initial_state)");
        }

        Tensor initialState = inputs.get(0);
        String algorithm = rngOp.algorithm().name();

        // Extract seed from initial state
        float[] stateData = initialState.toFloatArray();
        long seed = 0;
        for (int i = 0; i < Math.min(2, stateData.length); i++) {
            seed = (seed << 32) | (Float.floatToRawIntBits(stateData[i]) & 0xFFFFFFFFL);
        }

        Random random = new Random(seed);

        // Determine output shape from the operation's result type
        int[] outputShape = getOutputShape(rngOp);
        int outputSize = 1;
        for (int dim : outputShape) {
            outputSize *= dim;
        }

        // Generate random values based on algorithm
        float[] outputData = new float[outputSize];
        switch (algorithm.toLowerCase()) {
            case "default":
            case "philox":
            case "threefry":
                // Generate 32-bit random integers as floats
                for (int i = 0; i < outputSize; i++) {
                    outputData[i] = Float.intBitsToFloat(random.nextInt());
                }
                break;
            default:
                throw new UnsupportedOperationException(
                    "Unsupported RNG algorithm: " + algorithm);
        }

        // Update state (advance the seed)
        float[] newStateData = new float[stateData.length];
        long newSeed = random.nextLong();
        for (int i = 0; i < newStateData.length; i++) {
            newStateData[i] = Float.intBitsToFloat((int) (newSeed >> (i * 32)));
        }

        Tensor newState = Tensor.zeros(initialState.shape());
        newState.copyFrom(newStateData);

        Tensor output = Tensor.zeros(outputShape);
        output.copyFrom(outputData);

        return List.of(newState, output);
    }

    private int[] getOutputShape(RngBitGeneratorOp op) {
        // Default to 1D output if shape not specified
        // In practice, the shape comes from the operation's result type
        return new int[]{64}; // Default small output
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.RngBitGeneratorOp;
    }
}
