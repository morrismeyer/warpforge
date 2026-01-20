package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RealOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.real.
 *
 * <p>Extracts the real part from a complex tensor.
 * Assumes complex tensors are stored in interleaved format
 * where adjacent pairs represent (real, imag).
 */
public final class RealKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Real requires exactly 1 input");
        }

        Tensor input = inputs.get(0);
        float[] inputData = input.toFloatArray();

        // Extract real parts (even indices in interleaved format)
        float[] outputData = new float[inputData.length / 2];
        for (int i = 0; i < outputData.length; i++) {
            outputData[i] = inputData[2 * i];
        }

        // Output shape removes the trailing complex dimension
        int[] inputShape = input.shape();
        int[] outputShape = new int[inputShape.length - 1];
        System.arraycopy(inputShape, 0, outputShape, 0, outputShape.length);

        Tensor output = Tensor.zeros(outputShape);
        output.copyFrom(outputData);
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.RealOp;
    }
}
