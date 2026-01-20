package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ComplexOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.complex.
 *
 * <p>Constructs a complex tensor from real and imaginary parts.
 * The output is an interleaved representation where adjacent pairs
 * of elements represent (real, imag) components.
 */
public final class ComplexKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        ComplexOp complexOp = (ComplexOp) op;

        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Complex requires exactly 2 inputs (real, imag)");
        }

        Tensor real = inputs.get(0);
        Tensor imag = inputs.get(1);

        if (!java.util.Arrays.equals(real.shape(), imag.shape())) {
            throw new IllegalArgumentException("Real and imaginary parts must have same shape");
        }

        float[] realData = real.toFloatArray();
        float[] imagData = imag.toFloatArray();

        // Interleave real and imaginary parts
        float[] outputData = new float[realData.length * 2];
        for (int i = 0; i < realData.length; i++) {
            outputData[2 * i] = realData[i];
            outputData[2 * i + 1] = imagData[i];
        }

        // Output shape adds a trailing dimension of 2 for complex representation
        int[] inputShape = real.shape();
        int[] outputShape = new int[inputShape.length + 1];
        System.arraycopy(inputShape, 0, outputShape, 0, inputShape.length);
        outputShape[inputShape.length] = 2;

        Tensor output = Tensor.zeros(outputShape);
        output.copyFrom(outputData);
        return List.of(output);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.ComplexOp;
    }
}
