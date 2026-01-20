package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReducePrecisionOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.reduce_precision.
 *
 * <p>Reduces the precision of floating-point values by truncating
 * the mantissa and clamping the exponent to specified bit widths.
 */
public final class ReducePrecisionKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        ReducePrecisionOp reduceOp = (ReducePrecisionOp) op;

        if (inputs.size() != 1) {
            throw new IllegalArgumentException("reduce_precision requires exactly 1 input");
        }

        Tensor input = inputs.get(0);
        int exponentBits = reduceOp.exponentBits();
        int mantissaBits = reduceOp.mantissaBits();

        float[] inputData = input.toFloatArray();
        float[] outputData = new float[inputData.length];

        for (int i = 0; i < inputData.length; i++) {
            outputData[i] = reducePrecision(inputData[i], exponentBits, mantissaBits);
        }

        Tensor output = Tensor.zeros(input.shape());
        output.copyFrom(outputData);
        return List.of(output);
    }

    private float reducePrecision(float value, int exponentBits, int mantissaBits) {
        if (Float.isNaN(value) || Float.isInfinite(value)) {
            return value;
        }

        // Convert to bits
        int bits = Float.floatToRawIntBits(value);
        int sign = bits & 0x80000000;
        int exponent = (bits >> 23) & 0xFF;
        int mantissa = bits & 0x7FFFFF;

        // Clamp exponent to specified bits
        // IEEE 754 float has 8 exponent bits, bias 127
        int maxExp = (1 << exponentBits) - 1;
        int bias = (1 << (exponentBits - 1)) - 1;

        // Adjust exponent for reduced precision
        int unbiacedExp = exponent - 127;
        int newBias = (1 << (exponentBits - 1)) - 1;
        int minExpValue = -newBias + 1;
        int maxExpValue = maxExp - newBias;

        if (unbiacedExp < minExpValue) {
            // Underflow to zero
            return 0.0f;
        } else if (unbiacedExp > maxExpValue) {
            // Overflow to infinity
            return sign != 0 ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
        }

        // Truncate mantissa to specified bits
        int mantissaShift = 23 - mantissaBits;
        if (mantissaShift > 0) {
            // Round towards nearest even
            int roundBit = 1 << (mantissaShift - 1);
            int truncMask = ~((1 << mantissaShift) - 1);
            int remainder = mantissa & ((1 << mantissaShift) - 1);

            if (remainder > roundBit || (remainder == roundBit && (mantissa & (1 << mantissaShift)) != 0)) {
                mantissa += (1 << mantissaShift);
            }
            mantissa &= truncMask;
        }

        // Reconstruct float
        int newBits = sign | (exponent << 23) | (mantissa & 0x7FFFFF);
        return Float.intBitsToFloat(newBits);
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.ReducePrecisionOp;
    }
}
