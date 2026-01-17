package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.round_nearest_even - Round to nearest integer, ties to even (banker's rounding).
 */
public class RoundNearestEvenKernel extends UnaryElementwiseKernel {

    @Override
    protected float apply(float x) {
        return (float) Math.rint(x);
    }
}
