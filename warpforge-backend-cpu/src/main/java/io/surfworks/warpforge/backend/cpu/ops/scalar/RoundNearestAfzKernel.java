package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.round_nearest_afz - Round to nearest integer, ties away from zero.
 */
public class RoundNearestAfzKernel extends UnaryElementwiseKernel {

    @Override
    protected float apply(float x) {
        // Math.round rounds ties away from zero for positive numbers
        // but we need to handle negative numbers correctly
        if (x >= 0) {
            return (float) Math.floor(x + 0.5f);
        } else {
            return (float) Math.ceil(x - 0.5f);
        }
    }
}
