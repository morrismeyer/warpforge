package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.sign - Element-wise sign function.
 * Returns -1 for negative, 0 for zero, 1 for positive.
 */
public class SignKernel extends UnaryElementwiseKernel {

    @Override
    protected float apply(float x) {
        return Math.signum(x);
    }
}
