package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.negate - Element-wise negation. */
public class NegateKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return -x;
    }
}
