package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.exponential - Element-wise exponential. */
public class ExpKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) Math.exp(x);
    }
}
