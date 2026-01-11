package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.abs - Element-wise absolute value. */
public class AbsKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return Math.abs(x);
    }
}
