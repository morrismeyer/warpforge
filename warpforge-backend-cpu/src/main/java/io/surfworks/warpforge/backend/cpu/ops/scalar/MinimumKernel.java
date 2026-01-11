package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.minimum - Element-wise minimum. */
public class MinimumKernel extends BinaryElementwiseKernel {
    @Override
    protected float apply(float a, float b) {
        return Math.min(a, b);
    }
}
