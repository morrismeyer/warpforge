package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.maximum - Element-wise maximum. */
public class MaximumKernel extends BinaryElementwiseKernel {
    @Override
    protected float apply(float a, float b) {
        return Math.max(a, b);
    }
}
