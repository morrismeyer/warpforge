package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.subtract - Element-wise subtraction. */
public class SubtractKernel extends BinaryElementwiseKernel {
    @Override
    protected float apply(float a, float b) {
        return a - b;
    }
}
