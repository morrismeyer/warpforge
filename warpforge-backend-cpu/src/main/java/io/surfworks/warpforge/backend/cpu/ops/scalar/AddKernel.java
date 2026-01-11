package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.add - Element-wise addition. */
public class AddKernel extends BinaryElementwiseKernel {
    @Override
    protected float apply(float a, float b) {
        return a + b;
    }
}
