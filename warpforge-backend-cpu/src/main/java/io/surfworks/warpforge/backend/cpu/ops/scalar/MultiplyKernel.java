package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.multiply - Element-wise multiplication. */
public class MultiplyKernel extends BinaryElementwiseKernel {
    @Override
    protected float apply(float a, float b) {
        return a * b;
    }
}
