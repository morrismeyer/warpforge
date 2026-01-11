package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.divide - Element-wise division. */
public class DivideKernel extends BinaryElementwiseKernel {
    @Override
    protected float apply(float a, float b) {
        return a / b;
    }
}
