package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.log - Element-wise natural logarithm. */
public class LogKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) Math.log(x);
    }
}
