package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.sqrt - Element-wise square root. */
public class SqrtKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) Math.sqrt(x);
    }
}
