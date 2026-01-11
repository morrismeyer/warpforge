package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.rsqrt - Element-wise reciprocal square root (1/sqrt(x)). */
public class RsqrtKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) (1.0 / Math.sqrt(x));
    }
}
