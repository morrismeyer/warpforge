package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.tanh - Element-wise hyperbolic tangent. */
public class TanhKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) Math.tanh(x);
    }
}
