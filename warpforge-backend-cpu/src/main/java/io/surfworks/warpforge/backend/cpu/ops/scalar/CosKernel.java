package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.cosine - Element-wise cosine. */
public class CosKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) Math.cos(x);
    }
}
