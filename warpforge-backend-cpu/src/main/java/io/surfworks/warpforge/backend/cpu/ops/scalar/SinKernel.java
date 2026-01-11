package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.sine - Element-wise sine. */
public class SinKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) Math.sin(x);
    }
}
