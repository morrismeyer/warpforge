package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.ceil - Element-wise ceiling (round up).
 */
public class CeilKernel extends UnaryElementwiseKernel {

    @Override
    protected float apply(float x) {
        return (float) Math.ceil(x);
    }
}
