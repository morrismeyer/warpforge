package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.floor - Element-wise floor (round down).
 */
public class FloorKernel extends UnaryElementwiseKernel {

    @Override
    protected float apply(float x) {
        return (float) Math.floor(x);
    }
}
