package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.is_finite - Element-wise test for finite values.
 * Returns 1.0 if finite, 0.0 if infinite or NaN.
 */
public class IsFiniteKernel extends UnaryElementwiseKernel {

    @Override
    protected float apply(float x) {
        return Float.isFinite(x) ? 1.0f : 0.0f;
    }
}
