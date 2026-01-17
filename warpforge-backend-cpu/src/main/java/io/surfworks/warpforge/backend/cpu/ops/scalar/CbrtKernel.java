package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * stablehlo.cbrt - Element-wise cube root.
 */
public class CbrtKernel extends UnaryElementwiseKernel {

    @Override
    protected float apply(float x) {
        return (float) Math.cbrt(x);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        return v.lanewise(VectorOperators.CBRT);
    }
}
