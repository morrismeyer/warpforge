package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/** stablehlo.tanh - Element-wise hyperbolic tangent. */
public class TanhKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) Math.tanh(x);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        return v.lanewise(VectorOperators.TANH);
    }
}
