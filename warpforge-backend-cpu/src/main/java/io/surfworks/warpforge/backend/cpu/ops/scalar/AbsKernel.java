package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;

/** stablehlo.abs - Element-wise absolute value. */
public class AbsKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return Math.abs(x);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        return v.abs();
    }
}
