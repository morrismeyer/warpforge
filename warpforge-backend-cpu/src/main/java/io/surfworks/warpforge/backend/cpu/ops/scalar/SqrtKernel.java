package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;

/** stablehlo.sqrt - Element-wise square root. */
public class SqrtKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) Math.sqrt(x);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        return v.sqrt();
    }
}
