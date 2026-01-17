package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;

/** stablehlo.negate - Element-wise negation. */
public class NegateKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return -x;
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        return v.neg();
    }
}
