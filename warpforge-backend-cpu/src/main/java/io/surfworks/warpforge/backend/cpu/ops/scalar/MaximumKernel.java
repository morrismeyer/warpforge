package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;

/** stablehlo.maximum - Element-wise maximum. */
public class MaximumKernel extends BinaryElementwiseKernel {
    @Override
    protected float apply(float a, float b) {
        return Math.max(a, b);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector a, FloatVector b) {
        return a.max(b);
    }
}
