package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;

/** stablehlo.subtract - Element-wise subtraction. */
public class SubtractKernel extends BinaryElementwiseKernel {
    @Override
    protected float apply(float a, float b) {
        return a - b;
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector a, FloatVector b) {
        return a.sub(b);
    }
}
