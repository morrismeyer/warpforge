package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * stablehlo.atan2 - Element-wise two-argument arctangent.
 * Returns angle in radians between the positive x-axis and the point (b, a).
 */
public class Atan2Kernel extends BinaryElementwiseKernel {

    @Override
    protected float apply(float a, float b) {
        return (float) Math.atan2(a, b);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector a, FloatVector b) {
        return a.lanewise(VectorOperators.ATAN2, b);
    }
}
