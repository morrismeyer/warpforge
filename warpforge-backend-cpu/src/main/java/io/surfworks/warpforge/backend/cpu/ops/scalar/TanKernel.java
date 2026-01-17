package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * stablehlo.tan - Element-wise tangent.
 */
public class TanKernel extends UnaryElementwiseKernel {

    @Override
    protected float apply(float x) {
        return (float) Math.tan(x);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        return v.lanewise(VectorOperators.TAN);
    }
}
