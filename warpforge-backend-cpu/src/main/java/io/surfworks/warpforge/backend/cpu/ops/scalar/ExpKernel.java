package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/** stablehlo.exponential - Element-wise exponential. */
public class ExpKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) Math.exp(x);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        return v.lanewise(VectorOperators.EXP);
    }
}
