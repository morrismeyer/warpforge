package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/** stablehlo.log - Element-wise natural logarithm. */
public class LogKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) Math.log(x);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        return v.lanewise(VectorOperators.LOG);
    }
}
