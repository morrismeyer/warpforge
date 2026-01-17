package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * stablehlo.log_plus_one - Element-wise ln(1 + x).
 * More accurate than log(1 + x) for small values of x.
 */
public class Log1pKernel extends UnaryElementwiseKernel {

    @Override
    protected float apply(float x) {
        return (float) Math.log1p(x);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        return v.lanewise(VectorOperators.LOG1P);
    }
}
