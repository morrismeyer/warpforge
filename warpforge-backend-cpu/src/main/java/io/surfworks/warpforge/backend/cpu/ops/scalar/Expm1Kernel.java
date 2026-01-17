package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * stablehlo.exponential_minus_one - Element-wise exp(x) - 1.
 * More accurate than exp(x) - 1 for small values of x.
 */
public class Expm1Kernel extends UnaryElementwiseKernel {

    @Override
    protected float apply(float x) {
        return (float) Math.expm1(x);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        return v.lanewise(VectorOperators.EXPM1);
    }
}
