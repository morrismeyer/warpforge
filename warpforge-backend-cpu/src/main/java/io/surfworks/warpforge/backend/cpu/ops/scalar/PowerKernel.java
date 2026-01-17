package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * stablehlo.power - Element-wise exponentiation (a^b).
 */
public class PowerKernel extends BinaryElementwiseKernel {

    @Override
    protected float apply(float a, float b) {
        return (float) Math.pow(a, b);
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector a, FloatVector b) {
        return a.lanewise(VectorOperators.POW, b);
    }
}
