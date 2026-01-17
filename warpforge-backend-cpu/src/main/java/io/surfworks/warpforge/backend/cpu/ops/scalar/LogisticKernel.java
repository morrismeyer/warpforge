package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** stablehlo.logistic - Element-wise logistic (sigmoid) function: 1/(1+exp(-x)). */
public class LogisticKernel extends UnaryElementwiseKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    @Override
    protected float apply(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        FloatVector negV = v.neg();
        FloatVector expNegV = negV.lanewise(VectorOperators.EXP);
        FloatVector onePlusExp = FloatVector.broadcast(SPECIES, 1.0f).add(expNegV);
        return FloatVector.broadcast(SPECIES, 1.0f).div(onePlusExp);
    }
}
