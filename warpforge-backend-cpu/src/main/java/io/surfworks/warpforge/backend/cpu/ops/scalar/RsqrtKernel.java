package io.surfworks.warpforge.backend.cpu.ops.scalar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** stablehlo.rsqrt - Element-wise reciprocal square root (1/sqrt(x)). */
public class RsqrtKernel extends UnaryElementwiseKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    @Override
    protected float apply(float x) {
        return (float) (1.0 / Math.sqrt(x));
    }

    @Override
    protected boolean supportsVectorization() {
        return true;
    }

    @Override
    protected FloatVector applyVector(FloatVector v) {
        FloatVector sqrtV = v.sqrt();
        return FloatVector.broadcast(SPECIES, 1.0f).div(sqrtV);
    }
}
