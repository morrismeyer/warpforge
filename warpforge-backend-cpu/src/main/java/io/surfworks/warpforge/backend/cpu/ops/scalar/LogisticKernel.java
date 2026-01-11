package io.surfworks.warpforge.backend.cpu.ops.scalar;

/** stablehlo.logistic - Element-wise logistic (sigmoid) function: 1/(1+exp(-x)). */
public class LogisticKernel extends UnaryElementwiseKernel {
    @Override
    protected float apply(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }
}
