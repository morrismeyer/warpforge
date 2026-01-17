package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.remainder - Element-wise remainder (IEEE 754 semantics).
 */
public class RemainderKernel extends BinaryElementwiseKernel {

    @Override
    protected float apply(float a, float b) {
        return a % b;
    }
}
