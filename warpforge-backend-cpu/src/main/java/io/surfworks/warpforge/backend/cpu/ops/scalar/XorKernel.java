package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.xor - Element-wise bitwise XOR.
 */
public class XorKernel extends BinaryIntegerKernel {

    @Override
    protected int apply(int a, int b) {
        return a ^ b;
    }
}
