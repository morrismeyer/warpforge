package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.and - Element-wise bitwise AND.
 */
public class AndKernel extends BinaryIntegerKernel {

    @Override
    protected int apply(int a, int b) {
        return a & b;
    }
}
