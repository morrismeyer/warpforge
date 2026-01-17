package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.or - Element-wise bitwise OR.
 */
public class OrKernel extends BinaryIntegerKernel {

    @Override
    protected int apply(int a, int b) {
        return a | b;
    }
}
