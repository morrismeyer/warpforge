package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.shift_left - Element-wise left shift.
 */
public class ShiftLeftKernel extends BinaryIntegerKernel {

    @Override
    protected int apply(int a, int b) {
        return a << b;
    }
}
