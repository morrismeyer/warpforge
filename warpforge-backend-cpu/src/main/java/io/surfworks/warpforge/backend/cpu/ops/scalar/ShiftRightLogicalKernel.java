package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.shift_right_logical - Element-wise logical right shift.
 * Zero extension (sign bit not preserved).
 */
public class ShiftRightLogicalKernel extends BinaryIntegerKernel {

    @Override
    protected int apply(int a, int b) {
        return a >>> b;
    }
}
