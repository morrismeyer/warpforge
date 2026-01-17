package io.surfworks.warpforge.backend.cpu.ops.scalar;

/**
 * stablehlo.shift_right_arithmetic - Element-wise arithmetic right shift.
 * Sign bit is preserved (sign extension).
 */
public class ShiftRightArithmeticKernel extends BinaryIntegerKernel {

    @Override
    protected int apply(int a, int b) {
        return a >> b;
    }
}
