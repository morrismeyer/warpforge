package io.surfworks.warpforge.backend.cpu.ops.controlflow;

import java.util.ArrayList;
import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CaseOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.IfOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReturnOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.WhileOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * Stub kernels for control flow operations.
 *
 * <p>These operations require an interpreter context to execute properly.
 * These kernels provide basic pass-through behavior for simple cases,
 * but full execution requires the StableHLO interpreter.
 */
public final class ControlFlowKernels {

    private ControlFlowKernels() {}

    /**
     * CPU kernel for stablehlo.return.
     * Returns the input tensors as outputs (region terminator).
     */
    public static class ReturnKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            // Return operation simply passes through its inputs
            List<Tensor> outputs = new ArrayList<>(inputs.size());
            for (Tensor input : inputs) {
                outputs.add(input.copy());
            }
            return outputs;
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.ReturnOp;
        }
    }

    /**
     * CPU kernel for stablehlo.if.
     *
     * <p>Note: Full if-else execution requires an interpreter that can evaluate
     * the predicate and execute the appropriate branch. This kernel throws
     * an error to indicate that an interpreter is needed.
     */
    public static class IfKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            throw new UnsupportedOperationException(
                "stablehlo.if requires an interpreter to evaluate branches. " +
                "Use StableHloInterpreter for conditional execution.");
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.IfOp;
        }
    }

    /**
     * CPU kernel for stablehlo.while.
     *
     * <p>Note: Full while-loop execution requires an interpreter that can
     * repeatedly evaluate the condition and body. This kernel throws
     * an error to indicate that an interpreter is needed.
     */
    public static class WhileKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            throw new UnsupportedOperationException(
                "stablehlo.while requires an interpreter to execute loops. " +
                "Use StableHloInterpreter for loop execution.");
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.WhileOp;
        }
    }

    /**
     * CPU kernel for stablehlo.case.
     *
     * <p>Note: Full case/switch execution requires an interpreter that can
     * evaluate the index and execute the appropriate branch. This kernel
     * throws an error to indicate that an interpreter is needed.
     */
    public static class CaseKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            throw new UnsupportedOperationException(
                "stablehlo.case requires an interpreter to evaluate branches. " +
                "Use StableHloInterpreter for switch/case execution.");
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.CaseOp;
        }
    }
}
