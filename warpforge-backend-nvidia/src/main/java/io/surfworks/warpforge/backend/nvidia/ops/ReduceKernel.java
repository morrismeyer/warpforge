package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for parallel reduction operations.
 *
 * <p>Supports add, max, min, and mul reductions.
 *
 * <p>This implementation performs a full reduction (all elements to a single value).
 * Partial reductions over specific dimensions are not yet supported.
 *
 * @see CudaKernels#generateReduceF32
 */
public final class ReduceKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;
    private long moduleAdd;
    private long moduleMax;
    private long moduleMin;
    private long moduleMul;
    private long functionAdd;
    private long functionMax;
    private long functionMin;
    private long functionMul;
    private boolean initializedAdd;
    private boolean initializedMax;
    private boolean initializedMin;
    private boolean initializedMul;

    public ReduceKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensureInitialized(String reducer) {
        switch (reducer) {
            case "add" -> {
                if (!initializedAdd) {
                    String ptx = CudaKernels.generateReduceAddF32(salt);
                    moduleAdd = context.loadModule("reduce_add_f32_salt" + salt, ptx);
                    functionAdd = context.getFunction(moduleAdd, "reduce_add_f32");
                    initializedAdd = true;
                }
            }
            case "max" -> {
                if (!initializedMax) {
                    String ptx = CudaKernels.generateReduceMaxF32(salt);
                    moduleMax = context.loadModule("reduce_max_f32_salt" + salt, ptx);
                    functionMax = context.getFunction(moduleMax, "reduce_max_f32");
                    initializedMax = true;
                }
            }
            case "min" -> {
                if (!initializedMin) {
                    String ptx = CudaKernels.generateReduceMinF32(salt);
                    moduleMin = context.loadModule("reduce_min_f32_salt" + salt, ptx);
                    functionMin = context.getFunction(moduleMin, "reduce_min_f32");
                    initializedMin = true;
                }
            }
            case "mul" -> {
                if (!initializedMul) {
                    String ptx = CudaKernels.generateReduceMulF32(salt);
                    moduleMul = context.loadModule("reduce_mul_f32_salt" + salt, ptx);
                    functionMul = context.getFunction(moduleMul, "reduce_mul_f32");
                    initializedMul = true;
                }
            }
            default -> throw new IllegalArgumentException("Unknown reducer: " + reducer);
        }
    }

    private long getFunction(String reducer) {
        return switch (reducer) {
            case "add" -> functionAdd;
            case "max" -> functionMax;
            case "min" -> functionMin;
            case "mul" -> functionMul;
            default -> throw new IllegalArgumentException("Unknown reducer: " + reducer);
        };
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.ReduceOp reduceOp)) {
            throw new IllegalArgumentException(
                "Expected ReduceOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                "Reduce requires 2 inputs (operand and init), got: " + inputs.size());
        }

        String reducer = reduceOp.reducer();
        ensureInitialized(reducer);

        Tensor operand = inputs.get(0);
        // inputs.get(1) is the init value, which is baked into the PTX kernel

        int n = (int) operand.elementCount();
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        // For full reduction, output is a scalar (single element)
        int outputElements = (int) outputSpec.elementCount();
        if (outputElements != 1) {
            throw new UnsupportedOperationException(
                "Only full reduction to scalar is currently supported, got output size: " + outputElements);
        }

        long byteSize = n * 4L;
        long outputByteSize = 4L;

        long dInput = context.allocate(byteSize);
        long dOutput = context.allocate(outputByteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dInput, operand.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            // Single block reduction for simplicity
            // For large inputs, we'd need multi-pass reduction
            int blockSize = Math.min(CudaKernels.REDUCE_BLOCK_SIZE, n);
            // Round up to next power of 2 for shared memory efficiency
            blockSize = nextPowerOf2(blockSize);
            int sharedMemSize = blockSize * 4; // float per thread

            int gridSize = 1; // Single block for full reduction

            long function = getFunction(reducer);

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    sharedMemSize,
                    new long[]{dInput, dOutput, dTiming},
                    n
                );
            } else {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    sharedMemSize,
                    new long[]{dInput, dOutput},
                    n
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, outputByteSize);

            return List.of(output);

        } finally {
            context.free(dInput);
            context.free(dOutput);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    private static int nextPowerOf2(int n) {
        int power = 1;
        while (power < n && power < CudaKernels.REDUCE_BLOCK_SIZE) {
            power *= 2;
        }
        return power;
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.ReduceOp;
    }

    public int getSalt() {
        return salt;
    }
}
