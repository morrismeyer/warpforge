package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;
import java.util.function.IntFunction;

/**
 * CUDA kernel for unary integer operations.
 *
 * <p>This kernel handles element-wise unary operations on int32 tensors.
 * Supported operations: Popcnt (population count), Clz (count leading zeros).
 *
 * @see CudaKernels
 */
public final class IntegerUnaryKernel implements CudaOpKernel {

    private final String opName;
    private final Class<? extends StableHloAst.Operation> opClass;
    private final IntFunction<String> ptxGenerator;
    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    /**
     * Create an integer unary kernel.
     *
     * @param opName Operation name (e.g., "popcnt")
     * @param opClass The StableHLO operation class this kernel handles
     * @param ptxGenerator Function to generate PTX for a given salt level
     * @param context CUDA context for execution
     * @param salt Instrumentation level
     */
    public IntegerUnaryKernel(String opName,
                               Class<? extends StableHloAst.Operation> opClass,
                               IntFunction<String> ptxGenerator,
                               CudaContext context,
                               int salt) {
        this.opName = opName;
        this.opClass = opClass;
        this.ptxGenerator = ptxGenerator;
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        String ptxSource = ptxGenerator.apply(salt);
        String moduleName = opName + "_i32_module_salt" + salt;
        String functionName = opName + "_i32";

        module = context.loadModule(moduleName, ptxSource);
        function = context.getFunction(module, functionName);
        initialized = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!opClass.isInstance(op)) {
            throw new IllegalArgumentException(
                "Expected " + opClass.getSimpleName() + ", got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 1) {
            throw new IllegalArgumentException(
                opName + " requires exactly 1 input, got: " + inputs.size());
        }

        ensureInitialized();

        Tensor input = inputs.get(0);
        int n = (int) input.elementCount();
        long byteSize = n * 4L; // 4 bytes per int32

        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long dIn = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dIn, input.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            // PTX parameter order: (in_ptr, out_ptr, n, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dIn, dOut},
                    new int[]{n},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after n
                );
            } else {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dIn, dOut},
                    new int[]{n},
                    new float[]{}
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSize);

            return List.of(output);

        } finally {
            context.free(dIn);
            context.free(dOut);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return opClass.isInstance(op);
    }

    public int getSalt() {
        return salt;
    }

    public String getOpName() {
        return opName;
    }

    // ==================== Factory Methods ====================

    public static IntegerUnaryKernel popcnt(CudaContext context, int salt) {
        return new IntegerUnaryKernel(
            "popcnt",
            StableHloAst.PopcntOp.class,
            CudaKernels::generatePopcntI32,
            context,
            salt
        );
    }

    public static IntegerUnaryKernel clz(CudaContext context, int salt) {
        return new IntegerUnaryKernel(
            "clz",
            StableHloAst.ClzOp.class,
            CudaKernels::generateClzI32,
            context,
            salt
        );
    }
}
