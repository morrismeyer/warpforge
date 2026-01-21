package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;
import java.util.function.IntFunction;

/**
 * CUDA kernel for integer bitwise operations.
 *
 * <p>This kernel handles element-wise bitwise operations on two int32 tensors.
 * Supported operations: And, Or, Xor.
 *
 * @see CudaKernels
 */
public final class IntegerBitwiseKernel implements CudaOpKernel {

    private final String opName;
    private final Class<? extends StableHloAst.Operation> opClass;
    private final IntFunction<String> ptxGenerator;
    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    /**
     * Create an integer bitwise kernel.
     *
     * @param opName Operation name (e.g., "and")
     * @param opClass The StableHLO operation class this kernel handles
     * @param ptxGenerator Function to generate PTX for a given salt level
     * @param context CUDA context for execution
     * @param salt Instrumentation level
     */
    public IntegerBitwiseKernel(String opName,
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

        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                opName + " requires exactly 2 inputs, got: " + inputs.size());
        }

        ensureInitialized();

        Tensor lhs = inputs.get(0);
        Tensor rhs = inputs.get(1);

        if (lhs.elementCount() != rhs.elementCount()) {
            throw new IllegalArgumentException(
                "Input tensors must have same element count: " +
                lhs.elementCount() + " vs " + rhs.elementCount());
        }

        int n = (int) lhs.elementCount();
        long byteSize = n * 4L; // 4 bytes per int32

        // Output spec from AST, but ensure it's I32
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        long dA = context.allocate(byteSize);
        long dB = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dA, lhs.data());
            context.copyToDevice(dB, rhs.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridSize}, new int[]{blockSize},
                    0,
                    new long[]{dA, dB, dOut, dTiming},
                    n
                );
            } else {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridSize}, new int[]{blockSize},
                    0,
                    new long[]{dA, dB, dOut},
                    n
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSize);

            return List.of(output);

        } finally {
            context.free(dA);
            context.free(dB);
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

    public static IntegerBitwiseKernel and(CudaContext context, int salt) {
        return new IntegerBitwiseKernel(
            "and",
            StableHloAst.AndOp.class,
            CudaKernels::generateAndI32,
            context,
            salt
        );
    }

    public static IntegerBitwiseKernel or(CudaContext context, int salt) {
        return new IntegerBitwiseKernel(
            "or",
            StableHloAst.OrOp.class,
            CudaKernels::generateOrI32,
            context,
            salt
        );
    }

    public static IntegerBitwiseKernel xor(CudaContext context, int salt) {
        return new IntegerBitwiseKernel(
            "xor",
            StableHloAst.XorOp.class,
            CudaKernels::generateXorI32,
            context,
            salt
        );
    }
}
