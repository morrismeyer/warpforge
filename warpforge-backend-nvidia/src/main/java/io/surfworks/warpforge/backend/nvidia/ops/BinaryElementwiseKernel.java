package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;
import java.util.function.IntFunction;

/**
 * Generic CUDA kernel for binary elementwise operations.
 *
 * <p>This kernel handles element-wise operations on two float32 tensors.
 * Specific operations (add, subtract, multiply, divide, maximum, minimum)
 * are configured via the constructor.
 *
 * @see CudaKernels
 */
public final class BinaryElementwiseKernel implements CudaOpKernel {

    private final String opName;
    private final Class<? extends StableHloAst.Operation> opClass;
    private final IntFunction<String> ptxGenerator;
    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    /**
     * Create a binary elementwise kernel.
     *
     * @param opName Operation name (e.g., "subtract")
     * @param opClass The StableHLO operation class this kernel handles
     * @param ptxGenerator Function to generate PTX for a given salt level
     * @param context CUDA context for execution
     * @param salt Instrumentation level
     */
    public BinaryElementwiseKernel(String opName,
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
        String moduleName = opName + "_f32_module_salt" + salt;
        String functionName = opName + "_f32";

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
        long byteSize = n * 4L;

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

    public static BinaryElementwiseKernel subtract(CudaContext context, int salt) {
        return new BinaryElementwiseKernel(
            "subtract",
            StableHloAst.SubtractOp.class,
            CudaKernels::generateSubtractF32,
            context,
            salt
        );
    }

    public static BinaryElementwiseKernel divide(CudaContext context, int salt) {
        return new BinaryElementwiseKernel(
            "divide",
            StableHloAst.DivideOp.class,
            CudaKernels::generateDivideF32,
            context,
            salt
        );
    }

    public static BinaryElementwiseKernel maximum(CudaContext context, int salt) {
        return new BinaryElementwiseKernel(
            "maximum",
            StableHloAst.MaximumOp.class,
            CudaKernels::generateMaximumF32,
            context,
            salt
        );
    }

    public static BinaryElementwiseKernel minimum(CudaContext context, int salt) {
        return new BinaryElementwiseKernel(
            "minimum",
            StableHloAst.MinimumOp.class,
            CudaKernels::generateMinimumF32,
            context,
            salt
        );
    }

    public static BinaryElementwiseKernel power(CudaContext context, int salt) {
        return new BinaryElementwiseKernel(
            "power",
            StableHloAst.PowerOp.class,
            CudaKernels::generatePowerF32,
            context,
            salt
        );
    }

    public static BinaryElementwiseKernel remainder(CudaContext context, int salt) {
        return new BinaryElementwiseKernel(
            "remainder",
            StableHloAst.RemainderOp.class,
            CudaKernels::generateRemainderF32,
            context,
            salt
        );
    }
}
