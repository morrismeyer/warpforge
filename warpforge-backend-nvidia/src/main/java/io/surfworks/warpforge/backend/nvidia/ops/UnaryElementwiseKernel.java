package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;
import java.util.function.IntFunction;

/**
 * Generic CUDA kernel for unary elementwise operations.
 *
 * <p>This kernel handles element-wise operations on a single float32 tensor.
 * Specific operations (negate, abs, exp, log, sqrt, tanh) are configured
 * via the constructor.
 *
 * @see CudaKernels
 */
public final class UnaryElementwiseKernel implements CudaOpKernel {

    private final String opName;
    private final Class<? extends StableHloAst.Operation> opClass;
    private final IntFunction<String> ptxGenerator;
    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    /**
     * Create a unary elementwise kernel.
     *
     * @param opName Operation name (e.g., "negate")
     * @param opClass The StableHLO operation class this kernel handles
     * @param ptxGenerator Function to generate PTX for a given salt level
     * @param context CUDA context for execution
     * @param salt Instrumentation level
     */
    public UnaryElementwiseKernel(String opName,
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

        if (inputs.size() != 1) {
            throw new IllegalArgumentException(
                opName + " requires exactly 1 input, got: " + inputs.size());
        }

        ensureInitialized();

        Tensor input = inputs.get(0);
        int n = (int) input.elementCount();
        long byteSize = n * 4L;

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

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridSize}, new int[]{blockSize},
                    0,
                    new long[]{dIn, dOut, dTiming},
                    n
                );
            } else {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridSize}, new int[]{blockSize},
                    0,
                    new long[]{dIn, dOut},
                    n
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

    public static UnaryElementwiseKernel negate(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "negate",
            StableHloAst.NegateOp.class,
            CudaKernels::generateNegateF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel abs(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "abs",
            StableHloAst.AbsOp.class,
            CudaKernels::generateAbsF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel exp(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "exp",
            StableHloAst.ExpOp.class,
            CudaKernels::generateExpF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel log(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "log",
            StableHloAst.LogOp.class,
            CudaKernels::generateLogF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel sqrt(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "sqrt",
            StableHloAst.SqrtOp.class,
            CudaKernels::generateSqrtF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel tanh(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "tanh",
            StableHloAst.TanhOp.class,
            CudaKernels::generateTanhF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel rsqrt(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "rsqrt",
            StableHloAst.RsqrtOp.class,
            CudaKernels::generateRsqrtF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel sin(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "sin",
            StableHloAst.SinOp.class,
            CudaKernels::generateSinF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel cos(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "cos",
            StableHloAst.CosOp.class,
            CudaKernels::generateCosF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel ceil(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "ceil",
            StableHloAst.CeilOp.class,
            CudaKernels::generateCeilF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel floor(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "floor",
            StableHloAst.FloorOp.class,
            CudaKernels::generateFloorF32,
            context,
            salt
        );
    }

    public static UnaryElementwiseKernel sign(CudaContext context, int salt) {
        return new UnaryElementwiseKernel(
            "sign",
            StableHloAst.SignOp.class,
            CudaKernels::generateSignF32,
            context,
            salt
        );
    }
}
