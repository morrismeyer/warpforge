package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Dispatches StableHLO operations to CUDA kernel implementations.
 *
 * <p>Operations with real CUDA implementations use custom PTX kernels with
 * salt-based instrumentation. Operations without implementations yet throw
 * UnsupportedOperationException.
 *
 * @see CudaKernels for PTX generation with instrumentation
 */
public final class CudaOpDispatcher {

    private final Map<Class<? extends StableHloAst.Operation>, CudaOpKernel> kernels = new ConcurrentHashMap<>();
    private final CudaContext context;
    private final int salt;

    /**
     * Create a dispatcher without CUDA context (stub-only mode for testing).
     */
    public CudaOpDispatcher() {
        this(null, CudaKernels.SALT_NONE);
    }

    /**
     * Create a dispatcher with CUDA context and no instrumentation.
     */
    public CudaOpDispatcher(CudaContext context) {
        this(context, CudaKernels.SALT_NONE);
    }

    /**
     * Create a dispatcher with CUDA context and specified instrumentation level.
     *
     * @param context CUDA context for kernel execution (null for stub-only mode)
     * @param salt Instrumentation level (SALT_NONE, SALT_TIMING, SALT_TRACE)
     */
    public CudaOpDispatcher(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
        registerAllOperations();
    }

    public List<Tensor> dispatch(StableHloAst.Operation op, List<Tensor> inputs) {
        CudaOpKernel kernel = kernels.get(op.getClass());
        if (kernel == null) {
            throw new UnsupportedOperationException(
                "No CUDA kernel registered for operation: " + op.opName());
        }
        return kernel.execute(op, inputs);
    }

    public boolean supports(StableHloAst.Operation op) {
        return kernels.containsKey(op.getClass());
    }

    public List<String> supportedOps() {
        return kernels.keySet().stream()
            .map(Class::getSimpleName)
            .map(s -> s.replace("Op", ""))
            .sorted()
            .toList();
    }

    /**
     * Register all StableHLO operations.
     * Operations with real implementations get CUDA kernels.
     * Others get stubs that throw UnsupportedOperationException.
     */
    private void registerAllOperations() {
        // ==================== Implemented Operations ====================

        // Binary elementwise - Add (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.AddOp.class, new AddKernel(context, salt));
        } else {
            registerStub(StableHloAst.AddOp.class);
        }

        // Binary elementwise - Multiply (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.MultiplyOp.class, new MultiplyKernel(context, salt));
        } else {
            registerStub(StableHloAst.MultiplyOp.class);
        }

        // Binary elementwise - Subtract (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.SubtractOp.class, BinaryElementwiseKernel.subtract(context, salt));
        } else {
            registerStub(StableHloAst.SubtractOp.class);
        }

        // Binary elementwise - Divide (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.DivideOp.class, BinaryElementwiseKernel.divide(context, salt));
        } else {
            registerStub(StableHloAst.DivideOp.class);
        }

        // Binary elementwise - Maximum (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.MaximumOp.class, BinaryElementwiseKernel.maximum(context, salt));
        } else {
            registerStub(StableHloAst.MaximumOp.class);
        }

        // Binary elementwise - Minimum (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.MinimumOp.class, BinaryElementwiseKernel.minimum(context, salt));
        } else {
            registerStub(StableHloAst.MinimumOp.class);
        }

        // ==================== Unary Elementwise Operations (IMPLEMENTED) ====================

        // Unary elementwise - Negate (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.NegateOp.class, UnaryElementwiseKernel.negate(context, salt));
        } else {
            registerStub(StableHloAst.NegateOp.class);
        }

        // Unary elementwise - Abs (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.AbsOp.class, UnaryElementwiseKernel.abs(context, salt));
        } else {
            registerStub(StableHloAst.AbsOp.class);
        }

        // Unary elementwise - Exp (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.ExpOp.class, UnaryElementwiseKernel.exp(context, salt));
        } else {
            registerStub(StableHloAst.ExpOp.class);
        }

        // Unary elementwise - Log (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.LogOp.class, UnaryElementwiseKernel.log(context, salt));
        } else {
            registerStub(StableHloAst.LogOp.class);
        }

        // Unary elementwise - Sqrt (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.SqrtOp.class, UnaryElementwiseKernel.sqrt(context, salt));
        } else {
            registerStub(StableHloAst.SqrtOp.class);
        }

        // Unary elementwise - Tanh (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.TanhOp.class, UnaryElementwiseKernel.tanh(context, salt));
        } else {
            registerStub(StableHloAst.TanhOp.class);
        }

        // Unary elementwise - Rsqrt (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.RsqrtOp.class, UnaryElementwiseKernel.rsqrt(context, salt));
        } else {
            registerStub(StableHloAst.RsqrtOp.class);
        }

        // Unary elementwise - Sin (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.SinOp.class, UnaryElementwiseKernel.sin(context, salt));
        } else {
            registerStub(StableHloAst.SinOp.class);
        }

        // Unary elementwise - Cos (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.CosOp.class, UnaryElementwiseKernel.cos(context, salt));
        } else {
            registerStub(StableHloAst.CosOp.class);
        }

        // Unary elementwise - Ceil (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.CeilOp.class, UnaryElementwiseKernel.ceil(context, salt));
        } else {
            registerStub(StableHloAst.CeilOp.class);
        }

        // Unary elementwise - Floor (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.FloorOp.class, UnaryElementwiseKernel.floor(context, salt));
        } else {
            registerStub(StableHloAst.FloorOp.class);
        }

        // Unary elementwise - Sign (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.SignOp.class, UnaryElementwiseKernel.sign(context, salt));
        } else {
            registerStub(StableHloAst.SignOp.class);
        }

        // Unary elementwise - Tan (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.TanOp.class, UnaryElementwiseKernel.tan(context, salt));
        } else {
            registerStub(StableHloAst.TanOp.class);
        }

        // Unary elementwise - Logistic (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.LogisticOp.class, UnaryElementwiseKernel.logistic(context, salt));
        } else {
            registerStub(StableHloAst.LogisticOp.class);
        }

        // Unary elementwise - Expm1 (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.Expm1Op.class, UnaryElementwiseKernel.expm1(context, salt));
        } else {
            registerStub(StableHloAst.Expm1Op.class);
        }

        // Unary elementwise - Log1p (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.Log1pOp.class, UnaryElementwiseKernel.log1p(context, salt));
        } else {
            registerStub(StableHloAst.Log1pOp.class);
        }

        // Unary elementwise - Cbrt (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.CbrtOp.class, UnaryElementwiseKernel.cbrt(context, salt));
        } else {
            registerStub(StableHloAst.CbrtOp.class);
        }

        // Unary elementwise - IsFinite (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.IsFiniteOp.class, UnaryElementwiseKernel.isFinite(context, salt));
        } else {
            registerStub(StableHloAst.IsFiniteOp.class);
        }

        // Unary elementwise - RoundNearestEven (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.RoundNearestEvenOp.class, UnaryElementwiseKernel.roundNearestEven(context, salt));
        } else {
            registerStub(StableHloAst.RoundNearestEvenOp.class);
        }

        // Unary elementwise - RoundNearestAfz (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.RoundNearestAfzOp.class, UnaryElementwiseKernel.roundNearestAfz(context, salt));
        } else {
            registerStub(StableHloAst.RoundNearestAfzOp.class);
        }

        // Binary elementwise - Power (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.PowerOp.class, BinaryElementwiseKernel.power(context, salt));
        } else {
            registerStub(StableHloAst.PowerOp.class);
        }

        // Binary elementwise - Remainder (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.RemainderOp.class, BinaryElementwiseKernel.remainder(context, salt));
        } else {
            registerStub(StableHloAst.RemainderOp.class);
        }

        // ==================== Comparison and Selection Operations (IMPLEMENTED) ====================

        // Compare (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.CompareOp.class, new CompareKernel(context, salt));
        } else {
            registerStub(StableHloAst.CompareOp.class);
        }

        // Select (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.SelectOp.class, new SelectKernel(context, salt));
        } else {
            registerStub(StableHloAst.SelectOp.class);
        }

        // Clamp (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.ClampOp.class, new ClampKernel(context, salt));
        } else {
            registerStub(StableHloAst.ClampOp.class);
        }

        // ==================== Additional Elementwise Operations (IMPLEMENTED) ====================

        // Binary elementwise - Atan2 (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.Atan2Op.class, BinaryElementwiseKernel.atan2(context, salt));
        } else {
            registerStub(StableHloAst.Atan2Op.class);
        }

        // Unary elementwise - Not (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.NotOp.class, UnaryElementwiseKernel.not(context, salt));
        } else {
            registerStub(StableHloAst.NotOp.class);
        }

        // ==================== Integer Bitwise Operations (IMPLEMENTED) ====================

        // Integer bitwise - And (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.AndOp.class, IntegerBitwiseKernel.and(context, salt));
        } else {
            registerStub(StableHloAst.AndOp.class);
        }

        // Integer bitwise - Or (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.OrOp.class, IntegerBitwiseKernel.or(context, salt));
        } else {
            registerStub(StableHloAst.OrOp.class);
        }

        // Integer bitwise - Xor (IMPLEMENTED)
        if (context != null) {
            kernels.put(StableHloAst.XorOp.class, IntegerBitwiseKernel.xor(context, salt));
        } else {
            registerStub(StableHloAst.XorOp.class);
        }

        // ==================== Stub Operations ====================

        // Binary elementwise operations (stubs - shifts, not yet implemented)
        registerStub(StableHloAst.ShiftLeftOp.class);
        registerStub(StableHloAst.ShiftRightArithmeticOp.class);
        registerStub(StableHloAst.ShiftRightLogicalOp.class);

        // Unary elementwise operations (stubs - integer-specific)
        registerStub(StableHloAst.PopcntOp.class);
        registerStub(StableHloAst.ClzOp.class);

        // Constants
        registerStub(StableHloAst.ConstantOp.class);

        // Shape manipulation
        registerStub(StableHloAst.ReshapeOp.class);
        registerStub(StableHloAst.TransposeOp.class);
        registerStub(StableHloAst.BroadcastInDimOp.class);
        registerStub(StableHloAst.ConcatenateOp.class);
        registerStub(StableHloAst.SliceOp.class);
        registerStub(StableHloAst.ReverseOp.class);
        registerStub(StableHloAst.PadOp.class);
        registerStub(StableHloAst.IotaOp.class);
        registerStub(StableHloAst.GatherOp.class);
        registerStub(StableHloAst.ScatterOp.class);
        registerStub(StableHloAst.DynamicSliceOp.class);
        registerStub(StableHloAst.DynamicUpdateSliceOp.class);
        registerStub(StableHloAst.GetDimensionSizeOp.class);

        // Dynamic shape operations
        registerStub(StableHloAst.DynamicBroadcastInDimOp.class);
        registerStub(StableHloAst.DynamicGatherOp.class);
        registerStub(StableHloAst.DynamicIotaOp.class);
        registerStub(StableHloAst.DynamicPadOp.class);
        registerStub(StableHloAst.DynamicReshapeOp.class);
        registerStub(StableHloAst.DynamicConvOp.class);

        // Type conversion
        registerStub(StableHloAst.ConvertOp.class);
        registerStub(StableHloAst.BitcastConvertOp.class);

        // Quantization
        registerStub(StableHloAst.UniformQuantizeOp.class);
        registerStub(StableHloAst.UniformDequantizeOp.class);

        // Reduction
        registerStub(StableHloAst.ReduceOp.class);
        registerStub(StableHloAst.ReduceWindowOp.class);
        registerStub(StableHloAst.ReducePrecisionOp.class);
        registerStub(StableHloAst.SelectAndScatterOp.class);

        // Linear algebra
        registerStub(StableHloAst.DotOp.class);
        registerStub(StableHloAst.DotGeneralOp.class);
        registerStub(StableHloAst.CholeskyOp.class);
        registerStub(StableHloAst.TriangularSolveOp.class);

        // Convolution and neural network
        registerStub(StableHloAst.ConvolutionOp.class);
        registerStub(StableHloAst.BatchNormTrainingOp.class);
        registerStub(StableHloAst.BatchNormInferenceOp.class);
        registerStub(StableHloAst.BatchNormGradOp.class);

        // Control flow
        registerStub(StableHloAst.IfOp.class);
        registerStub(StableHloAst.WhileOp.class);
        registerStub(StableHloAst.CaseOp.class);
        registerStub(StableHloAst.MapOp.class);

        // Collective operations
        registerStub(StableHloAst.AfterAllOp.class);
        registerStub(StableHloAst.AllGatherOp.class);
        registerStub(StableHloAst.AllReduceOp.class);
        registerStub(StableHloAst.AllToAllOp.class);
        registerStub(StableHloAst.CollectiveBroadcastOp.class);
        registerStub(StableHloAst.CollectivePermuteOp.class);
        registerStub(StableHloAst.ReduceScatterOp.class);
        registerStub(StableHloAst.PartitionIdOp.class);
        registerStub(StableHloAst.ReplicaIdOp.class);

        // Communication
        registerStub(StableHloAst.InfeedOp.class);
        registerStub(StableHloAst.OutfeedOp.class);
        registerStub(StableHloAst.RecvOp.class);
        registerStub(StableHloAst.SendOp.class);

        // Tuple operations
        registerStub(StableHloAst.TupleOp.class);
        registerStub(StableHloAst.GetTupleElementOp.class);

        // Complex number operations
        registerStub(StableHloAst.RealOp.class);
        registerStub(StableHloAst.ImagOp.class);
        registerStub(StableHloAst.ComplexOp.class);

        // Signal processing
        registerStub(StableHloAst.FftOp.class);

        // Random number generation
        registerStub(StableHloAst.RngOp.class);
        registerStub(StableHloAst.RngBitGeneratorOp.class);

        // Other operations
        registerStub(StableHloAst.SortOp.class);
        registerStub(StableHloAst.OptimizationBarrierOp.class);
        registerStub(StableHloAst.CompositeOp.class);
        registerStub(StableHloAst.CustomCallOp.class);
        registerStub(StableHloAst.ReturnOp.class);
    }

    private <T extends StableHloAst.Operation> void registerStub(Class<T> opClass) {
        kernels.put(opClass, new CudaStubKernel(opClass.getSimpleName()));
    }

    /**
     * Stub kernel that throws UnsupportedOperationException.
     * Replace with real CUDA implementations.
     */
    private static class CudaStubKernel implements CudaOpKernel {
        private final String opName;

        CudaStubKernel(String opName) {
            this.opName = opName;
        }

        @Override
        public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
            throw new UnsupportedOperationException(
                "CUDA kernel not yet implemented for: " + opName +
                ". Use CPU backend for reference implementation.");
        }
    }
}
