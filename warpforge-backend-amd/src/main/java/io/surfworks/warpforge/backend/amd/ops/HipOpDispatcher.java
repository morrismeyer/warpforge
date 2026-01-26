package io.surfworks.warpforge.backend.amd.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.amd.hip.HipContext;
import io.surfworks.warpforge.backend.amd.hip.HipKernels;
import io.surfworks.warpforge.backend.amd.rocblas.RocblasRuntime;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Dispatches StableHLO operations to HIP kernel implementations.
 *
 * <p>This dispatcher follows the three-tier kernel architecture:
 * <ul>
 *   <li><b>PRODUCTION</b>: rocBLAS for matrix operations (DotOp) - fully implemented</li>
 *   <li><b>OPTIMIZED_OBSERVABLE</b>: HIP kernels with salt instrumentation - requires HIPRTC</li>
 *   <li><b>CORRECTNESS</b>: Naive HIP kernels with full tracing - requires HIPRTC</li>
 * </ul>
 *
 * <p>Operations without rocBLAS support (elementwise, etc.) currently throw
 * UnsupportedOperationException until HIPRTC integration is complete.
 */
public final class HipOpDispatcher {

    private final Map<Class<? extends StableHloAst.Operation>, HipOpKernel> kernels = new ConcurrentHashMap<>();
    private final HipContext context;
    private final int salt;
    private final boolean useRocblas;

    /**
     * Create a dispatcher without a HIP context (stub mode for testing).
     */
    public HipOpDispatcher() {
        this(null, HipKernels.SALT_NONE);
    }

    /**
     * Create a dispatcher with a HIP context.
     *
     * @param context HIP context for kernel execution (null for stub mode)
     * @param salt Instrumentation level for custom kernels
     */
    public HipOpDispatcher(HipContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.useRocblas = context != null && RocblasRuntime.isAvailable();
        registerAllOperations();
    }

    public List<Tensor> dispatch(StableHloAst.Operation op, List<Tensor> inputs) {
        HipOpKernel kernel = kernels.get(op.getClass());
        if (kernel == null) {
            throw new UnsupportedOperationException(
                "No HIP kernel registered for operation: " + op.opName());
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
     * Check if a specific operation has a real (non-stub) implementation.
     */
    public boolean hasRealImplementation(Class<? extends StableHloAst.Operation> opClass) {
        HipOpKernel kernel = kernels.get(opClass);
        return kernel != null && !(kernel instanceof HipStubKernel);
    }

    /**
     * Register all StableHLO operations.
     */
    private void registerAllOperations() {
        // ==================== PRODUCTION Tier (rocBLAS) ====================
        // These are fully functional when rocBLAS is available

        if (useRocblas) {
            kernels.put(StableHloAst.DotOp.class, new RocblasDotKernel(context));
        } else {
            registerStub(StableHloAst.DotOp.class, "DotOp requires rocBLAS or HIPRTC");
        }

        // ==================== Requires HIPRTC (stubs for now) ====================
        // These have HIP C++ source ready but need HIPRTC to compile

        // Binary elementwise operations
        registerHiprtcStub(StableHloAst.AddOp.class, "AddOp");
        registerHiprtcStub(StableHloAst.SubtractOp.class, "SubtractOp");
        registerHiprtcStub(StableHloAst.MultiplyOp.class, "MultiplyOp");
        registerHiprtcStub(StableHloAst.DivideOp.class, "DivideOp");
        registerHiprtcStub(StableHloAst.MaximumOp.class, "MaximumOp");
        registerHiprtcStub(StableHloAst.MinimumOp.class, "MinimumOp");
        registerHiprtcStub(StableHloAst.PowerOp.class, "PowerOp");
        registerHiprtcStub(StableHloAst.RemainderOp.class, "RemainderOp");
        registerHiprtcStub(StableHloAst.Atan2Op.class, "Atan2Op");
        registerHiprtcStub(StableHloAst.AndOp.class, "AndOp");
        registerHiprtcStub(StableHloAst.OrOp.class, "OrOp");
        registerHiprtcStub(StableHloAst.XorOp.class, "XorOp");
        registerHiprtcStub(StableHloAst.ShiftLeftOp.class, "ShiftLeftOp");
        registerHiprtcStub(StableHloAst.ShiftRightArithmeticOp.class, "ShiftRightArithmeticOp");
        registerHiprtcStub(StableHloAst.ShiftRightLogicalOp.class, "ShiftRightLogicalOp");

        // Unary elementwise operations
        registerHiprtcStub(StableHloAst.NegateOp.class, "NegateOp");
        registerHiprtcStub(StableHloAst.AbsOp.class, "AbsOp");
        registerHiprtcStub(StableHloAst.ExpOp.class, "ExpOp");
        registerHiprtcStub(StableHloAst.LogOp.class, "LogOp");
        registerHiprtcStub(StableHloAst.TanhOp.class, "TanhOp");
        registerHiprtcStub(StableHloAst.SqrtOp.class, "SqrtOp");
        registerHiprtcStub(StableHloAst.RsqrtOp.class, "RsqrtOp");
        registerHiprtcStub(StableHloAst.SinOp.class, "SinOp");
        registerHiprtcStub(StableHloAst.CosOp.class, "CosOp");
        registerHiprtcStub(StableHloAst.TanOp.class, "TanOp");
        registerHiprtcStub(StableHloAst.LogisticOp.class, "LogisticOp");
        registerHiprtcStub(StableHloAst.CeilOp.class, "CeilOp");
        registerHiprtcStub(StableHloAst.FloorOp.class, "FloorOp");
        registerHiprtcStub(StableHloAst.SignOp.class, "SignOp");
        registerHiprtcStub(StableHloAst.Expm1Op.class, "Expm1Op");
        registerHiprtcStub(StableHloAst.Log1pOp.class, "Log1pOp");
        registerHiprtcStub(StableHloAst.CbrtOp.class, "CbrtOp");
        registerHiprtcStub(StableHloAst.IsFiniteOp.class, "IsFiniteOp");
        registerHiprtcStub(StableHloAst.RoundNearestEvenOp.class, "RoundNearestEvenOp");
        registerHiprtcStub(StableHloAst.RoundNearestAfzOp.class, "RoundNearestAfzOp");
        registerHiprtcStub(StableHloAst.NotOp.class, "NotOp");
        registerHiprtcStub(StableHloAst.PopcntOp.class, "PopcntOp");
        registerHiprtcStub(StableHloAst.ClzOp.class, "ClzOp");

        // Comparison and selection
        registerHiprtcStub(StableHloAst.CompareOp.class, "CompareOp");
        registerHiprtcStub(StableHloAst.SelectOp.class, "SelectOp");
        registerHiprtcStub(StableHloAst.ClampOp.class, "ClampOp");

        // Constants
        registerStub(StableHloAst.ConstantOp.class, "ConstantOp");

        // Shape manipulation
        registerHiprtcStub(StableHloAst.ReshapeOp.class, "ReshapeOp");
        registerHiprtcStub(StableHloAst.TransposeOp.class, "TransposeOp");
        registerHiprtcStub(StableHloAst.BroadcastInDimOp.class, "BroadcastInDimOp");
        registerHiprtcStub(StableHloAst.ConcatenateOp.class, "ConcatenateOp");
        registerHiprtcStub(StableHloAst.SliceOp.class, "SliceOp");
        registerHiprtcStub(StableHloAst.ReverseOp.class, "ReverseOp");
        registerHiprtcStub(StableHloAst.PadOp.class, "PadOp");
        registerHiprtcStub(StableHloAst.IotaOp.class, "IotaOp");
        registerHiprtcStub(StableHloAst.GatherOp.class, "GatherOp");
        registerHiprtcStub(StableHloAst.ScatterOp.class, "ScatterOp");
        registerStub(StableHloAst.DynamicSliceOp.class, "DynamicSliceOp");
        registerStub(StableHloAst.DynamicUpdateSliceOp.class, "DynamicUpdateSliceOp");
        registerStub(StableHloAst.GetDimensionSizeOp.class, "GetDimensionSizeOp");

        // Dynamic shape operations
        registerStub(StableHloAst.DynamicBroadcastInDimOp.class, "DynamicBroadcastInDimOp");
        registerStub(StableHloAst.DynamicGatherOp.class, "DynamicGatherOp");
        registerStub(StableHloAst.DynamicIotaOp.class, "DynamicIotaOp");
        registerStub(StableHloAst.DynamicPadOp.class, "DynamicPadOp");
        registerStub(StableHloAst.DynamicReshapeOp.class, "DynamicReshapeOp");
        registerStub(StableHloAst.DynamicConvOp.class, "DynamicConvOp");

        // Type conversion
        registerHiprtcStub(StableHloAst.ConvertOp.class, "ConvertOp");
        registerStub(StableHloAst.BitcastConvertOp.class, "BitcastConvertOp");

        // Quantization
        registerStub(StableHloAst.UniformQuantizeOp.class, "UniformQuantizeOp");
        registerStub(StableHloAst.UniformDequantizeOp.class, "UniformDequantizeOp");

        // Reduction
        registerHiprtcStub(StableHloAst.ReduceOp.class, "ReduceOp");
        registerStub(StableHloAst.ReduceWindowOp.class, "ReduceWindowOp");
        registerStub(StableHloAst.ReducePrecisionOp.class, "ReducePrecisionOp");
        registerStub(StableHloAst.SelectAndScatterOp.class, "SelectAndScatterOp");

        // Linear algebra (DotOp handled above with rocBLAS)
        registerStub(StableHloAst.DotGeneralOp.class, "DotGeneralOp - use rocBLAS batched GEMM");
        registerStub(StableHloAst.CholeskyOp.class, "CholeskyOp");
        registerStub(StableHloAst.TriangularSolveOp.class, "TriangularSolveOp");

        // Convolution and neural network (future: MIOpen integration)
        registerStub(StableHloAst.ConvolutionOp.class, "ConvolutionOp - future MIOpen integration");
        registerStub(StableHloAst.BatchNormTrainingOp.class, "BatchNormTrainingOp");
        registerStub(StableHloAst.BatchNormInferenceOp.class, "BatchNormInferenceOp");
        registerStub(StableHloAst.BatchNormGradOp.class, "BatchNormGradOp");

        // Control flow
        registerStub(StableHloAst.IfOp.class, "IfOp");
        registerStub(StableHloAst.WhileOp.class, "WhileOp");
        registerStub(StableHloAst.CaseOp.class, "CaseOp");
        registerStub(StableHloAst.MapOp.class, "MapOp");

        // Collective operations (future: RCCL integration)
        registerStub(StableHloAst.AfterAllOp.class, "AfterAllOp");
        registerStub(StableHloAst.AllGatherOp.class, "AllGatherOp - future RCCL integration");
        registerStub(StableHloAst.AllReduceOp.class, "AllReduceOp - future RCCL integration");
        registerStub(StableHloAst.AllToAllOp.class, "AllToAllOp");
        registerStub(StableHloAst.CollectiveBroadcastOp.class, "CollectiveBroadcastOp");
        registerStub(StableHloAst.CollectivePermuteOp.class, "CollectivePermuteOp");
        registerStub(StableHloAst.ReduceScatterOp.class, "ReduceScatterOp");
        registerStub(StableHloAst.PartitionIdOp.class, "PartitionIdOp");
        registerStub(StableHloAst.ReplicaIdOp.class, "ReplicaIdOp");

        // Communication
        registerStub(StableHloAst.InfeedOp.class, "InfeedOp");
        registerStub(StableHloAst.OutfeedOp.class, "OutfeedOp");
        registerStub(StableHloAst.RecvOp.class, "RecvOp");
        registerStub(StableHloAst.SendOp.class, "SendOp");

        // Tuple operations
        registerStub(StableHloAst.TupleOp.class, "TupleOp");
        registerStub(StableHloAst.GetTupleElementOp.class, "GetTupleElementOp");

        // Complex number operations
        registerStub(StableHloAst.RealOp.class, "RealOp");
        registerStub(StableHloAst.ImagOp.class, "ImagOp");
        registerStub(StableHloAst.ComplexOp.class, "ComplexOp");

        // Signal processing
        registerStub(StableHloAst.FftOp.class, "FftOp - future rocFFT integration");

        // Random number generation
        registerStub(StableHloAst.RngOp.class, "RngOp - future rocRAND integration");
        registerStub(StableHloAst.RngBitGeneratorOp.class, "RngBitGeneratorOp");

        // Other operations
        registerStub(StableHloAst.SortOp.class, "SortOp");
        registerStub(StableHloAst.OptimizationBarrierOp.class, "OptimizationBarrierOp");
        registerStub(StableHloAst.CompositeOp.class, "CompositeOp");
        // CustomCall operations for transformers (HIPRTC required)
        if (context != null) {
            kernels.put(StableHloAst.CustomCallOp.class, new HipCustomCallKernel(context, salt));
        } else {
            registerStub(StableHloAst.CustomCallOp.class, "CustomCallOp");
        }
        registerStub(StableHloAst.ReturnOp.class, "ReturnOp");
    }

    private <T extends StableHloAst.Operation> void registerStub(Class<T> opClass, String message) {
        kernels.put(opClass, new HipStubKernel(opClass.getSimpleName(), message));
    }

    private <T extends StableHloAst.Operation> void registerHiprtcStub(Class<T> opClass, String opName) {
        kernels.put(opClass, new HipStubKernel(opName,
            opName + " requires HIPRTC integration. HIP C++ source is ready in HipKernels."));
    }

    /**
     * Stub kernel that throws UnsupportedOperationException with helpful message.
     */
    private static class HipStubKernel implements HipOpKernel {
        private final String opName;
        private final String message;

        HipStubKernel(String opName, String message) {
            this.opName = opName;
            this.message = message;
        }

        @Override
        public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
            throw new UnsupportedOperationException(
                "HIP kernel not yet implemented for: " + opName + ". " + message +
                " Use CPU backend for reference implementation.");
        }
    }
}
