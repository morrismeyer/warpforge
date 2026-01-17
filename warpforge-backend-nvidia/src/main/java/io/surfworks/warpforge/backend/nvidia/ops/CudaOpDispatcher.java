package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Dispatches StableHLO operations to CUDA kernel stubs.
 * All operations are registered but throw UnsupportedOperationException
 * until real CUDA implementations are provided.
 */
public final class CudaOpDispatcher {

    private final Map<Class<? extends StableHloAst.Operation>, CudaOpKernel> kernels = new ConcurrentHashMap<>();

    public CudaOpDispatcher() {
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
     * Register all StableHLO operations as stubs.
     * Each stub will throw UnsupportedOperationException until
     * a real CUDA kernel implementation is provided.
     */
    private void registerAllOperations() {
        // Binary elementwise operations
        registerStub(StableHloAst.AddOp.class);
        registerStub(StableHloAst.SubtractOp.class);
        registerStub(StableHloAst.MultiplyOp.class);
        registerStub(StableHloAst.DivideOp.class);
        registerStub(StableHloAst.MaximumOp.class);
        registerStub(StableHloAst.MinimumOp.class);
        registerStub(StableHloAst.PowerOp.class);
        registerStub(StableHloAst.RemainderOp.class);
        registerStub(StableHloAst.Atan2Op.class);
        registerStub(StableHloAst.AndOp.class);
        registerStub(StableHloAst.OrOp.class);
        registerStub(StableHloAst.XorOp.class);
        registerStub(StableHloAst.ShiftLeftOp.class);
        registerStub(StableHloAst.ShiftRightArithmeticOp.class);
        registerStub(StableHloAst.ShiftRightLogicalOp.class);

        // Unary elementwise operations
        registerStub(StableHloAst.NegateOp.class);
        registerStub(StableHloAst.AbsOp.class);
        registerStub(StableHloAst.ExpOp.class);
        registerStub(StableHloAst.LogOp.class);
        registerStub(StableHloAst.TanhOp.class);
        registerStub(StableHloAst.SqrtOp.class);
        registerStub(StableHloAst.RsqrtOp.class);
        registerStub(StableHloAst.SinOp.class);
        registerStub(StableHloAst.CosOp.class);
        registerStub(StableHloAst.TanOp.class);
        registerStub(StableHloAst.LogisticOp.class);
        registerStub(StableHloAst.CeilOp.class);
        registerStub(StableHloAst.FloorOp.class);
        registerStub(StableHloAst.SignOp.class);
        registerStub(StableHloAst.Expm1Op.class);
        registerStub(StableHloAst.Log1pOp.class);
        registerStub(StableHloAst.CbrtOp.class);
        registerStub(StableHloAst.IsFiniteOp.class);
        registerStub(StableHloAst.RoundNearestEvenOp.class);
        registerStub(StableHloAst.RoundNearestAfzOp.class);
        registerStub(StableHloAst.NotOp.class);
        registerStub(StableHloAst.PopcntOp.class);
        registerStub(StableHloAst.ClzOp.class);

        // Comparison and selection
        registerStub(StableHloAst.CompareOp.class);
        registerStub(StableHloAst.SelectOp.class);
        registerStub(StableHloAst.ClampOp.class);

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
