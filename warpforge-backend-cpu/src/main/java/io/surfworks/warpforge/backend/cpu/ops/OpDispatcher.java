package io.surfworks.warpforge.backend.cpu.ops;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.scalar.AbsKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.AddKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.AndKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.Atan2Kernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.BatchNormInferenceKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.BatchNormTrainingKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.BitcastConvertKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.BroadcastInDimKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.CbrtKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.CeilKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ClampKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ClzKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.CompareKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ConcatenateKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ConstantKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ConvertKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ConvolutionKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.CosKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.DivideKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.DotGeneralKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.DotKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.DynamicSliceKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.DynamicUpdateSliceKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ExpKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.Expm1Kernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.FloorKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.GatherKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.GetDimensionSizeKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.IotaKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.IsFiniteKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.Log1pKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.LogKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.LogisticKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.MaximumKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.MinimumKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.MultiplyKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.NegateKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.NotKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.OrKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.PadKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.PopcntKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.PowerKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ReduceKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ReduceWindowKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.RemainderKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ReshapeKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ReverseKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.RngKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.RoundNearestAfzKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.RoundNearestEvenKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.RsqrtKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ScatterKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.SelectKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ShiftLeftKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ShiftRightArithmeticKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.ShiftRightLogicalKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.SignKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.SinKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.SliceKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.SortKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.SqrtKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.SubtractKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.TanKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.TanhKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.TransposeKernel;
import io.surfworks.warpforge.backend.cpu.ops.scalar.XorKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * Dispatches StableHLO operations to their corresponding kernels.
 */
public final class OpDispatcher {

    private final Map<Class<? extends StableHloAst.Operation>, OpKernel> kernels = new ConcurrentHashMap<>();

    public OpDispatcher() {
        registerDefaultKernels();
    }

    /**
     * Register a kernel for an operation type.
     */
    public <T extends StableHloAst.Operation> void register(Class<T> opClass, OpKernel kernel) {
        kernels.put(opClass, kernel);
    }

    /**
     * Execute an operation.
     *
     * @param op     The operation to execute
     * @param inputs Input tensors
     * @return Output tensors
     * @throws UnsupportedOperationException if no kernel is registered
     */
    public List<Tensor> dispatch(StableHloAst.Operation op, List<Tensor> inputs) {
        OpKernel kernel = kernels.get(op.getClass());
        if (kernel == null) {
            throw new UnsupportedOperationException(
                "No kernel registered for operation: " + op.opName());
        }
        return kernel.execute(op, inputs);
    }

    /**
     * Check if an operation is supported.
     */
    public boolean supports(StableHloAst.Operation op) {
        OpKernel kernel = kernels.get(op.getClass());
        return kernel != null && kernel.supports(op);
    }

    /**
     * Get list of supported operation types.
     */
    public List<String> supportedOps() {
        return kernels.keySet().stream()
            .map(Class::getSimpleName)
            .map(s -> s.replace("Op", ""))
            .sorted()
            .toList();
    }

    private void registerDefaultKernels() {
        // Binary elementwise operations (float)
        register(StableHloAst.AddOp.class, new AddKernel());
        register(StableHloAst.SubtractOp.class, new SubtractKernel());
        register(StableHloAst.MultiplyOp.class, new MultiplyKernel());
        register(StableHloAst.DivideOp.class, new DivideKernel());
        register(StableHloAst.MaximumOp.class, new MaximumKernel());
        register(StableHloAst.MinimumOp.class, new MinimumKernel());
        register(StableHloAst.PowerOp.class, new PowerKernel());
        register(StableHloAst.RemainderOp.class, new RemainderKernel());
        register(StableHloAst.Atan2Op.class, new Atan2Kernel());

        // Binary elementwise operations (integer/bitwise)
        register(StableHloAst.AndOp.class, new AndKernel());
        register(StableHloAst.OrOp.class, new OrKernel());
        register(StableHloAst.XorOp.class, new XorKernel());
        register(StableHloAst.ShiftLeftOp.class, new ShiftLeftKernel());
        register(StableHloAst.ShiftRightArithmeticOp.class, new ShiftRightArithmeticKernel());
        register(StableHloAst.ShiftRightLogicalOp.class, new ShiftRightLogicalKernel());

        // Unary elementwise operations (float)
        register(StableHloAst.NegateOp.class, new NegateKernel());
        register(StableHloAst.AbsOp.class, new AbsKernel());
        register(StableHloAst.ExpOp.class, new ExpKernel());
        register(StableHloAst.LogOp.class, new LogKernel());
        register(StableHloAst.TanhOp.class, new TanhKernel());
        register(StableHloAst.SqrtOp.class, new SqrtKernel());
        register(StableHloAst.RsqrtOp.class, new RsqrtKernel());
        register(StableHloAst.SinOp.class, new SinKernel());
        register(StableHloAst.CosOp.class, new CosKernel());
        register(StableHloAst.TanOp.class, new TanKernel());
        register(StableHloAst.LogisticOp.class, new LogisticKernel());
        register(StableHloAst.CeilOp.class, new CeilKernel());
        register(StableHloAst.FloorOp.class, new FloorKernel());
        register(StableHloAst.SignOp.class, new SignKernel());
        register(StableHloAst.Expm1Op.class, new Expm1Kernel());
        register(StableHloAst.Log1pOp.class, new Log1pKernel());
        register(StableHloAst.CbrtOp.class, new CbrtKernel());
        register(StableHloAst.IsFiniteOp.class, new IsFiniteKernel());
        register(StableHloAst.RoundNearestEvenOp.class, new RoundNearestEvenKernel());
        register(StableHloAst.RoundNearestAfzOp.class, new RoundNearestAfzKernel());

        // Unary elementwise operations (integer/bitwise)
        register(StableHloAst.NotOp.class, new NotKernel());
        register(StableHloAst.PopcntOp.class, new PopcntKernel());
        register(StableHloAst.ClzOp.class, new ClzKernel());

        // Comparison and selection
        register(StableHloAst.CompareOp.class, new CompareKernel());
        register(StableHloAst.SelectOp.class, new SelectKernel());
        register(StableHloAst.ClampOp.class, new ClampKernel());

        // Constants
        register(StableHloAst.ConstantOp.class, new ConstantKernel());

        // Shape manipulation
        register(StableHloAst.ReshapeOp.class, new ReshapeKernel());
        register(StableHloAst.TransposeOp.class, new TransposeKernel());
        register(StableHloAst.BroadcastInDimOp.class, new BroadcastInDimKernel());
        register(StableHloAst.ConcatenateOp.class, new ConcatenateKernel());
        register(StableHloAst.SliceOp.class, new SliceKernel());
        register(StableHloAst.ReverseOp.class, new ReverseKernel());
        register(StableHloAst.PadOp.class, new PadKernel());
        register(StableHloAst.IotaOp.class, new IotaKernel());
        register(StableHloAst.GatherOp.class, new GatherKernel());
        register(StableHloAst.ScatterOp.class, new ScatterKernel());
        register(StableHloAst.DynamicSliceOp.class, new DynamicSliceKernel());
        register(StableHloAst.DynamicUpdateSliceOp.class, new DynamicUpdateSliceKernel());
        register(StableHloAst.GetDimensionSizeOp.class, new GetDimensionSizeKernel());

        // Type conversion
        register(StableHloAst.ConvertOp.class, new ConvertKernel());
        register(StableHloAst.BitcastConvertOp.class, new BitcastConvertKernel());

        // Reduction operations
        register(StableHloAst.ReduceOp.class, new ReduceKernel());
        register(StableHloAst.ReduceWindowOp.class, new ReduceWindowKernel());

        // Linear algebra
        register(StableHloAst.DotOp.class, new DotKernel());
        register(StableHloAst.DotGeneralOp.class, new DotGeneralKernel());

        // Convolution and neural network
        register(StableHloAst.ConvolutionOp.class, new ConvolutionKernel());
        register(StableHloAst.BatchNormInferenceOp.class, new BatchNormInferenceKernel());
        register(StableHloAst.BatchNormTrainingOp.class, new BatchNormTrainingKernel());

        // Sorting
        register(StableHloAst.SortOp.class, new SortKernel());

        // Random number generation
        register(StableHloAst.RngOp.class, new RngKernel());
    }
}
