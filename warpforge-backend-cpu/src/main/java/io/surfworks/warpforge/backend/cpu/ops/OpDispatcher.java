package io.surfworks.warpforge.backend.cpu.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.scalar.*;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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
        // Binary elementwise operations
        register(StableHloAst.AddOp.class, new AddKernel());
        register(StableHloAst.SubtractOp.class, new SubtractKernel());
        register(StableHloAst.MultiplyOp.class, new MultiplyKernel());
        register(StableHloAst.DivideOp.class, new DivideKernel());
        register(StableHloAst.MaximumOp.class, new MaximumKernel());
        register(StableHloAst.MinimumOp.class, new MinimumKernel());

        // Unary elementwise operations
        register(StableHloAst.NegateOp.class, new NegateKernel());
        register(StableHloAst.AbsOp.class, new AbsKernel());
        register(StableHloAst.ExpOp.class, new ExpKernel());
        register(StableHloAst.LogOp.class, new LogKernel());
        register(StableHloAst.TanhOp.class, new TanhKernel());
        register(StableHloAst.SqrtOp.class, new SqrtKernel());
        register(StableHloAst.RsqrtOp.class, new RsqrtKernel());
        register(StableHloAst.SinOp.class, new SinKernel());
        register(StableHloAst.CosOp.class, new CosKernel());
        register(StableHloAst.LogisticOp.class, new LogisticKernel());

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

        // Type conversion
        register(StableHloAst.ConvertOp.class, new ConvertKernel());
    }
}
