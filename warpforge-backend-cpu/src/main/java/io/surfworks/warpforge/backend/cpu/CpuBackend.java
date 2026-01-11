package io.surfworks.warpforge.backend.cpu;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpDispatcher;
import io.surfworks.warpforge.core.backend.Backend;
import io.surfworks.warpforge.core.backend.BackendCapabilities;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;
import java.util.Set;

/**
 * CPU backend for executing StableHLO operations.
 * Uses scalar implementations with optional Vector API (SIMD) acceleration.
 */
public class CpuBackend implements Backend {

    private final OpDispatcher dispatcher;
    private final BackendCapabilities capabilities;
    private volatile boolean closed = false;

    public CpuBackend() {
        this.dispatcher = new OpDispatcher();
        this.capabilities = new BackendCapabilities(
            Set.of(ScalarType.F32, ScalarType.F64, ScalarType.I32, ScalarType.I64),
            hasVectorSupport(),
            false, // No async support
            8,     // Max tensor rank
            Integer.MAX_VALUE
        );
    }

    @Override
    public String name() {
        return "cpu";
    }

    @Override
    public BackendCapabilities capabilities() {
        return capabilities;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        checkNotClosed();
        return dispatcher.dispatch(op, inputs);
    }

    @Override
    public Tensor allocate(TensorSpec spec) {
        checkNotClosed();
        return Tensor.zeros(spec.dtype(), spec.shape());
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return dispatcher.supports(op);
    }

    /**
     * Get list of supported operation types.
     */
    public List<String> supportedOperations() {
        return dispatcher.supportedOps();
    }

    @Override
    public void close() {
        closed = true;
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("Backend has been closed");
        }
    }

    private static boolean hasVectorSupport() {
        try {
            // Check if Vector API is available
            Class.forName("jdk.incubator.vector.FloatVector");
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }
}
