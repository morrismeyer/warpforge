package io.surfworks.warpforge.backend.amd.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.amd.hip.HipContext;
import io.surfworks.warpforge.backend.amd.hip.HipKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * HIP kernel implementation for stablehlo.multiply operation.
 *
 * <p>This kernel performs element-wise multiplication of two float32 tensors on AMD GPUs.
 *
 * <h2>Implementation Status</h2>
 * <p>This kernel generates HIP C++ source code via {@link HipKernels#generateMultiplyF32(int)}.
 * Full implementation requires HIPRTC FFM bindings. Until then, throws UnsupportedOperationException.
 *
 * @see HipKernels#generateMultiplyF32(int)
 */
public final class MultiplyKernel implements HipOpKernel {

    private static final String MODULE_NAME = "multiply_f32_module";
    private static final String FUNCTION_NAME = "multiply_f32";

    private final HipContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    public MultiplyKernel(HipContext context) {
        this(context, HipKernels.SALT_NONE);
    }

    public MultiplyKernel(HipContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        String hipSource = HipKernels.generateMultiplyF32(salt);

        throw new UnsupportedOperationException(
            "MultiplyKernel requires HIPRTC integration to compile HIP C++ at runtime. " +
            "Generated source (" + hipSource.length() + " chars) is ready for compilation. " +
            "Use CPU backend for elementwise operations until HIPRTC is integrated.");
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.MultiplyOp)) {
            throw new IllegalArgumentException("Expected MultiplyOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Multiply requires exactly 2 inputs, got: " + inputs.size());
        }

        ensureInitialized();

        Tensor lhs = inputs.get(0);
        Tensor rhs = inputs.get(1);

        if (lhs.elementCount() != rhs.elementCount()) {
            throw new IllegalArgumentException(
                "Input tensors must have same element count: " + lhs.elementCount() + " vs " + rhs.elementCount());
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

            if (salt >= HipKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = HipKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = HipKernels.calculateGridSize(n);

            if (salt >= HipKernels.SALT_TIMING) {
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
        return op instanceof StableHloAst.MultiplyOp;
    }

    public int getSalt() {
        return salt;
    }
}
