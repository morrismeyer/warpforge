package io.surfworks.warpforge.backend.amd.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.amd.hip.HipContext;
import io.surfworks.warpforge.backend.amd.hip.HipKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * HIP kernel implementation for stablehlo.add operation.
 *
 * <p>This kernel performs element-wise addition of two float32 tensors on AMD GPUs.
 * It uses HIP C++ with salt-based instrumentation, enabling:
 * <ul>
 *   <li>Full observability via timing instrumentation</li>
 *   <li>Single code path for production and profiling</li>
 *   <li>Foundation for Phase 2+ kernel fusion</li>
 * </ul>
 *
 * <h2>Implementation Status</h2>
 * <p>This kernel generates HIP C++ source code via {@link HipKernels#generateAddF32(int)}.
 * Full implementation requires HIPRTC (HIP Runtime Compilation) FFM bindings to compile
 * the generated source at runtime. Until then, this kernel throws UnsupportedOperationException.
 *
 * <p>For PRODUCTION tier matrix operations, use {@link RocblasDotKernel} which is fully functional.
 *
 * @see HipKernels#generateAddF32(int)
 */
public final class AddKernel implements HipOpKernel {

    private static final String MODULE_NAME = "add_f32_module";
    private static final String FUNCTION_NAME = "add_f32";

    private final HipContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    /**
     * Create an Add kernel with no instrumentation.
     */
    public AddKernel(HipContext context) {
        this(context, HipKernels.SALT_NONE);
    }

    /**
     * Create an Add kernel with the specified instrumentation level.
     *
     * @param context HIP context for execution
     * @param salt Instrumentation level (SALT_NONE, SALT_TIMING, SALT_TRACE)
     */
    public AddKernel(HipContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    /**
     * Lazily initialize the kernel module and function.
     *
     * <p>TODO: This requires HIPRTC FFM bindings to compile HIP C++ at runtime.
     * The generated source is available via {@link HipKernels#generateAddF32(int)}.
     */
    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        // Generate HIP C++ source (for documentation/future use)
        String hipSource = HipKernels.generateAddF32(salt);

        // TODO: Compile via HIPRTC
        // byte[] hsaco = HiprtcRuntime.compile(hipSource, FUNCTION_NAME);
        // module = context.loadModule(MODULE_NAME + "_salt" + salt, hsaco);
        // function = context.getFunction(module, FUNCTION_NAME);

        throw new UnsupportedOperationException(
            "AddKernel requires HIPRTC integration to compile HIP C++ at runtime. " +
            "Generated source (" + hipSource.length() + " chars) is ready for compilation. " +
            "Use CPU backend for elementwise operations until HIPRTC is integrated.");
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.AddOp)) {
            throw new IllegalArgumentException("Expected AddOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Add requires exactly 2 inputs, got: " + inputs.size());
        }

        ensureInitialized();

        Tensor lhs = inputs.get(0);
        Tensor rhs = inputs.get(1);

        // Validate input shapes match
        if (lhs.elementCount() != rhs.elementCount()) {
            throw new IllegalArgumentException(
                "Input tensors must have same element count: " + lhs.elementCount() + " vs " + rhs.elementCount());
        }

        int n = (int) lhs.elementCount();
        long byteSize = n * 4L; // float32 = 4 bytes

        // Get output spec from the operation
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        // Allocate device memory
        long dA = context.allocate(byteSize);
        long dB = context.allocate(byteSize);
        long dOut = context.allocate(byteSize);
        long dTiming = 0;

        try {
            // Copy inputs to device
            context.copyToDevice(dA, lhs.data());
            context.copyToDevice(dB, rhs.data());

            // Allocate timing accumulator if instrumented
            if (salt >= HipKernels.SALT_TIMING) {
                dTiming = context.allocate(8); // u64
            }

            // Calculate launch configuration
            int blockSize = HipKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = HipKernels.calculateGridSize(n);

            // Launch kernel
            if (salt >= HipKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridSize}, new int[]{blockSize},
                    0, // shared memory
                    new long[]{dA, dB, dOut, dTiming},
                    n
                );
            } else {
                context.launchKernelWithIntParams(
                    function,
                    new int[]{gridSize}, new int[]{blockSize},
                    0, // shared memory
                    new long[]{dA, dB, dOut},
                    n
                );
            }

            // Synchronize to ensure kernel completion
            context.synchronize();

            // Allocate output tensor and copy result back
            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSize);

            return List.of(output);

        } finally {
            // Free device memory
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
        return op instanceof StableHloAst.AddOp;
    }

    /**
     * Get the current instrumentation salt level.
     */
    public int getSalt() {
        return salt;
    }
}
