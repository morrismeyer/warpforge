package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel implementation for stablehlo.add operation.
 *
 * <p>This kernel performs element-wise addition of two float32 tensors on the GPU.
 * It uses custom PTX with salt-based instrumentation rather than cuBLAS, enabling:
 * <ul>
 *   <li>Full observability via timing instrumentation</li>
 *   <li>Single code path for production and profiling</li>
 *   <li>Foundation for Phase 2+ kernel fusion</li>
 * </ul>
 *
 * @see CudaKernels#generateAddF32(int)
 */
public final class AddKernel implements CudaOpKernel {

    private static final String MODULE_NAME = "add_f32_module";
    private static final String FUNCTION_NAME = "add_f32";

    private final CudaContext context;
    private final int salt;
    private long module;
    private long function;
    private boolean initialized;

    /**
     * Create an Add kernel with no instrumentation.
     */
    public AddKernel(CudaContext context) {
        this(context, CudaKernels.SALT_NONE);
    }

    /**
     * Create an Add kernel with the specified instrumentation level.
     *
     * @param context CUDA context for execution
     * @param salt Instrumentation level (SALT_NONE, SALT_TIMING, SALT_TRACE)
     */
    public AddKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
        this.initialized = false;
    }

    /**
     * Lazily initialize the kernel module and function.
     * This allows the kernel to be created before the CUDA context is fully ready.
     */
    private synchronized void ensureInitialized() {
        if (initialized) {
            return;
        }

        String ptxSource = CudaKernels.generateAddF32(salt);
        String moduleName = MODULE_NAME + "_salt" + salt;

        module = context.loadModule(moduleName, ptxSource);
        function = context.getFunction(module, FUNCTION_NAME);
        initialized = true;
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
            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8); // u64
                // Zero the timing accumulator
                // Note: In production we'd use cudaMemset, for now we copy a zero
            }

            // Calculate launch configuration
            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            // Launch kernel
            // PTX parameter order: (a_ptr, b_ptr, out_ptr, n, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0, // shared memory
                    new long[]{dA, dB, dOut},
                    new int[]{n},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after n
                );
            } else {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0, // shared memory
                    new long[]{dA, dB, dOut},
                    new int[]{n},
                    new float[]{}
                );
            }

            // Synchronize to ensure kernel completion
            context.synchronize();

            // Allocate output tensor and copy result back
            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOut, byteSize);

            // TODO: If SALT_TIMING, read back timing data and record via JFR

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
