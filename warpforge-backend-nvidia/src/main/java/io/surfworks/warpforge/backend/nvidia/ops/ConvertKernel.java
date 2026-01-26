package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * CUDA kernel for type conversion operations.
 *
 * <p>Converts tensor elements from one dtype to another. Supports:
 * <ul>
 *   <li>F32 to I32 (truncation)</li>
 *   <li>I32 to F32</li>
 *   <li>F32 to F32 (identity)</li>
 *   <li>I32 to I32 (identity)</li>
 * </ul>
 *
 * @see CudaKernels#generateConvertF32toI32
 * @see CudaKernels#generateConvertI32toF32
 */
public final class ConvertKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;

    // Lazy-initialized modules for different conversion types
    private long moduleF32toI32;
    private long functionF32toI32;
    private boolean initializedF32toI32;

    private long moduleI32toF32;
    private long functionI32toF32;
    private boolean initializedI32toF32;

    private long moduleF32toF32;
    private long functionF32toF32;
    private boolean initializedF32toF32;

    private long moduleI32toI32;
    private long functionI32toI32;
    private boolean initializedI32toI32;

    public ConvertKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensureF32toI32Initialized() {
        if (initializedF32toI32) return;
        String ptx = CudaKernels.generateConvertF32toI32(salt);
        moduleF32toI32 = context.loadModule("convert_f32_to_i32_salt" + salt, ptx);
        functionF32toI32 = context.getFunction(moduleF32toI32, "convert_f32_to_i32");
        initializedF32toI32 = true;
    }

    private synchronized void ensureI32toF32Initialized() {
        if (initializedI32toF32) return;
        String ptx = CudaKernels.generateConvertI32toF32(salt);
        moduleI32toF32 = context.loadModule("convert_i32_to_f32_salt" + salt, ptx);
        functionI32toF32 = context.getFunction(moduleI32toF32, "convert_i32_to_f32");
        initializedI32toF32 = true;
    }

    private synchronized void ensureF32toF32Initialized() {
        if (initializedF32toF32) return;
        String ptx = CudaKernels.generateConvertF32toF32(salt);
        moduleF32toF32 = context.loadModule("convert_f32_to_f32_salt" + salt, ptx);
        functionF32toF32 = context.getFunction(moduleF32toF32, "reshape_f32");
        initializedF32toF32 = true;
    }

    private synchronized void ensureI32toI32Initialized() {
        if (initializedI32toI32) return;
        String ptx = CudaKernels.generateConvertI32toI32(salt);
        moduleI32toI32 = context.loadModule("convert_i32_to_i32_salt" + salt, ptx);
        functionI32toI32 = context.getFunction(moduleI32toI32, "convert_i32_to_i32");
        initializedI32toI32 = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.ConvertOp)) {
            throw new IllegalArgumentException(
                "Expected ConvertOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 1) {
            throw new IllegalArgumentException(
                "Convert requires exactly 1 input, got: " + inputs.size());
        }

        Tensor input = inputs.get(0);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        ScalarType inputDtype = input.dtype();
        ScalarType outputDtype = outputSpec.dtype();

        // Determine conversion type and dispatch
        if (inputDtype == ScalarType.F32 && outputDtype == ScalarType.I32) {
            return executeF32toI32(input, outputSpec);
        } else if (inputDtype == ScalarType.I32 && outputDtype == ScalarType.F32) {
            return executeI32toF32(input, outputSpec);
        } else if (inputDtype == ScalarType.F32 && outputDtype == ScalarType.F32) {
            return executeF32toF32(input, outputSpec);
        } else if (inputDtype == ScalarType.I32 && outputDtype == ScalarType.I32) {
            return executeI32toI32(input, outputSpec);
        } else {
            throw new UnsupportedOperationException(
                "Unsupported conversion: " + inputDtype + " -> " + outputDtype);
        }
    }

    private List<Tensor> executeF32toI32(Tensor input, TensorSpec outputSpec) {
        ensureF32toI32Initialized();
        return executeConversion(input, outputSpec, functionF32toI32, 4, 4);
    }

    private List<Tensor> executeI32toF32(Tensor input, TensorSpec outputSpec) {
        ensureI32toF32Initialized();
        return executeConversion(input, outputSpec, functionI32toF32, 4, 4);
    }

    private List<Tensor> executeF32toF32(Tensor input, TensorSpec outputSpec) {
        ensureF32toF32Initialized();
        return executeConversion(input, outputSpec, functionF32toF32, 4, 4);
    }

    private List<Tensor> executeI32toI32(Tensor input, TensorSpec outputSpec) {
        ensureI32toI32Initialized();
        return executeConversion(input, outputSpec, functionI32toI32, 4, 4);
    }

    private List<Tensor> executeConversion(Tensor input, TensorSpec outputSpec,
                                            long function, int inputElementSize, int outputElementSize) {
        int n = (int) input.elementCount();
        long inputByteSize = n * (long) inputElementSize;
        long outputByteSize = n * (long) outputElementSize;

        long dInput = context.allocate(inputByteSize);
        long dOutput = context.allocate(outputByteSize);
        long dTiming = 0;

        try {
            context.copyToDevice(dInput, input.data());

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(n, blockSize);

            // PTX parameter order: (in_ptr, out_ptr, n, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput},
                    new int[]{n},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after n
                );
            } else {
                context.launchKernelWithMixedParams(
                    function,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput},
                    new int[]{n},
                    new float[]{}
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, outputByteSize);

            return List.of(output);

        } finally {
            context.free(dInput);
            context.free(dOutput);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.ConvertOp;
    }

    public int getSalt() {
        return salt;
    }
}
