package io.surfworks.warpforge.backend.nvidia.ops;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaContext;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * CUDA kernel for pad operation.
 *
 * <p>Pads a tensor with specified low and high padding for each dimension.
 * Currently supports 1D and 2D tensors with edge padding only (no interior padding).
 *
 * @see CudaKernels#generatePad1DF32
 * @see CudaKernels#generatePad2DF32
 */
public final class PadKernel implements CudaOpKernel {

    private static final int BLOCK_SIZE_2D = 16;

    private final CudaContext context;
    private final int salt;

    private long module1D;
    private long function1D;
    private boolean initialized1D;

    private long module2D;
    private long function2D;
    private boolean initialized2D;

    public PadKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensure1DInitialized() {
        if (initialized1D) return;
        String ptx = CudaKernels.generatePad1DF32(salt);
        module1D = context.loadModule("pad_1d_f32_salt" + salt, ptx);
        function1D = context.getFunction(module1D, "pad_1d_f32");
        initialized1D = true;
    }

    private synchronized void ensure2DInitialized() {
        if (initialized2D) return;
        String ptx = CudaKernels.generatePad2DF32(salt);
        module2D = context.loadModule("pad_2d_f32_salt" + salt, ptx);
        function2D = context.getFunction(module2D, "pad_2d_f32");
        initialized2D = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.PadOp padOp)) {
            throw new IllegalArgumentException(
                "Expected PadOp, got: " + op.getClass().getSimpleName());
        }

        if (inputs.size() != 2) {
            throw new IllegalArgumentException(
                "Pad requires exactly 2 inputs (operand and padding_value), got: " + inputs.size());
        }

        Tensor input = inputs.get(0);
        Tensor paddingValueTensor = inputs.get(1);
        int ndim = input.shape().length;

        // Get padding value from scalar tensor
        float paddingValue = paddingValueTensor.getFloatFlat(0);

        List<Long> edgePaddingLow = padOp.edgePaddingLow();
        List<Long> edgePaddingHigh = padOp.edgePaddingHigh();
        List<Long> interiorPadding = padOp.interiorPadding();

        // Check for interior padding - not currently supported
        for (Long interior : interiorPadding) {
            if (interior != 0) {
                throw new UnsupportedOperationException(
                    "Interior padding is not currently supported");
            }
        }

        if (ndim == 1) {
            return execute1DPad(input, paddingValue, padOp, edgePaddingLow, edgePaddingHigh);
        } else if (ndim == 2) {
            return execute2DPad(input, paddingValue, padOp, edgePaddingLow, edgePaddingHigh);
        } else {
            throw new UnsupportedOperationException(
                "Only 1D and 2D padding is currently supported, got " + ndim + "D tensor");
        }
    }

    private List<Tensor> execute1DPad(Tensor input, float paddingValue,
                                       StableHloAst.PadOp padOp,
                                       List<Long> edgePaddingLow,
                                       List<Long> edgePaddingHigh) {
        ensure1DInitialized();

        int inSize = input.shape()[0];
        int lowPad = edgePaddingLow.get(0).intValue();
        int highPad = edgePaddingHigh.get(0).intValue();
        int outSize = lowPad + inSize + highPad;

        TensorSpec outputSpec = TensorSpec.fromAst(padOp.tensorResultType());

        long inputByteSize = inSize * 4L;
        long outputByteSize = outSize * 4L;

        long dInput = context.allocate(inputByteSize);
        long dOutput = context.allocate(outputByteSize);
        long dPadValue = context.allocate(4L);
        long dTiming = 0;

        try (Arena arena = Arena.ofConfined()) {
            context.copyToDevice(dInput, input.data());

            // Copy padding value to device
            MemorySegment padValueHost = arena.allocate(ValueLayout.JAVA_FLOAT);
            padValueHost.set(ValueLayout.JAVA_FLOAT, 0, paddingValue);
            context.copyToDevice(dPadValue, padValueHost);

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(outSize, blockSize);

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function1D,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput, dPadValue, dTiming},
                    inSize, outSize, lowPad
                );
            } else {
                context.launchKernelWithIntParams(
                    function1D,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dInput, dOutput, dPadValue},
                    inSize, outSize, lowPad
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, outputByteSize);

            return List.of(output);

        } finally {
            context.free(dInput);
            context.free(dOutput);
            context.free(dPadValue);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    private List<Tensor> execute2DPad(Tensor input, float paddingValue,
                                       StableHloAst.PadOp padOp,
                                       List<Long> edgePaddingLow,
                                       List<Long> edgePaddingHigh) {
        ensure2DInitialized();

        int inRows = input.shape()[0];
        int inCols = input.shape()[1];
        int lowPad0 = edgePaddingLow.get(0).intValue();
        int lowPad1 = edgePaddingLow.get(1).intValue();
        int highPad0 = edgePaddingHigh.get(0).intValue();
        int highPad1 = edgePaddingHigh.get(1).intValue();
        int outRows = lowPad0 + inRows + highPad0;
        int outCols = lowPad1 + inCols + highPad1;

        TensorSpec outputSpec = TensorSpec.fromAst(padOp.tensorResultType());

        int inTotal = inRows * inCols;
        int outTotal = outRows * outCols;
        long inputByteSize = inTotal * 4L;
        long outputByteSize = outTotal * 4L;

        long dInput = context.allocate(inputByteSize);
        long dOutput = context.allocate(outputByteSize);
        long dPadValue = context.allocate(4L);
        long dTiming = 0;

        try (Arena arena = Arena.ofConfined()) {
            context.copyToDevice(dInput, input.data());

            // Copy padding value to device
            MemorySegment padValueHost = arena.allocate(ValueLayout.JAVA_FLOAT);
            padValueHost.set(ValueLayout.JAVA_FLOAT, 0, paddingValue);
            context.copyToDevice(dPadValue, padValueHost);

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int gridX = (outCols + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;
            int gridY = (outRows + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D;

            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithIntParams(
                    function2D,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                    0,
                    new long[]{dInput, dOutput, dPadValue, dTiming},
                    inRows, inCols, outRows, outCols, lowPad0, lowPad1
                );
            } else {
                context.launchKernelWithIntParams(
                    function2D,
                    new int[]{gridX, gridY, 1}, new int[]{BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1},
                    0,
                    new long[]{dInput, dOutput, dPadValue},
                    inRows, inCols, outRows, outCols, lowPad0, lowPad1
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, outputByteSize);

            return List.of(output);

        } finally {
            context.free(dInput);
            context.free(dOutput);
            context.free(dPadValue);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.PadOp;
    }

    public int getSalt() {
        return salt;
    }
}
