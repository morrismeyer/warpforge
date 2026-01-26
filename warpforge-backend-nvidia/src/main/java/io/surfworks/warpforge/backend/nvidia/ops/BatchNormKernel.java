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
 * CUDA kernel for batch normalization operations.
 *
 * <p>Supports:
 * - BatchNormInference: output = (input - mean) / sqrt(var + eps) * scale + offset
 * - BatchNormTraining: (stub - computes inference + returns dummy mean/var)
 * - BatchNormGrad: (stub - throws UnsupportedOperationException)
 *
 * <p>Uses NCHW layout where feature index is derived from position.
 *
 * @see CudaKernels#generateBatchNormInferenceF32
 */
public final class BatchNormKernel implements CudaOpKernel {

    private final CudaContext context;
    private final int salt;

    private long moduleInference;
    private long functionInference;
    private boolean initializedInference;

    public BatchNormKernel(CudaContext context, int salt) {
        this.context = context;
        this.salt = salt;
    }

    private synchronized void ensureInferenceInitialized() {
        if (initializedInference) return;
        String ptx = CudaKernels.generateBatchNormInferenceF32(salt);
        moduleInference = context.loadModule("batchnorm_inference_f32_salt" + salt, ptx);
        functionInference = context.getFunction(moduleInference, "batchnorm_inference_f32");
        initializedInference = true;
    }

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (op instanceof StableHloAst.BatchNormInferenceOp inferenceOp) {
            return executeBatchNormInference(inferenceOp, inputs);
        } else if (op instanceof StableHloAst.BatchNormTrainingOp trainingOp) {
            return executeBatchNormTraining(trainingOp, inputs);
        } else if (op instanceof StableHloAst.BatchNormGradOp) {
            throw new UnsupportedOperationException(
                "BatchNormGrad is not yet implemented");
        } else {
            throw new IllegalArgumentException(
                "Expected BatchNorm operation, got: " + op.getClass().getSimpleName());
        }
    }

    private List<Tensor> executeBatchNormInference(StableHloAst.BatchNormInferenceOp op, List<Tensor> inputs) {
        // BatchNormInference inputs: operand, scale, offset, mean, variance
        if (inputs.size() != 5) {
            throw new IllegalArgumentException(
                "BatchNormInference requires 5 inputs, got: " + inputs.size());
        }

        ensureInferenceInitialized();

        Tensor operand = inputs.get(0);   // [N, C, H, W] or [N, C] or similar
        Tensor scale = inputs.get(1);     // [C]
        Tensor offset = inputs.get(2);    // [C]
        Tensor mean = inputs.get(3);      // [C]
        Tensor variance = inputs.get(4);  // [C]

        int[] shape = operand.shape();
        float epsilon = op.epsilon();

        // Calculate dimensions based on shape
        int batchSize;
        int numFeatures;
        int spatialSize;

        if (shape.length == 4) {
            // NCHW format
            batchSize = shape[0];
            numFeatures = shape[1];
            spatialSize = shape[2] * shape[3];
        } else if (shape.length == 2) {
            // NC format (dense layers)
            batchSize = shape[0];
            numFeatures = shape[1];
            spatialSize = 1;
        } else if (shape.length == 3) {
            // NCH format (1D sequence)
            batchSize = shape[0];
            numFeatures = shape[1];
            spatialSize = shape[2];
        } else {
            throw new UnsupportedOperationException(
                "Unsupported tensor shape for BatchNorm: " + shape.length + "D");
        }

        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());
        int totalElements = batchSize * numFeatures * spatialSize;

        long operandByteSize = (long) totalElements * 4L;
        long featureByteSize = (long) numFeatures * 4L;

        long dOperand = context.allocate(operandByteSize);
        long dScale = context.allocate(featureByteSize);
        long dOffset = context.allocate(featureByteSize);
        long dMean = context.allocate(featureByteSize);
        long dVariance = context.allocate(featureByteSize);
        long dOutput = context.allocate(operandByteSize);
        long dEpsilon = context.allocate(4);
        long dTiming = 0;

        try (Arena arena = Arena.ofConfined()) {
            context.copyToDevice(dOperand, operand.data());
            context.copyToDevice(dScale, scale.data());
            context.copyToDevice(dOffset, offset.data());
            context.copyToDevice(dMean, mean.data());
            context.copyToDevice(dVariance, variance.data());

            // Copy epsilon to device
            MemorySegment epsHost = arena.allocate(ValueLayout.JAVA_FLOAT);
            epsHost.set(ValueLayout.JAVA_FLOAT, 0, epsilon);
            context.copyToDevice(dEpsilon, epsHost);

            if (salt >= CudaKernels.SALT_TIMING) {
                dTiming = context.allocate(8);
            }

            int blockSize = CudaKernels.ELEMENTWISE_BLOCK_SIZE;
            int gridSize = CudaKernels.calculateGridSize(totalElements, blockSize);

            // PTX parameter order: (operand_ptr, scale_ptr, offset_ptr, mean_ptr, variance_ptr, output_ptr, epsilon_ptr, batchSize, numFeatures, spatialSize, [timing_ptr])
            if (salt >= CudaKernels.SALT_TIMING) {
                context.launchKernelWithMixedParams(
                    functionInference,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dOperand, dScale, dOffset, dMean, dVariance, dOutput, dEpsilon},
                    new int[]{batchSize, numFeatures, spatialSize},
                    new float[]{},
                    new long[]{dTiming}  // timing_ptr comes after int params
                );
            } else {
                context.launchKernelWithMixedParams(
                    functionInference,
                    new int[]{gridSize, 1, 1}, new int[]{blockSize, 1, 1},
                    0,
                    new long[]{dOperand, dScale, dOffset, dMean, dVariance, dOutput, dEpsilon},
                    new int[]{batchSize, numFeatures, spatialSize},
                    new float[]{}
                );
            }

            context.synchronize();

            Tensor output = Tensor.zeros(outputSpec.dtype(), outputSpec.shape());
            context.copyToHost(output.data(), dOutput, operandByteSize);

            return List.of(output);

        } finally {
            context.free(dOperand);
            context.free(dScale);
            context.free(dOffset);
            context.free(dMean);
            context.free(dVariance);
            context.free(dOutput);
            context.free(dEpsilon);
            if (dTiming != 0) {
                context.free(dTiming);
            }
        }
    }

    private List<Tensor> executeBatchNormTraining(StableHloAst.BatchNormTrainingOp op, List<Tensor> inputs) {
        // BatchNormTraining inputs: operand, scale, offset
        if (inputs.size() != 3) {
            throw new IllegalArgumentException(
                "BatchNormTraining requires 3 inputs, got: " + inputs.size());
        }

        // For training, we would need to compute mean and variance from the batch.
        // For now, this is a stub that returns zeros for mean and variance.
        throw new UnsupportedOperationException(
            "BatchNormTraining is not yet implemented. Use BatchNormInference with precomputed statistics.");
    }

    @Override
    public boolean supports(StableHloAst.Operation op) {
        return op instanceof StableHloAst.BatchNormInferenceOp
            || op instanceof StableHloAst.BatchNormTrainingOp
            || op instanceof StableHloAst.BatchNormGradOp;
    }

    public int getSalt() {
        return salt;
    }
}
