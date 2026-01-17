package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.convolution - N-dimensional convolution.
 */
public class ConvolutionKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Convolution requires exactly 2 inputs, got " + inputs.size());
        }

        StableHloAst.ConvolutionOp convOp = (StableHloAst.ConvolutionOp) op;
        Tensor input = inputs.get(0);
        Tensor kernel = inputs.get(1);
        TensorSpec outputSpec = TensorSpec.fromAst(op.tensorResultType());

        int[] inputShape = input.shape();
        int[] kernelShape = kernel.shape();
        int[] outputShape = outputSpec.shape();

        float[] inputData = input.toFloatArray();
        float[] kernelData = kernel.toFloatArray();
        float[] outputData = new float[(int) outputSpec.elementCount()];

        // Extract dimension specifications
        int inputBatchDim = (int) convOp.inputBatchDimension();
        int inputFeatureDim = (int) convOp.inputFeatureDimension();
        List<Long> inputSpatialDims = convOp.inputSpatialDimensions();

        int kernelInputFeatureDim = (int) convOp.kernelInputFeatureDimension();
        int kernelOutputFeatureDim = (int) convOp.kernelOutputFeatureDimension();
        List<Long> kernelSpatialDims = convOp.kernelSpatialDimensions();

        int outputBatchDim = (int) convOp.outputBatchDimension();
        int outputFeatureDim = (int) convOp.outputFeatureDimension();
        List<Long> outputSpatialDims = convOp.outputSpatialDimensions();

        List<Long> windowStrides = convOp.windowStrides();
        List<Long> paddingLow = convOp.paddingLow();
        List<Long> paddingHigh = convOp.paddingHigh();
        List<Long> lhsDilation = convOp.lhsDilation();
        List<Long> rhsDilation = convOp.rhsDilation();

        long featureGroupCount = convOp.featureGroupCount();
        long batchGroupCount = convOp.batchGroupCount();

        int numSpatialDims = inputSpatialDims.size();

        long[] inputStrides = computeStrides(inputShape);
        long[] kernelStrides = computeStrides(kernelShape);
        long[] outputStrides = computeStrides(outputShape);

        int batchSize = inputShape[inputBatchDim];
        int inputFeatures = inputShape[inputFeatureDim];
        int outputFeatures = kernelShape[kernelOutputFeatureDim];

        // Iterate over output
        int[] outputIdx = new int[outputShape.length];
        for (int outFlatIdx = 0; outFlatIdx < outputData.length; outFlatIdx++) {
            unflattenIndex(outFlatIdx, outputStrides, outputIdx);

            int batch = outputIdx[outputBatchDim];
            int outputFeature = outputIdx[outputFeatureDim];

            // Get output spatial coordinates
            int[] outputSpatial = new int[numSpatialDims];
            for (int s = 0; s < numSpatialDims; s++) {
                outputSpatial[s] = outputIdx[outputSpatialDims.get(s).intValue()];
            }

            float sum = 0;

            // Determine input feature range based on feature groups
            int inputFeaturesPerGroup = inputFeatures / (int) featureGroupCount;
            int featureGroup = outputFeature / (outputFeatures / (int) featureGroupCount);
            int inputFeatureStart = featureGroup * inputFeaturesPerGroup;
            int inputFeatureEnd = inputFeatureStart + inputFeaturesPerGroup;

            // Iterate over kernel
            for (int inputFeature = inputFeatureStart; inputFeature < inputFeatureEnd; inputFeature++) {
                // Iterate over kernel spatial dimensions
                int[] kernelSpatial = new int[numSpatialDims];
                int[] kernelSpatialSizes = new int[numSpatialDims];
                for (int s = 0; s < numSpatialDims; s++) {
                    kernelSpatialSizes[s] = kernelShape[kernelSpatialDims.get(s).intValue()];
                }

                boolean kernelDone = false;
                while (!kernelDone) {
                    // Compute input spatial coordinates
                    int[] inputSpatial = new int[numSpatialDims];
                    boolean validInput = true;

                    for (int s = 0; s < numSpatialDims; s++) {
                        int stride = windowStrides.isEmpty() ? 1 : windowStrides.get(s).intValue();
                        int padLow = paddingLow.isEmpty() ? 0 : paddingLow.get(s).intValue();
                        int lhsDil = lhsDilation.isEmpty() ? 1 : lhsDilation.get(s).intValue();
                        int rhsDil = rhsDilation.isEmpty() ? 1 : rhsDilation.get(s).intValue();

                        int dilatedKernelPos = kernelSpatial[s] * rhsDil;
                        int inputPos = outputSpatial[s] * stride - padLow + dilatedKernelPos;

                        // Handle input dilation
                        if (lhsDil > 1) {
                            if (inputPos % lhsDil != 0) {
                                validInput = false;
                                break;
                            }
                            inputPos /= lhsDil;
                        }

                        if (inputPos < 0 || inputPos >= inputShape[inputSpatialDims.get(s).intValue()]) {
                            validInput = false;
                            break;
                        }
                        inputSpatial[s] = inputPos;
                    }

                    if (validInput) {
                        // Build input index
                        int[] inputIdx = new int[inputShape.length];
                        inputIdx[inputBatchDim] = batch;
                        inputIdx[inputFeatureDim] = inputFeature;
                        for (int s = 0; s < numSpatialDims; s++) {
                            inputIdx[inputSpatialDims.get(s).intValue()] = inputSpatial[s];
                        }

                        // Build kernel index
                        int[] kernelIdx = new int[kernelShape.length];
                        kernelIdx[kernelOutputFeatureDim] = outputFeature;
                        kernelIdx[kernelInputFeatureDim] = inputFeature - inputFeatureStart;
                        for (int s = 0; s < numSpatialDims; s++) {
                            kernelIdx[kernelSpatialDims.get(s).intValue()] = kernelSpatial[s];
                        }

                        int inputFlatIdx = flattenIndex(inputIdx, inputStrides);
                        int kernelFlatIdx = flattenIndex(kernelIdx, kernelStrides);

                        sum += inputData[inputFlatIdx] * kernelData[kernelFlatIdx];
                    }

                    // Advance kernel spatial
                    for (int s = numSpatialDims - 1; s >= 0; s--) {
                        kernelSpatial[s]++;
                        if (kernelSpatial[s] < kernelSpatialSizes[s]) {
                            break;
                        }
                        kernelSpatial[s] = 0;
                        if (s == 0) {
                            kernelDone = true;
                        }
                    }
                    if (numSpatialDims == 0) kernelDone = true;
                }
            }

            outputData[outFlatIdx] = sum;
        }

        Tensor output = Tensor.fromFloatArray(outputData, outputShape);
        return List.of(output);
    }

    private long[] computeStrides(int[] shape) {
        long[] strides = new long[shape.length];
        long stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private void unflattenIndex(long flatIdx, long[] strides, int[] result) {
        for (int i = 0; i < strides.length; i++) {
            result[i] = (int) (flatIdx / strides[i]);
            flatIdx %= strides[i];
        }
    }

    private int flattenIndex(int[] indices, long[] strides) {
        long result = 0;
        for (int i = 0; i < indices.length; i++) {
            result += indices[i] * strides[i];
        }
        return (int) result;
    }
}
