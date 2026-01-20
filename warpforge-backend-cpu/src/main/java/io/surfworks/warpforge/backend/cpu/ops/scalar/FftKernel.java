package io.surfworks.warpforge.backend.cpu.ops.scalar;

import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.FftOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU kernel for stablehlo.fft.
 *
 * <p>Computes the Fast Fourier Transform (FFT) or its inverse.
 * Supports FFT, IFFT, RFFT, and IRFFT types.
 */
public final class FftKernel implements OpKernel {

    @Override
    public List<Tensor> execute(Operation op, List<Tensor> inputs) {
        FftOp fftOp = (FftOp) op;

        if (inputs.size() != 1) {
            throw new IllegalArgumentException("fft requires exactly 1 input");
        }

        Tensor input = inputs.get(0);
        String fftType = fftOp.fftType();
        List<Long> fftLength = fftOp.fftLength();

        int[] shape = input.shape();
        float[] inputData = input.toFloatArray();

        // Determine FFT size from the last dimension(s)
        int fftSize = fftLength.isEmpty() ? shape[shape.length - 1] : fftLength.get(0).intValue();

        switch (fftType.toUpperCase()) {
            case "FFT":
                return computeFft(inputData, shape, fftSize, false);
            case "IFFT":
                return computeFft(inputData, shape, fftSize, true);
            case "RFFT":
                return computeRfft(inputData, shape, fftSize);
            case "IRFFT":
                return computeIrfft(inputData, shape, fftSize);
            default:
                throw new UnsupportedOperationException("Unsupported FFT type: " + fftType);
        }
    }

    private List<Tensor> computeFft(float[] input, int[] shape, int n, boolean inverse) {
        // Complex FFT: input has shape [..., n, 2] where last dim is (real, imag)
        int batchSize = 1;
        for (int i = 0; i < shape.length - 2; i++) {
            batchSize *= shape[i];
        }

        float[] output = new float[input.length];

        for (int batch = 0; batch < batchSize; batch++) {
            int offset = batch * n * 2;
            fft1d(input, offset, output, offset, n, inverse);
        }

        Tensor result = Tensor.zeros(shape);
        result.copyFrom(output);
        return List.of(result);
    }

    private List<Tensor> computeRfft(float[] input, int[] shape, int n) {
        // Real FFT: input is real, output is complex with shape [..., n/2+1, 2]
        int batchSize = 1;
        for (int i = 0; i < shape.length - 1; i++) {
            batchSize *= shape[i];
        }

        int outputComplexSize = n / 2 + 1;
        int[] outputShape = new int[shape.length + 1];
        System.arraycopy(shape, 0, outputShape, 0, shape.length - 1);
        outputShape[shape.length - 1] = outputComplexSize;
        outputShape[shape.length] = 2;

        float[] output = new float[batchSize * outputComplexSize * 2];

        for (int batch = 0; batch < batchSize; batch++) {
            int inOffset = batch * n;
            int outOffset = batch * outputComplexSize * 2;
            rfft1d(input, inOffset, output, outOffset, n);
        }

        Tensor result = Tensor.zeros(outputShape);
        result.copyFrom(output);
        return List.of(result);
    }

    private List<Tensor> computeIrfft(float[] input, int[] shape, int n) {
        // Inverse real FFT: input is complex, output is real
        int batchSize = 1;
        for (int i = 0; i < shape.length - 2; i++) {
            batchSize *= shape[i];
        }

        int inputComplexSize = shape[shape.length - 2];
        int[] outputShape = new int[shape.length - 1];
        System.arraycopy(shape, 0, outputShape, 0, shape.length - 2);
        outputShape[shape.length - 2] = n;

        float[] output = new float[batchSize * n];

        for (int batch = 0; batch < batchSize; batch++) {
            int inOffset = batch * inputComplexSize * 2;
            int outOffset = batch * n;
            irfft1d(input, inOffset, output, outOffset, inputComplexSize, n);
        }

        Tensor result = Tensor.zeros(outputShape);
        result.copyFrom(output);
        return List.of(result);
    }

    // Simple DFT implementation (O(n^2), for correctness; production would use Cooley-Tukey)
    private void fft1d(float[] input, int inOffset, float[] output, int outOffset, int n, boolean inverse) {
        double sign = inverse ? 1.0 : -1.0;
        double scale = inverse ? 1.0 / n : 1.0;

        for (int k = 0; k < n; k++) {
            double realSum = 0;
            double imagSum = 0;

            for (int j = 0; j < n; j++) {
                double angle = sign * 2.0 * Math.PI * k * j / n;
                double cos = Math.cos(angle);
                double sin = Math.sin(angle);

                double realIn = input[inOffset + j * 2];
                double imagIn = input[inOffset + j * 2 + 1];

                realSum += realIn * cos - imagIn * sin;
                imagSum += realIn * sin + imagIn * cos;
            }

            output[outOffset + k * 2] = (float) (realSum * scale);
            output[outOffset + k * 2 + 1] = (float) (imagSum * scale);
        }
    }

    private void rfft1d(float[] input, int inOffset, float[] output, int outOffset, int n) {
        int outputSize = n / 2 + 1;

        for (int k = 0; k < outputSize; k++) {
            double realSum = 0;
            double imagSum = 0;

            for (int j = 0; j < n; j++) {
                double angle = -2.0 * Math.PI * k * j / n;
                realSum += input[inOffset + j] * Math.cos(angle);
                imagSum += input[inOffset + j] * Math.sin(angle);
            }

            output[outOffset + k * 2] = (float) realSum;
            output[outOffset + k * 2 + 1] = (float) imagSum;
        }
    }

    private void irfft1d(float[] input, int inOffset, float[] output, int outOffset, int inputSize, int n) {
        for (int k = 0; k < n; k++) {
            double sum = 0;

            for (int j = 0; j < inputSize; j++) {
                double angle = 2.0 * Math.PI * k * j / n;
                double realIn = input[inOffset + j * 2];
                double imagIn = input[inOffset + j * 2 + 1];

                sum += realIn * Math.cos(angle) - imagIn * Math.sin(angle);

                // Mirror frequencies (except DC and Nyquist)
                if (j > 0 && j < inputSize - 1) {
                    sum += realIn * Math.cos(angle) + imagIn * Math.sin(angle);
                }
            }

            output[outOffset + k] = (float) (sum / n);
        }
    }

    @Override
    public boolean supports(Operation op) {
        return op instanceof StableHloAst.FftOp;
    }
}
