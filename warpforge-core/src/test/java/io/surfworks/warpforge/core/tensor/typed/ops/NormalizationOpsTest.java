package io.surfworks.warpforge.core.tensor.typed.ops;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Tests for NormalizationOps normalization operations.
 */
@DisplayName("NormalizationOps")
class NormalizationOpsTest {

    private static final float EPSILON = 1e-4f;

    @Nested
    @DisplayName("LayerNorm on Vector")
    class LayerNormVectorTests {

        @Test
        @DisplayName("layerNorm normalizes vector to zero mean")
        void layerNormNormalizesVectorToZeroMean() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5}, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{1, 1, 1, 1, 1}, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.fromFloatArray(
                    new float[]{0, 0, 0, 0, 0}, new Vector(5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = NormalizationOps.layerNormVector(
                         input, weight, bias, 1e-5f)) {

                float[] data = output.underlying().toFloatArray();

                // Mean should be approximately 0
                float mean = 0;
                for (float v : data) {
                    mean += v;
                }
                mean /= data.length;
                assertEquals(0, mean, EPSILON);
            }
        }

        @Test
        @DisplayName("layerNorm normalizes vector to unit variance")
        void layerNormNormalizesVectorToUnitVariance() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{10, 20, 30, 40}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{1, 1, 1, 1}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.fromFloatArray(
                    new float[]{0, 0, 0, 0}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = NormalizationOps.layerNormVector(
                         input, weight, bias, 1e-5f)) {

                float[] data = output.underlying().toFloatArray();

                // Compute variance
                float mean = 0;
                for (float v : data) {
                    mean += v;
                }
                mean /= data.length;

                float variance = 0;
                for (float v : data) {
                    variance += (v - mean) * (v - mean);
                }
                variance /= data.length;

                // Variance should be approximately 1
                assertEquals(1.0f, variance, 0.01f);
            }
        }

        @Test
        @DisplayName("layerNorm applies scale and shift")
        void layerNormAppliesScaleAndShift() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{0, 0, 0}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{2, 2, 2}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = NormalizationOps.layerNormVector(
                         input, weight, bias, 1e-5f)) {

                float[] data = output.underlying().toFloatArray();

                // Normalized zero stays zero, then scaled by 2 and shifted by [1, 2, 3]
                assertEquals(1, data[0], EPSILON);
                assertEquals(2, data[1], EPSILON);
                assertEquals(3, data[2], EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("LayerNorm on Matrix")
    class LayerNormMatrixTests {

        @Test
        @DisplayName("layerNorm normalizes each row independently")
        void layerNormNormalizesEachRowIndependently() {
            try (TypedTensor<Matrix, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6},
                    new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{1, 1, 1}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.fromFloatArray(
                    new float[]{0, 0, 0}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> output = NormalizationOps.layerNormMatrix(
                         input, weight, bias, 1e-5f)) {

                assertArrayEquals(new int[]{2, 3}, output.dimensions());

                float[] data = output.underlying().toFloatArray();

                // Each row should have mean ≈ 0
                float row0Mean = (data[0] + data[1] + data[2]) / 3;
                float row1Mean = (data[3] + data[4] + data[5]) / 3;

                assertEquals(0, row0Mean, EPSILON);
                assertEquals(0, row1Mean, EPSILON);
            }
        }

        @Test
        @DisplayName("layerNorm rejects weight dimension mismatch")
        void layerNormRejectsWeightDimensionMismatch() {
            try (TypedTensor<Matrix, F32, Cpu> input = TypedTensor.zeros(
                    new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.zeros(
                    new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.zeros(
                    new Vector(3), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> NormalizationOps.layerNormMatrix(input, weight, bias, 1e-5f));
            }
        }
    }

    @Nested
    @DisplayName("LayerNorm on Rank3")
    class LayerNormRank3Tests {

        @Test
        @DisplayName("layerNorm normalizes along hidden dimension")
        void layerNormNormalizesAlongHiddenDimension() {
            // Shape: [batch=2, seq=2, hidden=4]
            float[] inputData = new float[16];
            for (int i = 0; i < 16; i++) {
                inputData[i] = i;
            }

            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.fromFloatArray(
                    inputData, new Rank3(2, 2, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{1, 1, 1, 1}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.fromFloatArray(
                    new float[]{0, 0, 0, 0}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> output = NormalizationOps.layerNorm(
                         input, weight, bias, 1e-5f)) {

                assertArrayEquals(new int[]{2, 2, 4}, output.dimensions());

                float[] data = output.underlying().toFloatArray();

                // Each [hidden] slice should have mean ≈ 0
                for (int i = 0; i < 4; i++) {
                    int offset = i * 4;
                    float mean = (data[offset] + data[offset+1] + data[offset+2] + data[offset+3]) / 4;
                    assertEquals(0, mean, EPSILON);
                }
            }
        }

        @Test
        @DisplayName("layerNorm preserves shape")
        void layerNormPreservesShape() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.zeros(
                    new Rank3(4, 8, 64), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.zeros(
                    new Vector(64), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.zeros(
                    new Vector(64), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> output = NormalizationOps.layerNorm(
                         input, weight, bias, 1e-5f)) {

                assertArrayEquals(new int[]{4, 8, 64}, output.dimensions());
            }
        }

        @Test
        @DisplayName("layerNormNoAffine works without scale/shift")
        void layerNormNoAffineWorksWithoutScaleShift() {
            float[] inputData = new float[8];
            for (int i = 0; i < 8; i++) {
                inputData[i] = i * 10;
            }

            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.fromFloatArray(
                    inputData, new Rank3(2, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> output = NormalizationOps.layerNormNoAffine(
                         input, 1e-5f)) {

                assertArrayEquals(new int[]{2, 2, 2}, output.dimensions());

                float[] data = output.underlying().toFloatArray();

                // Each [hidden] pair should have mean ≈ 0
                for (int i = 0; i < 4; i++) {
                    float mean = (data[i * 2] + data[i * 2 + 1]) / 2;
                    assertEquals(0, mean, EPSILON);
                }
            }
        }
    }

    @Nested
    @DisplayName("RMSNorm on Vector")
    class RmsNormVectorTests {

        @Test
        @DisplayName("rmsNorm normalizes by root mean square")
        void rmsNormNormalizesByRootMeanSquare() {
            // RMS of [3, 4] = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.536
            // After normalization: [3/3.536, 4/3.536] ≈ [0.849, 1.131]
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{3, 4}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{1, 1}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = NormalizationOps.rmsNormVector(
                         input, weight, 1e-5f)) {

                float[] data = output.underlying().toFloatArray();

                // Verify RMS normalization
                float rms = (float) Math.sqrt((3*3 + 4*4) / 2.0);
                assertEquals(3 / rms, data[0], EPSILON);
                assertEquals(4 / rms, data[1], EPSILON);
            }
        }

        @Test
        @DisplayName("rmsNorm applies scale")
        void rmsNormAppliesScale() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{3, 4}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{2, 0.5f}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = NormalizationOps.rmsNormVector(
                         input, weight, 1e-5f)) {

                float[] data = output.underlying().toFloatArray();

                float rms = (float) Math.sqrt((3*3 + 4*4) / 2.0);
                assertEquals(2 * 3 / rms, data[0], EPSILON);
                assertEquals(0.5f * 4 / rms, data[1], EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("RMSNorm on Rank3")
    class RmsNormRank3Tests {

        @Test
        @DisplayName("rmsNorm normalizes along hidden dimension")
        void rmsNormNormalizesAlongHiddenDimension() {
            // Shape: [batch=1, seq=2, hidden=2]
            // Vectors: [3, 4] and [6, 8]
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{3, 4, 6, 8}, new Rank3(1, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{1, 1}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> output = NormalizationOps.rmsNorm(
                         input, weight, 1e-5f)) {

                assertArrayEquals(new int[]{1, 2, 2}, output.dimensions());

                float[] data = output.underlying().toFloatArray();

                // First vector: RMS = sqrt((9 + 16) / 2) = sqrt(12.5)
                float rms1 = (float) Math.sqrt(12.5);
                assertEquals(3 / rms1, data[0], EPSILON);
                assertEquals(4 / rms1, data[1], EPSILON);

                // Second vector: RMS = sqrt((36 + 64) / 2) = sqrt(50)
                float rms2 = (float) Math.sqrt(50);
                assertEquals(6 / rms2, data[2], EPSILON);
                assertEquals(8 / rms2, data[3], EPSILON);
            }
        }

        @Test
        @DisplayName("rmsNormNoScale works without scale")
        void rmsNormNoScaleWorksWithoutScale() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{3, 4, 6, 8}, new Rank3(1, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> output = NormalizationOps.rmsNormNoScale(
                         input, 1e-5f)) {

                assertArrayEquals(new int[]{1, 2, 2}, output.dimensions());

                float[] data = output.underlying().toFloatArray();

                // Verify normalization without scale
                float rms1 = (float) Math.sqrt(12.5);
                assertEquals(3 / rms1, data[0], EPSILON);
                assertEquals(4 / rms1, data[1], EPSILON);
            }
        }

        @Test
        @DisplayName("rmsNorm preserves shape")
        void rmsNormPreservesShape() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.zeros(
                    new Rank3(4, 8, 64), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.zeros(
                    new Vector(64), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> output = NormalizationOps.rmsNorm(
                         input, weight, 1e-5f)) {

                assertArrayEquals(new int[]{4, 8, 64}, output.dimensions());
            }
        }

        @Test
        @DisplayName("rmsNorm rejects weight dimension mismatch")
        void rmsNormRejectsWeightDimensionMismatch() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.zeros(
                    new Rank3(2, 4, 8), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.zeros(
                    new Vector(16), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> NormalizationOps.rmsNorm(input, weight, 1e-5f));
            }
        }
    }

    @Nested
    @DisplayName("Numerical Stability")
    class NumericalStabilityTests {

        @Test
        @DisplayName("layerNorm handles very small epsilon")
        void layerNormHandlesVerySmallEpsilon() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{1, 1, 1}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.fromFloatArray(
                    new float[]{0, 0, 0}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = NormalizationOps.layerNormVector(
                         input, weight, bias, 1e-12f)) {

                float[] data = output.underlying().toFloatArray();
                for (float v : data) {
                    assertTrue(Float.isFinite(v), "Output should be finite");
                }
            }
        }

        @Test
        @DisplayName("layerNorm handles uniform input (zero variance)")
        void layerNormHandlesUniformInput() {
            // When all values are the same, variance = 0
            // LayerNorm should still work due to epsilon
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{5, 5, 5, 5}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{1, 1, 1, 1}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.fromFloatArray(
                    new float[]{0, 0, 0, 0}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = NormalizationOps.layerNormVector(
                         input, weight, bias, 1e-5f)) {

                float[] data = output.underlying().toFloatArray();
                for (float v : data) {
                    assertTrue(Float.isFinite(v), "Output should be finite");
                    // With zero variance, all normalized values should be near zero
                    assertEquals(0, v, 0.01f);
                }
            }
        }

        @Test
        @DisplayName("rmsNorm handles near-zero input")
        void rmsNormHandlesNearZeroInput() {
            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    new float[]{1e-10f, 1e-10f}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{1, 1}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> output = NormalizationOps.rmsNormVector(
                         input, weight, 1e-5f)) {

                float[] data = output.underlying().toFloatArray();
                for (float v : data) {
                    assertTrue(Float.isFinite(v), "Output should be finite");
                }
            }
        }
    }

    @Nested
    @DisplayName("Comparison with LayerNorm and RMSNorm")
    class ComparisonTests {

        @Test
        @DisplayName("layerNorm and rmsNorm differ in centering")
        void layerNormAndRmsNormDifferInCentering() {
            // LayerNorm subtracts mean, RMSNorm doesn't
            float[] inputData = {1, 2, 3, 4};

            try (TypedTensor<Vector, F32, Cpu> input = TypedTensor.fromFloatArray(
                    inputData, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> weight = TypedTensor.fromFloatArray(
                    new float[]{1, 1, 1, 1}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> bias = TypedTensor.fromFloatArray(
                    new float[]{0, 0, 0, 0}, new Vector(4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> layerNormed = NormalizationOps.layerNormVector(
                         input, weight, bias, 1e-5f);
                 TypedTensor<Vector, F32, Cpu> rmsNormed = NormalizationOps.rmsNormVector(
                         input, weight, 1e-5f)) {

                float[] lnData = layerNormed.underlying().toFloatArray();
                float[] rmsData = rmsNormed.underlying().toFloatArray();

                // LayerNorm output has mean ≈ 0
                float lnMean = (lnData[0] + lnData[1] + lnData[2] + lnData[3]) / 4;
                assertEquals(0, lnMean, EPSILON);

                // RMSNorm output does NOT have mean 0 (it preserves the sign/magnitude structure)
                float rmsMean = (rmsData[0] + rmsData[1] + rmsData[2] + rmsData[3]) / 4;
                assertTrue(Math.abs(rmsMean) > 0.1f, "RMSNorm should not center the data");
            }
        }
    }
}
