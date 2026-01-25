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

/**
 * Tests for EmbeddingOps embedding operations.
 */
@DisplayName("EmbeddingOps")
class EmbeddingOpsTest {

    private static final float EPSILON = 1e-5f;

    @Nested
    @DisplayName("Token Embedding")
    class TokenEmbeddingTests {

        @Test
        @DisplayName("embedding looks up correct rows")
        void embeddingLooksUpCorrectRows() {
            // Embedding table: 4 tokens x 3 dims
            // [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
            float[] embedData = new float[12];
            for (int i = 0; i < 12; i++) {
                embedData[i] = i;
            }

            try (TypedTensor<Matrix, F32, Cpu> embeddings = TypedTensor.fromFloatArray(
                    embedData, new Matrix(4, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                // Batch=2, Seq=2
                int[][] indices = {{0, 3}, {2, 1}};

                try (TypedTensor<Rank3, F32, Cpu> output = EmbeddingOps.embedding(embeddings, indices)) {
                    assertArrayEquals(new int[]{2, 2, 3}, output.dimensions());

                    float[] data = output.underlying().toFloatArray();

                    // Batch 0, Token 0 (index 0): [0, 1, 2]
                    assertEquals(0, data[0], EPSILON);
                    assertEquals(1, data[1], EPSILON);
                    assertEquals(2, data[2], EPSILON);

                    // Batch 0, Token 1 (index 3): [9, 10, 11]
                    assertEquals(9, data[3], EPSILON);
                    assertEquals(10, data[4], EPSILON);
                    assertEquals(11, data[5], EPSILON);

                    // Batch 1, Token 0 (index 2): [6, 7, 8]
                    assertEquals(6, data[6], EPSILON);
                    assertEquals(7, data[7], EPSILON);
                    assertEquals(8, data[8], EPSILON);

                    // Batch 1, Token 1 (index 1): [3, 4, 5]
                    assertEquals(3, data[9], EPSILON);
                    assertEquals(4, data[10], EPSILON);
                    assertEquals(5, data[11], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("embeddingSingle looks up for single sequence")
        void embeddingSingleLooksUpForSingleSequence() {
            // Embedding table: 5 tokens x 2 dims
            float[] embedData = {10, 11, 20, 21, 30, 31, 40, 41, 50, 51};

            try (TypedTensor<Matrix, F32, Cpu> embeddings = TypedTensor.fromFloatArray(
                    embedData, new Matrix(5, 2), F32.INSTANCE, Cpu.INSTANCE)) {

                int[] indices = {4, 1, 3};

                try (TypedTensor<Matrix, F32, Cpu> output = EmbeddingOps.embeddingSingle(embeddings, indices)) {
                    assertArrayEquals(new int[]{3, 2}, output.dimensions());

                    float[] data = output.underlying().toFloatArray();

                    // Token 4: [50, 51]
                    assertEquals(50, data[0], EPSILON);
                    assertEquals(51, data[1], EPSILON);

                    // Token 1: [20, 21]
                    assertEquals(20, data[2], EPSILON);
                    assertEquals(21, data[3], EPSILON);

                    // Token 3: [40, 41]
                    assertEquals(40, data[4], EPSILON);
                    assertEquals(41, data[5], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("embedding handles repeated indices")
        void embeddingHandlesRepeatedIndices() {
            float[] embedData = {1, 2, 3, 4, 5, 6};

            try (TypedTensor<Matrix, F32, Cpu> embeddings = TypedTensor.fromFloatArray(
                    embedData, new Matrix(3, 2), F32.INSTANCE, Cpu.INSTANCE)) {

                int[][] indices = {{0, 0, 0}, {1, 1, 1}};

                try (TypedTensor<Rank3, F32, Cpu> output = EmbeddingOps.embedding(embeddings, indices)) {
                    float[] data = output.underlying().toFloatArray();

                    // All of batch 0 should be [1, 2]
                    for (int i = 0; i < 3; i++) {
                        assertEquals(1, data[i * 2], EPSILON);
                        assertEquals(2, data[i * 2 + 1], EPSILON);
                    }

                    // All of batch 1 should be [3, 4]
                    for (int i = 0; i < 3; i++) {
                        assertEquals(3, data[6 + i * 2], EPSILON);
                        assertEquals(4, data[6 + i * 2 + 1], EPSILON);
                    }
                }
            }
        }
    }

    @Nested
    @DisplayName("Position Embedding")
    class PositionEmbeddingTests {

        @Test
        @DisplayName("positionEmbedding extracts correct rows")
        void positionEmbeddingExtractsCorrectRows() {
            // Position table: 10 positions x 4 dims
            float[] posData = new float[40];
            for (int i = 0; i < 40; i++) {
                posData[i] = i;
            }

            try (TypedTensor<Matrix, F32, Cpu> posTable = TypedTensor.fromFloatArray(
                    posData, new Matrix(10, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> positions = EmbeddingOps.positionEmbedding(posTable, 3)) {

                assertArrayEquals(new int[]{3, 4}, positions.dimensions());

                float[] data = positions.underlying().toFloatArray();

                // Position 0: [0, 1, 2, 3]
                assertEquals(0, data[0], EPSILON);
                assertEquals(1, data[1], EPSILON);
                assertEquals(2, data[2], EPSILON);
                assertEquals(3, data[3], EPSILON);

                // Position 1: [4, 5, 6, 7]
                assertEquals(4, data[4], EPSILON);
                assertEquals(5, data[5], EPSILON);
            }
        }

        @Test
        @DisplayName("addPositionEmbedding adds positions to tokens")
        void addPositionEmbeddingAddsPositionsToTokens() {
            // Tokens: [2, 2, 3] with values all 1.0
            // Positions: [2, 3] with row-specific values
            float[] tokenData = new float[6];
            java.util.Arrays.fill(tokenData, 1.0f);

            float[] posData = {10, 11, 12, 20, 21, 22};

            try (TypedTensor<Rank3, F32, Cpu> tokens = TypedTensor.fromFloatArray(
                    tokenData, new Rank3(2, 1, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> positions = TypedTensor.fromFloatArray(
                    posData, new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                // Slice to get only seq=1
                try (TypedTensor<Matrix, F32, Cpu> pos1 = SliceOps.sliceRows(positions, 0, 1);
                     TypedTensor<Rank3, F32, Cpu> result = EmbeddingOps.addPositionEmbedding(tokens, pos1)) {

                    assertArrayEquals(new int[]{2, 1, 3}, result.dimensions());

                    float[] data = result.underlying().toFloatArray();

                    // Batch 0: tokens [1, 1, 1] + positions [10, 11, 12] = [11, 12, 13]
                    assertEquals(11, data[0], EPSILON);
                    assertEquals(12, data[1], EPSILON);
                    assertEquals(13, data[2], EPSILON);

                    // Batch 1: same positions added
                    assertEquals(11, data[3], EPSILON);
                    assertEquals(12, data[4], EPSILON);
                    assertEquals(13, data[5], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("addPositionEmbedding rejects shape mismatch")
        void addPositionEmbeddingRejectsShapeMismatch() {
            try (TypedTensor<Rank3, F32, Cpu> tokens = TypedTensor.zeros(
                    new Rank3(2, 4, 8), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> positions = TypedTensor.zeros(
                    new Matrix(5, 8), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> EmbeddingOps.addPositionEmbedding(tokens, positions));
            }
        }
    }

    @Nested
    @DisplayName("Combined Embedding")
    class CombinedEmbeddingTests {

        @Test
        @DisplayName("transformerEmbedding combines token and position")
        void transformerEmbeddingCombinesTokenAndPosition() {
            // Token table: 5 tokens x 3 dims
            float[] tokenData = new float[15];
            for (int i = 0; i < 15; i++) {
                tokenData[i] = i;
            }

            // Position table: 10 positions x 3 dims
            float[] posData = new float[30];
            for (int i = 0; i < 30; i++) {
                posData[i] = 100 + i;
            }

            try (TypedTensor<Matrix, F32, Cpu> tokenTable = TypedTensor.fromFloatArray(
                    tokenData, new Matrix(5, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> posTable = TypedTensor.fromFloatArray(
                    posData, new Matrix(10, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                int[][] indices = {{0, 2}};  // Single batch, 2 tokens

                try (TypedTensor<Rank3, F32, Cpu> output = EmbeddingOps.transformerEmbedding(
                        tokenTable, posTable, indices)) {

                    assertArrayEquals(new int[]{1, 2, 3}, output.dimensions());

                    float[] data = output.underlying().toFloatArray();

                    // Token 0 [0, 1, 2] + Position 0 [100, 101, 102] = [100, 102, 104]
                    assertEquals(100, data[0], EPSILON);
                    assertEquals(102, data[1], EPSILON);
                    assertEquals(104, data[2], EPSILON);

                    // Token 2 [6, 7, 8] + Position 1 [103, 104, 105] = [109, 111, 113]
                    assertEquals(109, data[3], EPSILON);
                    assertEquals(111, data[4], EPSILON);
                    assertEquals(113, data[5], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("learnedPositionEmbedding expands to batch")
        void learnedPositionEmbeddingExpandsToBatch() {
            float[] posData = new float[8];
            for (int i = 0; i < 8; i++) {
                posData[i] = i;
            }

            try (TypedTensor<Matrix, F32, Cpu> posTable = TypedTensor.fromFloatArray(
                    posData, new Matrix(4, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> output = EmbeddingOps.learnedPositionEmbedding(
                         posTable, 3, 2)) {

                assertArrayEquals(new int[]{3, 2, 2}, output.dimensions());

                float[] data = output.underlying().toFloatArray();

                // All 3 batches should have the same position embeddings
                // Position 0: [0, 1], Position 1: [2, 3]
                for (int b = 0; b < 3; b++) {
                    int offset = b * 4;
                    assertEquals(0, data[offset], EPSILON);
                    assertEquals(1, data[offset + 1], EPSILON);
                    assertEquals(2, data[offset + 2], EPSILON);
                    assertEquals(3, data[offset + 3], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("learnedPositionEmbedding rejects seq > max")
        void learnedPositionEmbeddingRejectsSeqGreaterThanMax() {
            try (TypedTensor<Matrix, F32, Cpu> posTable = TypedTensor.zeros(
                    new Matrix(10, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> EmbeddingOps.learnedPositionEmbedding(posTable, 2, 15));
            }
        }
    }

    @Nested
    @DisplayName("Sinusoidal Position Embedding")
    class SinusoidalPositionEmbeddingTests {

        @Test
        @DisplayName("sinusoidalPositionEmbedding has correct shape")
        void sinusoidalPositionEmbeddingHasCorrectShape() {
            try (TypedTensor<Matrix, F32, Cpu> sinusoidal = EmbeddingOps.sinusoidalPositionEmbedding(
                    128, 64, Cpu.INSTANCE)) {

                assertArrayEquals(new int[]{128, 64}, sinusoidal.dimensions());
            }
        }

        @Test
        @DisplayName("sinusoidalPositionEmbedding has alternating sin/cos")
        void sinusoidalPositionEmbeddingHasAlternatingSinCos() {
            try (TypedTensor<Matrix, F32, Cpu> sinusoidal = EmbeddingOps.sinusoidalPositionEmbedding(
                    10, 4, Cpu.INSTANCE)) {

                float[] data = sinusoidal.underlying().toFloatArray();

                // Position 0: sin(0) = 0, cos(0) = 1 for first pair
                assertEquals(0, data[0], EPSILON);  // sin(0)
                assertEquals(1, data[1], EPSILON);  // cos(0)

                // Position 1: sin(1 * divTerm) and cos(1 * divTerm) for first pair
                // divTerm for i=0 is exp(-log(10000) * 0 / 2) = 1
                // So we expect sin(1) ≈ 0.841 and cos(1) ≈ 0.540
                assertEquals((float) Math.sin(1), data[4], 0.01f);
                assertEquals((float) Math.cos(1), data[5], 0.01f);
            }
        }

        @Test
        @DisplayName("sinusoidalPositionEmbedding rejects odd dimension")
        void sinusoidalPositionEmbeddingRejectsOddDimension() {
            assertThrows(IllegalArgumentException.class,
                    () -> EmbeddingOps.sinusoidalPositionEmbedding(10, 5, Cpu.INSTANCE));
        }

        @Test
        @DisplayName("sinusoidalPositionEmbedding values are bounded")
        void sinusoidalPositionEmbeddingValuesAreBounded() {
            try (TypedTensor<Matrix, F32, Cpu> sinusoidal = EmbeddingOps.sinusoidalPositionEmbedding(
                    512, 256, Cpu.INSTANCE)) {

                float[] data = sinusoidal.underlying().toFloatArray();

                // All values should be in [-1, 1] since they're sin/cos
                for (float v : data) {
                    assertTrue(v >= -1.0f && v <= 1.0f,
                            "Sinusoidal values should be in [-1, 1], got " + v);
                }
            }
        }
    }
}
