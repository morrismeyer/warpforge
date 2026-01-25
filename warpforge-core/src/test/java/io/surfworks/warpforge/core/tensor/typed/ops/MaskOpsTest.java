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
import io.surfworks.warpforge.core.tensor.typed.shape.Rank4;

/**
 * Tests for MaskOps attention mask operations.
 */
@DisplayName("MaskOps")
class MaskOpsTest {

    private static final float EPSILON = 1e-5f;
    private static final float MASK_VALUE = -1e9f;

    @Nested
    @DisplayName("Causal Mask")
    class CausalMaskTests {

        @Test
        @DisplayName("causalMask creates lower triangular pattern")
        void causalMaskCreatesLowerTriangularPattern() {
            try (TypedTensor<Matrix, F32, Cpu> mask = MaskOps.causalMask(4, Cpu.INSTANCE)) {
                assertArrayEquals(new int[]{4, 4}, mask.dimensions());

                float[] data = mask.underlying().toFloatArray();

                // Row 0: [0, -inf, -inf, -inf] - can only attend to pos 0
                assertEquals(0.0f, data[0], EPSILON);
                assertEquals(MASK_VALUE, data[1], EPSILON);
                assertEquals(MASK_VALUE, data[2], EPSILON);
                assertEquals(MASK_VALUE, data[3], EPSILON);

                // Row 1: [0, 0, -inf, -inf] - can attend to pos 0, 1
                assertEquals(0.0f, data[4], EPSILON);
                assertEquals(0.0f, data[5], EPSILON);
                assertEquals(MASK_VALUE, data[6], EPSILON);
                assertEquals(MASK_VALUE, data[7], EPSILON);

                // Row 2: [0, 0, 0, -inf] - can attend to pos 0, 1, 2
                assertEquals(0.0f, data[8], EPSILON);
                assertEquals(0.0f, data[9], EPSILON);
                assertEquals(0.0f, data[10], EPSILON);
                assertEquals(MASK_VALUE, data[11], EPSILON);

                // Row 3: [0, 0, 0, 0] - can attend to all
                assertEquals(0.0f, data[12], EPSILON);
                assertEquals(0.0f, data[13], EPSILON);
                assertEquals(0.0f, data[14], EPSILON);
                assertEquals(0.0f, data[15], EPSILON);
            }
        }

        @Test
        @DisplayName("causalMask diagonal is always zero")
        void causalMaskDiagonalIsAlwaysZero() {
            try (TypedTensor<Matrix, F32, Cpu> mask = MaskOps.causalMask(8, Cpu.INSTANCE)) {
                float[] data = mask.underlying().toFloatArray();

                for (int i = 0; i < 8; i++) {
                    assertEquals(0.0f, data[i * 8 + i], EPSILON, "Diagonal [" + i + "," + i + "] should be 0");
                }
            }
        }

        @Test
        @DisplayName("causalMask upper triangle is masked")
        void causalMaskUpperTriangleIsMasked() {
            try (TypedTensor<Matrix, F32, Cpu> mask = MaskOps.causalMask(5, Cpu.INSTANCE)) {
                float[] data = mask.underlying().toFloatArray();

                for (int i = 0; i < 5; i++) {
                    for (int j = i + 1; j < 5; j++) {
                        assertEquals(MASK_VALUE, data[i * 5 + j], EPSILON,
                                "Position [" + i + "," + j + "] should be masked (upper triangle)");
                    }
                }
            }
        }

        @Test
        @DisplayName("causalMaskRank4 creates broadcastable shape")
        void causalMaskRank4CreatesBroadcastableShape() {
            try (TypedTensor<Rank4, F32, Cpu> mask = MaskOps.causalMaskRank4(6, Cpu.INSTANCE)) {
                assertArrayEquals(new int[]{1, 1, 6, 6}, mask.dimensions());

                // Check pattern is same as 2D causal mask
                float[] data = mask.underlying().toFloatArray();
                assertEquals(0.0f, data[0], EPSILON);  // [0,0]
                assertEquals(MASK_VALUE, data[1], EPSILON);  // [0,1]
                assertEquals(0.0f, data[6], EPSILON);  // [1,0]
                assertEquals(0.0f, data[7], EPSILON);  // [1,1]
            }
        }

        @Test
        @DisplayName("causalMask size 1 has single valid position")
        void causalMaskSize1HasSingleValidPosition() {
            try (TypedTensor<Matrix, F32, Cpu> mask = MaskOps.causalMask(1, Cpu.INSTANCE)) {
                assertArrayEquals(new int[]{1, 1}, mask.dimensions());
                assertEquals(0.0f, mask.underlying().toFloatArray()[0], EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Padding Mask")
    class PaddingMaskTests {

        @Test
        @DisplayName("paddingMask masks positions beyond sequence length")
        void paddingMaskMasksPositionsBeyondSequenceLength() {
            int[] lengths = {3, 5, 2};
            int maxLen = 6;

            try (TypedTensor<Matrix, F32, Cpu> mask = MaskOps.paddingMask(lengths, maxLen, Cpu.INSTANCE)) {
                assertArrayEquals(new int[]{3, 6}, mask.dimensions());

                float[] data = mask.underlying().toFloatArray();

                // Batch 0, length 3: [0, 0, 0, -inf, -inf, -inf]
                assertEquals(0.0f, data[0], EPSILON);
                assertEquals(0.0f, data[1], EPSILON);
                assertEquals(0.0f, data[2], EPSILON);
                assertEquals(MASK_VALUE, data[3], EPSILON);
                assertEquals(MASK_VALUE, data[4], EPSILON);
                assertEquals(MASK_VALUE, data[5], EPSILON);

                // Batch 1, length 5: [0, 0, 0, 0, 0, -inf]
                assertEquals(0.0f, data[6], EPSILON);
                assertEquals(0.0f, data[10], EPSILON);
                assertEquals(MASK_VALUE, data[11], EPSILON);

                // Batch 2, length 2: [0, 0, -inf, -inf, -inf, -inf]
                assertEquals(0.0f, data[12], EPSILON);
                assertEquals(0.0f, data[13], EPSILON);
                assertEquals(MASK_VALUE, data[14], EPSILON);
            }
        }

        @Test
        @DisplayName("paddingMask full length sequence has no masking")
        void paddingMaskFullLengthSequenceHasNoMasking() {
            int[] lengths = {4};
            int maxLen = 4;

            try (TypedTensor<Matrix, F32, Cpu> mask = MaskOps.paddingMask(lengths, maxLen, Cpu.INSTANCE)) {
                float[] data = mask.underlying().toFloatArray();

                for (int i = 0; i < 4; i++) {
                    assertEquals(0.0f, data[i], EPSILON, "Position " + i + " should not be masked");
                }
            }
        }

        @Test
        @DisplayName("paddingMask zero length masks everything")
        void paddingMaskZeroLengthMasksEverything() {
            int[] lengths = {0};
            int maxLen = 3;

            try (TypedTensor<Matrix, F32, Cpu> mask = MaskOps.paddingMask(lengths, maxLen, Cpu.INSTANCE)) {
                float[] data = mask.underlying().toFloatArray();

                for (int i = 0; i < 3; i++) {
                    assertEquals(MASK_VALUE, data[i], EPSILON, "Position " + i + " should be masked");
                }
            }
        }

        @Test
        @DisplayName("paddingMaskRank4 creates broadcastable shape")
        void paddingMaskRank4CreatesBroadcastableShape() {
            int[] lengths = {2, 4};
            int maxLen = 5;

            try (TypedTensor<Rank4, F32, Cpu> mask = MaskOps.paddingMaskRank4(lengths, maxLen, Cpu.INSTANCE)) {
                assertArrayEquals(new int[]{2, 1, 1, 5}, mask.dimensions());

                float[] data = mask.underlying().toFloatArray();

                // Batch 0, length 2: [0, 0, -inf, -inf, -inf]
                assertEquals(0.0f, data[0], EPSILON);
                assertEquals(0.0f, data[1], EPSILON);
                assertEquals(MASK_VALUE, data[2], EPSILON);

                // Batch 1, length 4: [0, 0, 0, 0, -inf]
                assertEquals(0.0f, data[5], EPSILON);
                assertEquals(0.0f, data[8], EPSILON);
                assertEquals(MASK_VALUE, data[9], EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("From Attention Mask")
    class FromAttentionMaskTests {

        @Test
        @DisplayName("fromAttentionMask converts boolean mask")
        void fromAttentionMaskConvertsBooleanMask() {
            // Input: 1.0 = valid, 0.0 = padding
            float[] input = {1, 1, 1, 0, 0, 1, 1, 0, 0, 0};

            try (TypedTensor<Matrix, F32, Cpu> attMask = TypedTensor.fromFloatArray(
                    input, new Matrix(2, 5), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> result = MaskOps.fromAttentionMask(attMask)) {

                assertArrayEquals(new int[]{2, 5}, result.dimensions());

                float[] data = result.underlying().toFloatArray();

                // Batch 0: [valid, valid, valid, pad, pad]
                assertEquals(0.0f, data[0], EPSILON);
                assertEquals(0.0f, data[1], EPSILON);
                assertEquals(0.0f, data[2], EPSILON);
                assertEquals(MASK_VALUE, data[3], EPSILON);
                assertEquals(MASK_VALUE, data[4], EPSILON);

                // Batch 1: [valid, valid, pad, pad, pad]
                assertEquals(0.0f, data[5], EPSILON);
                assertEquals(0.0f, data[6], EPSILON);
                assertEquals(MASK_VALUE, data[7], EPSILON);
                assertEquals(MASK_VALUE, data[8], EPSILON);
                assertEquals(MASK_VALUE, data[9], EPSILON);
            }
        }

        @Test
        @DisplayName("fromAttentionMask handles all valid")
        void fromAttentionMaskHandlesAllValid() {
            float[] input = {1, 1, 1, 1};

            try (TypedTensor<Matrix, F32, Cpu> attMask = TypedTensor.fromFloatArray(
                    input, new Matrix(1, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> result = MaskOps.fromAttentionMask(attMask)) {

                float[] data = result.underlying().toFloatArray();
                for (int i = 0; i < 4; i++) {
                    assertEquals(0.0f, data[i], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("fromAttentionMask handles all padding")
        void fromAttentionMaskHandlesAllPadding() {
            float[] input = {0, 0, 0};

            try (TypedTensor<Matrix, F32, Cpu> attMask = TypedTensor.fromFloatArray(
                    input, new Matrix(1, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> result = MaskOps.fromAttentionMask(attMask)) {

                float[] data = result.underlying().toFloatArray();
                for (int i = 0; i < 3; i++) {
                    assertEquals(MASK_VALUE, data[i], EPSILON);
                }
            }
        }
    }

    @Nested
    @DisplayName("Combined Causal + Padding Mask")
    class CausalPaddingMaskTests {

        @Test
        @DisplayName("causalPaddingMask combines both constraints")
        void causalPaddingMaskCombinesBothConstraints() {
            int[] lengths = {3};
            int maxLen = 4;

            try (TypedTensor<Rank4, F32, Cpu> mask = MaskOps.causalPaddingMask(lengths, maxLen, Cpu.INSTANCE)) {
                assertArrayEquals(new int[]{1, 1, 4, 4}, mask.dimensions());

                float[] data = mask.underlying().toFloatArray();

                // Row 0: can attend to pos 0 (causal), pos 0-2 valid (padding)
                // So only pos 0 is valid
                assertEquals(0.0f, data[0], EPSILON);
                assertEquals(MASK_VALUE, data[1], EPSILON);
                assertEquals(MASK_VALUE, data[2], EPSILON);
                assertEquals(MASK_VALUE, data[3], EPSILON);

                // Row 1: can attend to pos 0-1 (causal), pos 0-2 valid (padding)
                // So pos 0-1 are valid
                assertEquals(0.0f, data[4], EPSILON);
                assertEquals(0.0f, data[5], EPSILON);
                assertEquals(MASK_VALUE, data[6], EPSILON);
                assertEquals(MASK_VALUE, data[7], EPSILON);

                // Row 2: can attend to pos 0-2 (causal), pos 0-2 valid (padding)
                // So pos 0-2 are valid
                assertEquals(0.0f, data[8], EPSILON);
                assertEquals(0.0f, data[9], EPSILON);
                assertEquals(0.0f, data[10], EPSILON);
                assertEquals(MASK_VALUE, data[11], EPSILON);

                // Row 3: can attend to pos 0-3 (causal), but pos 3 is padding
                // So pos 0-2 are valid
                assertEquals(0.0f, data[12], EPSILON);
                assertEquals(0.0f, data[13], EPSILON);
                assertEquals(0.0f, data[14], EPSILON);
                assertEquals(MASK_VALUE, data[15], EPSILON);
            }
        }

        @Test
        @DisplayName("causalPaddingMask handles full length sequence")
        void causalPaddingMaskHandlesFullLengthSequence() {
            int[] lengths = {3};
            int maxLen = 3;

            try (TypedTensor<Rank4, F32, Cpu> mask = MaskOps.causalPaddingMask(lengths, maxLen, Cpu.INSTANCE)) {
                float[] data = mask.underlying().toFloatArray();

                // Should be identical to pure causal mask
                // Row 0: [0, -inf, -inf]
                assertEquals(0.0f, data[0], EPSILON);
                assertEquals(MASK_VALUE, data[1], EPSILON);
                assertEquals(MASK_VALUE, data[2], EPSILON);

                // Row 1: [0, 0, -inf]
                assertEquals(0.0f, data[3], EPSILON);
                assertEquals(0.0f, data[4], EPSILON);
                assertEquals(MASK_VALUE, data[5], EPSILON);

                // Row 2: [0, 0, 0]
                assertEquals(0.0f, data[6], EPSILON);
                assertEquals(0.0f, data[7], EPSILON);
                assertEquals(0.0f, data[8], EPSILON);
            }
        }

        @Test
        @DisplayName("causalPaddingMask handles multiple batches")
        void causalPaddingMaskHandlesMultipleBatches() {
            int[] lengths = {2, 3};
            int maxLen = 3;

            try (TypedTensor<Rank4, F32, Cpu> mask = MaskOps.causalPaddingMask(lengths, maxLen, Cpu.INSTANCE)) {
                assertArrayEquals(new int[]{2, 1, 3, 3}, mask.dimensions());

                float[] data = mask.underlying().toFloatArray();

                // Batch 0, length 2:
                // Row 2: can attend to 0-2 causally but only 0-1 are valid
                assertEquals(0.0f, data[6], EPSILON);  // pos 0
                assertEquals(0.0f, data[7], EPSILON);  // pos 1
                assertEquals(MASK_VALUE, data[8], EPSILON);  // pos 2 (padding)

                // Batch 1, length 3:
                // Row 2: can attend to 0-2 causally, all valid
                assertEquals(0.0f, data[15], EPSILON);  // pos 0
                assertEquals(0.0f, data[16], EPSILON);  // pos 1
                assertEquals(0.0f, data[17], EPSILON);  // pos 2
            }
        }
    }

    @Nested
    @DisplayName("Add Causal Mask")
    class AddCausalMaskTests {

        @Test
        @DisplayName("addCausalMask adds causal constraints")
        void addCausalMaskAddsCausalConstraints() {
            // Start with zeros (no masking)
            float[] input = new float[4];  // [1, 1, 2, 2]

            try (TypedTensor<Rank4, F32, Cpu> mask = TypedTensor.fromFloatArray(
                    input, new Rank4(1, 1, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> result = MaskOps.addCausalMask(mask)) {

                float[] data = result.underlying().toFloatArray();

                // Row 0: [0+0, 0+(-inf)] = [0, -inf]
                assertEquals(0.0f, data[0], EPSILON);
                assertEquals(MASK_VALUE, data[1], EPSILON);

                // Row 1: [0+0, 0+0] = [0, 0]
                assertEquals(0.0f, data[2], EPSILON);
                assertEquals(0.0f, data[3], EPSILON);
            }
        }

        @Test
        @DisplayName("addCausalMask preserves existing masking")
        void addCausalMaskPreservesExistingMasking() {
            // Start with some existing mask values
            float[] input = {0, MASK_VALUE, MASK_VALUE, MASK_VALUE};

            try (TypedTensor<Rank4, F32, Cpu> mask = TypedTensor.fromFloatArray(
                    input, new Rank4(1, 1, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> result = MaskOps.addCausalMask(mask)) {

                float[] data = result.underlying().toFloatArray();

                // Position [0,1] was already masked, causal adds more masking
                // -1e9 + -1e9 = -2e9
                assertTrue(data[1] < -1.9e9f);

                // Position [1,0] was masked but causal allows it
                // -1e9 + 0 = -1e9
                assertEquals(MASK_VALUE, data[2], EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Apply Mask")
    class ApplyMaskTests {

        @Test
        @DisplayName("applyMask adds mask to scores")
        void applyMaskAddsMaskToScores() {
            float[] scores = {1, 2, 3, 4};  // [1, 1, 2, 2]
            float[] maskData = {0, MASK_VALUE, 0, 0};  // Same shape

            try (TypedTensor<Rank4, F32, Cpu> scoresTensor = TypedTensor.fromFloatArray(
                    scores, new Rank4(1, 1, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> mask = TypedTensor.fromFloatArray(
                    maskData, new Rank4(1, 1, 2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> result = MaskOps.applyMask(scoresTensor, mask)) {

                float[] data = result.underlying().toFloatArray();

                assertEquals(1.0f, data[0], EPSILON);  // 1 + 0
                assertEquals(2.0f + MASK_VALUE, data[1], EPSILON);  // 2 + (-1e9)
                assertEquals(3.0f, data[2], EPSILON);  // 3 + 0
                assertEquals(4.0f, data[3], EPSILON);  // 4 + 0
            }
        }

        @Test
        @DisplayName("applyMask broadcasts mask dimensions")
        void applyMaskBroadcastsMaskDimensions() {
            // Scores: [2, 2, 2, 3]
            float[] scores = new float[24];
            for (int i = 0; i < 24; i++) {
                scores[i] = 1.0f;
            }

            // Mask: [1, 1, 1, 3] - broadcasts to all batches, heads, and query positions
            float[] maskData = {0, MASK_VALUE, 0};

            try (TypedTensor<Rank4, F32, Cpu> scoresTensor = TypedTensor.fromFloatArray(
                    scores, new Rank4(2, 2, 2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> mask = TypedTensor.fromFloatArray(
                    maskData, new Rank4(1, 1, 1, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> result = MaskOps.applyMask(scoresTensor, mask)) {

                float[] data = result.underlying().toFloatArray();

                // Check a few positions - position 1 in key dim should always be masked
                // First batch, first head, first query
                assertEquals(1.0f, data[0], EPSILON);  // k=0
                assertEquals(1.0f + MASK_VALUE, data[1], EPSILON);  // k=1
                assertEquals(1.0f, data[2], EPSILON);  // k=2

                // Second batch, second head, second query - same mask pattern
                int idx = 1 * 12 + 1 * 6 + 1 * 3;  // batch=1, head=1, q=1
                assertEquals(1.0f, data[idx], EPSILON);
                assertEquals(1.0f + MASK_VALUE, data[idx + 1], EPSILON);
                assertEquals(1.0f, data[idx + 2], EPSILON);
            }
        }

        @Test
        @DisplayName("applyMask rejects non-broadcastable shapes")
        void applyMaskRejectsNonBroadcastableShapes() {
            try (TypedTensor<Rank4, F32, Cpu> scores = TypedTensor.zeros(
                    new Rank4(2, 4, 8, 8), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> mask = TypedTensor.zeros(
                    new Rank4(3, 1, 1, 8), F32.INSTANCE, Cpu.INSTANCE)) {  // batch=3 != 2

                assertThrows(IllegalArgumentException.class,
                        () -> MaskOps.applyMask(scores, mask));
            }
        }
    }

    @Nested
    @DisplayName("Attention Pattern Integration")
    class AttentionPatternTests {

        @Test
        @DisplayName("masked attention score pattern is correct")
        void maskedAttentionScorePatternIsCorrect() {
            // Simulate attention scores before softmax
            int seqLen = 3;
            float[] scores = new float[9];
            for (int i = 0; i < 9; i++) {
                scores[i] = 1.0f;  // Uniform scores
            }

            try (TypedTensor<Matrix, F32, Cpu> causal = MaskOps.causalMask(seqLen, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> scoresR4 = TypedTensor.fromFloatArray(
                    scores, new Rank4(1, 1, 3, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> maskR4 = TypedTensor.fromFloatArray(
                    causal.underlying().toFloatArray(), new Rank4(1, 1, 3, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank4, F32, Cpu> masked = MaskOps.applyMask(scoresR4, maskR4)) {

                float[] data = masked.underlying().toFloatArray();

                // After masking:
                // Row 0: [1, -inf, -inf] - only first position valid
                assertEquals(1.0f, data[0], EPSILON);
                assertTrue(data[1] < -1e8f);
                assertTrue(data[2] < -1e8f);

                // Row 1: [1, 1, -inf] - first two positions valid
                assertEquals(1.0f, data[3], EPSILON);
                assertEquals(1.0f, data[4], EPSILON);
                assertTrue(data[5] < -1e8f);

                // Row 2: [1, 1, 1] - all positions valid
                assertEquals(1.0f, data[6], EPSILON);
                assertEquals(1.0f, data[7], EPSILON);
                assertEquals(1.0f, data[8], EPSILON);
            }
        }

        @Test
        @DisplayName("BERT-style padding mask blocks attention to padding")
        void bertStylePaddingMaskBlocksAttentionToPadding() {
            // BERT uses padding mask to prevent attending to [PAD] tokens
            int[] lengths = {3, 2};  // Two sequences with different lengths
            int maxLen = 4;

            try (TypedTensor<Rank4, F32, Cpu> paddingMask = MaskOps.paddingMaskRank4(lengths, maxLen, Cpu.INSTANCE)) {
                float[] data = paddingMask.underlying().toFloatArray();

                // Batch 0: [0, 0, 0, -inf] - can attend to first 3
                assertEquals(0.0f, data[0], EPSILON);
                assertEquals(0.0f, data[2], EPSILON);
                assertEquals(MASK_VALUE, data[3], EPSILON);

                // Batch 1: [0, 0, -inf, -inf] - can attend to first 2
                assertEquals(0.0f, data[4], EPSILON);
                assertEquals(0.0f, data[5], EPSILON);
                assertEquals(MASK_VALUE, data[6], EPSILON);
                assertEquals(MASK_VALUE, data[7], EPSILON);
            }
        }

        @Test
        @DisplayName("GPT-style causal+padding for variable-length generation")
        void gptStyleCausalPaddingForVariableLengthGeneration() {
            // GPT-2 with batched generation where sequences have different lengths
            int[] lengths = {2, 3};
            int maxLen = 3;

            try (TypedTensor<Rank4, F32, Cpu> mask = MaskOps.causalPaddingMask(lengths, maxLen, Cpu.INSTANCE)) {
                float[] data = mask.underlying().toFloatArray();

                // Batch 0, length 2:
                // Position 2 generating token: can attend to 0, 1 (causal + padding)
                // but NOT position 2 (it's padding in batch 0)
                int b0Row2 = 6;  // batch 0, row 2
                assertEquals(0.0f, data[b0Row2], EPSILON);      // pos 0 valid
                assertEquals(0.0f, data[b0Row2 + 1], EPSILON);  // pos 1 valid
                assertEquals(MASK_VALUE, data[b0Row2 + 2], EPSILON);  // pos 2 padding

                // Batch 1, length 3:
                // Position 2: can attend to all (causal allows, all valid)
                int b1Row2 = 9 + 6;  // batch 1, row 2
                assertEquals(0.0f, data[b1Row2], EPSILON);
                assertEquals(0.0f, data[b1Row2 + 1], EPSILON);
                assertEquals(0.0f, data[b1Row2 + 2], EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("large sequence length causal mask")
        void largeSequenceLengthCausalMask() {
            int seqLen = 128;
            try (TypedTensor<Matrix, F32, Cpu> mask = MaskOps.causalMask(seqLen, Cpu.INSTANCE)) {
                assertArrayEquals(new int[]{seqLen, seqLen}, mask.dimensions());

                float[] data = mask.underlying().toFloatArray();

                // Spot check: last row should be all zeros (can attend everywhere)
                int lastRowStart = (seqLen - 1) * seqLen;
                for (int j = 0; j < seqLen; j++) {
                    assertEquals(0.0f, data[lastRowStart + j], EPSILON);
                }

                // First row should only have first element as zero
                assertEquals(0.0f, data[0], EPSILON);
                assertEquals(MASK_VALUE, data[1], EPSILON);
            }
        }

        @Test
        @DisplayName("single element batch padding mask")
        void singleElementBatchPaddingMask() {
            int[] lengths = {1};
            int maxLen = 1;

            try (TypedTensor<Matrix, F32, Cpu> mask = MaskOps.paddingMask(lengths, maxLen, Cpu.INSTANCE)) {
                assertArrayEquals(new int[]{1, 1}, mask.dimensions());
                assertEquals(0.0f, mask.underlying().toFloatArray()[0], EPSILON);
            }
        }

        @Test
        @DisplayName("varied batch lengths padding mask")
        void variedBatchLengthsPaddingMask() {
            int[] lengths = {10, 1, 5, 10};
            int maxLen = 10;

            try (TypedTensor<Matrix, F32, Cpu> mask = MaskOps.paddingMask(lengths, maxLen, Cpu.INSTANCE)) {
                float[] data = mask.underlying().toFloatArray();

                // Batch 0: all valid
                for (int i = 0; i < 10; i++) {
                    assertEquals(0.0f, data[i], EPSILON);
                }

                // Batch 1: only first valid
                assertEquals(0.0f, data[10], EPSILON);
                for (int i = 11; i < 20; i++) {
                    assertEquals(MASK_VALUE, data[i], EPSILON);
                }

                // Batch 2: first 5 valid
                for (int i = 20; i < 25; i++) {
                    assertEquals(0.0f, data[i], EPSILON);
                }
                for (int i = 25; i < 30; i++) {
                    assertEquals(MASK_VALUE, data[i], EPSILON);
                }
            }
        }
    }
}
