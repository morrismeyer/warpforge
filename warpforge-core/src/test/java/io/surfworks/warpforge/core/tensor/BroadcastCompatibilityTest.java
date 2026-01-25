package io.surfworks.warpforge.core.tensor;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Comprehensive tests for tensor broadcast compatibility.
 * Broadcasting rules follow NumPy/PyTorch semantics:
 * - Dimensions are compared from right to left (trailing dimensions first)
 * - Dimensions are compatible if they are equal, or one of them is 1
 * - Missing dimensions are treated as size 1
 */
@DisplayName("Broadcast Compatibility")
class BroadcastCompatibilityTest {

    private static TensorSpec spec(int... shape) {
        return TensorSpec.of(ScalarType.F32, shape);
    }

    @Nested
    @DisplayName("Compatible Broadcasts - Same Rank")
    class CompatibleSameRank {

        @Test
        @DisplayName("[1,3] and [2,3] are broadcastable -> [2,3]")
        void broadcast_1x3_with_2x3() {
            TensorSpec a = spec(1, 3);
            TensorSpec b = spec(2, 3);
            assertTrue(a.isBroadcastableWith(b));
            assertTrue(b.isBroadcastableWith(a)); // Symmetric
        }

        @Test
        @DisplayName("[2,1] and [1,3] are broadcastable -> [2,3]")
        void broadcast_2x1_with_1x3() {
            TensorSpec a = spec(2, 1);
            TensorSpec b = spec(1, 3);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[2,1] and [2,3] are broadcastable -> [2,3]")
        void broadcast_2x1_with_2x3() {
            TensorSpec a = spec(2, 1);
            TensorSpec b = spec(2, 3);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[1,1] and [2,3] are broadcastable -> [2,3]")
        void broadcast_1x1_with_2x3() {
            TensorSpec a = spec(1, 1);
            TensorSpec b = spec(2, 3);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[2,3,4] and [2,3,4] are broadcastable (same shape)")
        void broadcast_same_shape() {
            TensorSpec a = spec(2, 3, 4);
            TensorSpec b = spec(2, 3, 4);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[1,1,1] and [2,3,4] are broadcastable -> [2,3,4]")
        void broadcast_all_ones_with_3d() {
            TensorSpec a = spec(1, 1, 1);
            TensorSpec b = spec(2, 3, 4);
            assertTrue(a.isBroadcastableWith(b));
        }
    }

    @Nested
    @DisplayName("Compatible Broadcasts - Different Rank")
    class CompatibleDifferentRank {

        @Test
        @DisplayName("scalar and [2,3,4] are broadcastable")
        void broadcast_scalar_with_3d() {
            TensorSpec scalar = spec(); // Scalar
            TensorSpec b = spec(2, 3, 4);
            assertTrue(scalar.isBroadcastableWith(b));
            assertTrue(b.isBroadcastableWith(scalar));
        }

        @Test
        @DisplayName("[3] and [2,3] are broadcastable -> [2,3]")
        void broadcast_1d_with_2d() {
            TensorSpec a = spec(3);
            TensorSpec b = spec(2, 3);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[4] and [2,3,4] are broadcastable -> [2,3,4]")
        void broadcast_1d_with_3d() {
            TensorSpec a = spec(4);
            TensorSpec b = spec(2, 3, 4);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[3,4] and [2,3,4] are broadcastable -> [2,3,4]")
        void broadcast_2d_with_3d() {
            TensorSpec a = spec(3, 4);
            TensorSpec b = spec(2, 3, 4);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[1] and [2,3,4,5] are broadcastable")
        void broadcast_1d_one_with_4d() {
            TensorSpec a = spec(1);
            TensorSpec b = spec(2, 3, 4, 5);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[1,4] and [2,3,4] are broadcastable -> [2,3,4]")
        void broadcast_2d_with_leading_one_and_3d() {
            TensorSpec a = spec(1, 4);
            TensorSpec b = spec(2, 3, 4);
            assertTrue(a.isBroadcastableWith(b));
        }
    }

    @Nested
    @DisplayName("Incompatible Broadcasts")
    class IncompatibleBroadcasts {

        @Test
        @DisplayName("[2,3] and [4,5] are NOT broadcastable")
        void incompatible_different_shapes() {
            TensorSpec a = spec(2, 3);
            TensorSpec b = spec(4, 5);
            assertFalse(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[2,3] and [2,4] are NOT broadcastable (trailing dimension mismatch)")
        void incompatible_trailing_mismatch() {
            TensorSpec a = spec(2, 3);
            TensorSpec b = spec(2, 4);
            assertFalse(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[3,4] and [2,4] are NOT broadcastable (leading dimension mismatch)")
        void incompatible_leading_mismatch() {
            TensorSpec a = spec(3, 4);
            TensorSpec b = spec(2, 4);
            assertFalse(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[2,3,4] and [2,5,4] are NOT broadcastable (middle dimension mismatch)")
        void incompatible_middle_mismatch() {
            TensorSpec a = spec(2, 3, 4);
            TensorSpec b = spec(2, 5, 4);
            assertFalse(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[3] and [2,4] are NOT broadcastable (1D trailing mismatch)")
        void incompatible_1d_trailing_mismatch() {
            TensorSpec a = spec(3);
            TensorSpec b = spec(2, 4);
            assertFalse(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[5] and [2,3,4] are NOT broadcastable (1D doesn't match trailing)")
        void incompatible_1d_doesnt_match_trailing() {
            TensorSpec a = spec(5);
            TensorSpec b = spec(2, 3, 4);
            assertFalse(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("[2,3] and [3,2] are NOT broadcastable (transposed)")
        void incompatible_transposed() {
            TensorSpec a = spec(2, 3);
            TensorSpec b = spec(3, 2);
            assertFalse(a.isBroadcastableWith(b));
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {

        @Test
        @DisplayName("Empty tensor [0] and [2,3] are broadcastable")
        void empty_tensor_broadcastable() {
            TensorSpec a = spec(0);
            TensorSpec b = spec(2, 3);
            // Empty tensors follow same broadcast rules
            // [0] matches with [3] because trailing dims: 0 vs 3 - NOT compatible unless one is 1
            assertFalse(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("Empty tensor [0] and [2,0] are broadcastable")
        void empty_tensors_broadcastable() {
            TensorSpec a = spec(0);
            TensorSpec b = spec(2, 0);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("Empty tensor [1,0] and [2,0] are broadcastable")
        void empty_tensors_with_one() {
            TensorSpec a = spec(1, 0);
            TensorSpec b = spec(2, 0);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("Single element [1] and [1] are broadcastable")
        void single_element_broadcastable() {
            TensorSpec a = spec(1);
            TensorSpec b = spec(1);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("Two scalars are broadcastable")
        void two_scalars_broadcastable() {
            TensorSpec a = spec();
            TensorSpec b = spec();
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("Tensor is broadcastable with itself")
        void self_broadcast() {
            TensorSpec a = spec(2, 3, 4);
            assertTrue(a.isBroadcastableWith(a));
        }

        @Test
        @DisplayName("High-dimensional broadcast [1,1,1,1] with [2,3,4,5]")
        void high_dimensional_broadcast() {
            TensorSpec a = spec(1, 1, 1, 1);
            TensorSpec b = spec(2, 3, 4, 5);
            assertTrue(a.isBroadcastableWith(b));
        }

        @Test
        @DisplayName("Broadcast with very different ranks")
        void very_different_ranks() {
            TensorSpec a = spec(5);
            TensorSpec b = spec(1, 2, 3, 4, 5);
            assertTrue(a.isBroadcastableWith(b));
        }
    }

    @Nested
    @DisplayName("Common ML Tensor Shapes")
    class CommonMLShapes {

        @Test
        @DisplayName("Batch broadcast: [N,C,H,W] with [1,C,1,1]")
        void batch_broadcast_nchw_with_channel_bias() {
            TensorSpec features = spec(32, 64, 28, 28);  // [N, C, H, W]
            TensorSpec bias = spec(1, 64, 1, 1);         // Channel bias
            assertTrue(features.isBroadcastableWith(bias));
        }

        @Test
        @DisplayName("Batch broadcast: [N,C,H,W] with [C,1,1]")
        void batch_broadcast_nchw_with_channel_only() {
            TensorSpec features = spec(32, 64, 28, 28);
            TensorSpec channelScale = spec(64, 1, 1);
            assertTrue(features.isBroadcastableWith(channelScale));
        }

        @Test
        @DisplayName("Attention mask: [B,1,1,S] with [B,H,S,S]")
        void attention_mask_broadcast() {
            TensorSpec mask = spec(4, 1, 1, 128);       // [B, 1, 1, S]
            TensorSpec attention = spec(4, 8, 128, 128); // [B, H, S, S]
            assertTrue(mask.isBroadcastableWith(attention));
        }

        @Test
        @DisplayName("Matrix-vector: [M,N] with [N]")
        void matrix_vector_broadcast() {
            TensorSpec matrix = spec(64, 128);
            TensorSpec vector = spec(128);
            assertTrue(matrix.isBroadcastableWith(vector));
        }

        @Test
        @DisplayName("Matrix-vector incompatible: [M,N] with [M]")
        void matrix_vector_incompatible() {
            TensorSpec matrix = spec(64, 128);
            TensorSpec wrongVector = spec(64);  // Wrong dimension
            assertFalse(matrix.isBroadcastableWith(wrongVector));
        }

        @Test
        @DisplayName("Batch matmul: [B,M,K] with [K,N] after expansion")
        void batch_matmul_broadcast() {
            // In batch matmul, the non-batch dims must be compatible
            // This tests the broadcast of batch dimension
            TensorSpec batched = spec(8, 64, 32);  // [B, M, K]
            TensorSpec single = spec(64, 32);      // [M, K]
            assertTrue(batched.isBroadcastableWith(single));
        }
    }

    @Nested
    @DisplayName("Symmetry Property")
    class SymmetryProperty {

        @Test
        @DisplayName("Broadcast compatibility is symmetric")
        void symmetry_basic() {
            TensorSpec a = spec(1, 3);
            TensorSpec b = spec(2, 3);
            assertTrue(a.isBroadcastableWith(b) == b.isBroadcastableWith(a));
        }

        @Test
        @DisplayName("Incompatibility is also symmetric")
        void symmetry_incompatible() {
            TensorSpec a = spec(2, 3);
            TensorSpec b = spec(4, 5);
            assertFalse(a.isBroadcastableWith(b));
            assertFalse(b.isBroadcastableWith(a));
        }

        @Test
        @DisplayName("Different rank symmetry")
        void symmetry_different_rank() {
            TensorSpec a = spec(3);
            TensorSpec b = spec(2, 3);
            assertTrue(a.isBroadcastableWith(b) == b.isBroadcastableWith(a));
        }
    }
}
