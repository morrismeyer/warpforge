package io.surfworks.warpforge.core.tensor.typed.ops;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Tests for SliceOps slicing and indexing operations.
 */
@DisplayName("SliceOps")
class SliceOpsTest {

    private static final float EPSILON = 1e-5f;
    private static final double EPSILON_D = 1e-10;

    @Nested
    @DisplayName("Gather Operations")
    class GatherTests {

        @Test
        @DisplayName("gather performs embedding lookup")
        void gatherPerformsEmbeddingLookup() {
            // Embedding table: 5 tokens x 3 dims
            // [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
            float[] embeddingData = new float[15];
            for (int i = 0; i < 15; i++) {
                embeddingData[i] = i;
            }

            try (TypedTensor<Matrix, F32, Cpu> embeddings = TypedTensor.fromFloatArray(
                    embeddingData, new Matrix(5, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                // Indices: batch=2, seq=3
                int[][] indices = {{0, 2, 4}, {1, 3, 0}};

                try (TypedTensor<Rank3, F32, Cpu> result = SliceOps.gather(embeddings, indices)) {
                    assertArrayEquals(new int[]{2, 3, 3}, result.dimensions());

                    float[] data = result.underlying().toFloatArray();

                    // First batch: tokens 0, 2, 4
                    // Token 0: [0, 1, 2]
                    assertEquals(0, data[0], EPSILON);
                    assertEquals(1, data[1], EPSILON);
                    assertEquals(2, data[2], EPSILON);

                    // Token 2: [6, 7, 8]
                    assertEquals(6, data[3], EPSILON);
                    assertEquals(7, data[4], EPSILON);
                    assertEquals(8, data[5], EPSILON);

                    // Second batch: tokens 1, 3, 0
                    // Token 1: [3, 4, 5]
                    assertEquals(3, data[9], EPSILON);
                    assertEquals(4, data[10], EPSILON);
                    assertEquals(5, data[11], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("gather rejects out of bounds indices")
        void gatherRejectsOutOfBoundsIndices() {
            try (TypedTensor<Matrix, F32, Cpu> embeddings = TypedTensor.zeros(
                    new Matrix(5, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                int[][] indices = {{0, 5}};  // 5 is out of bounds

                assertThrows(IndexOutOfBoundsException.class,
                        () -> SliceOps.gather(embeddings, indices));
            }
        }

        @Test
        @DisplayName("gatherRows selects rows from matrix")
        void gatherRowsSelectsRowsFromMatrix() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9},
                    new Matrix(3, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                int[] indices = {2, 0, 2};

                try (TypedTensor<Matrix, F32, Cpu> result = SliceOps.gatherRows(mat, indices)) {
                    assertArrayEquals(new int[]{3, 3}, result.dimensions());

                    float[] data = result.underlying().toFloatArray();

                    // Row 2: [7, 8, 9]
                    assertEquals(7, data[0], EPSILON);
                    assertEquals(8, data[1], EPSILON);
                    assertEquals(9, data[2], EPSILON);

                    // Row 0: [1, 2, 3]
                    assertEquals(1, data[3], EPSILON);
                    assertEquals(2, data[4], EPSILON);
                    assertEquals(3, data[5], EPSILON);

                    // Row 2 again: [7, 8, 9]
                    assertEquals(7, data[6], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("gatherVector selects elements from vector")
        void gatherVectorSelectsElements() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{10, 20, 30, 40, 50}, new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                int[] indices = {4, 2, 0, 3};

                try (TypedTensor<Vector, F32, Cpu> result = SliceOps.gatherVector(vec, indices)) {
                    assertEquals(4, result.shapeType().length());
                    assertArrayEquals(new float[]{50, 30, 10, 40}, result.underlying().toFloatArray(), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("gather works with F64")
        void gatherWorksWithF64() {
            try (TypedTensor<Matrix, F64, Cpu> embeddings = TypedTensor.fromDoubleArray(
                    new double[]{1, 2, 3, 4, 5, 6}, new Matrix(2, 3), F64.INSTANCE, Cpu.INSTANCE)) {

                int[][] indices = {{0}, {1}};

                try (TypedTensor<Rank3, F64, Cpu> result = SliceOps.gather(embeddings, indices)) {
                    assertArrayEquals(new int[]{2, 1, 3}, result.dimensions());
                }
            }
        }
    }

    @Nested
    @DisplayName("Slice Operations")
    class SliceTests {

        @Test
        @DisplayName("sliceRows extracts row range")
        void sliceRowsExtractsRowRange() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                    new Matrix(4, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                // Extract rows 1 and 2 (indices 1 to 3 exclusive)
                try (TypedTensor<Matrix, F32, Cpu> sliced = SliceOps.sliceRows(mat, 1, 3)) {
                    assertArrayEquals(new int[]{2, 3}, sliced.dimensions());

                    // Row 1: [4, 5, 6], Row 2: [7, 8, 9]
                    assertArrayEquals(new float[]{4, 5, 6, 7, 8, 9},
                            sliced.underlying().toFloatArray(), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("sliceRows handles single row")
        void sliceRowsHandlesSingleRow() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6},
                    new Matrix(3, 2), F32.INSTANCE, Cpu.INSTANCE)) {

                try (TypedTensor<Matrix, F32, Cpu> sliced = SliceOps.sliceRows(mat, 1, 2)) {
                    assertArrayEquals(new int[]{1, 2}, sliced.dimensions());
                    assertArrayEquals(new float[]{3, 4}, sliced.underlying().toFloatArray(), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("sliceRows rejects invalid range")
        void sliceRowsRejectsInvalidRange() {
            try (TypedTensor<Matrix, F32, Cpu> mat = TypedTensor.zeros(
                    new Matrix(4, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class, () -> SliceOps.sliceRows(mat, 3, 2));
                assertThrows(IllegalArgumentException.class, () -> SliceOps.sliceRows(mat, -1, 2));
                assertThrows(IllegalArgumentException.class, () -> SliceOps.sliceRows(mat, 0, 5));
            }
        }

        @Test
        @DisplayName("sliceVector extracts range")
        void sliceVectorExtractsRange() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5}, new Vector(5), F32.INSTANCE, Cpu.INSTANCE)) {

                try (TypedTensor<Vector, F32, Cpu> sliced = SliceOps.sliceVector(vec, 1, 4)) {
                    assertEquals(3, sliced.shapeType().length());
                    assertArrayEquals(new float[]{2, 3, 4}, sliced.underlying().toFloatArray(), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("sliceSequence extracts sequence range")
        void sliceSequenceExtractsSequenceRange() {
            // Input: [2, 5, 3] (batch=2, seq=5, hidden=3)
            float[] data = new float[30];
            for (int i = 0; i < 30; i++) {
                data[i] = i;
            }

            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.fromFloatArray(
                    data, new Rank3(2, 5, 3), F32.INSTANCE, Cpu.INSTANCE)) {

                // Extract sequences 1-3 (exclusive)
                try (TypedTensor<Rank3, F32, Cpu> sliced = SliceOps.sliceSequence(input, 1, 3)) {
                    assertArrayEquals(new int[]{2, 2, 3}, sliced.dimensions());

                    float[] result = sliced.underlying().toFloatArray();

                    // Batch 0, seq 1: [3, 4, 5]
                    assertEquals(3, result[0], EPSILON);
                    assertEquals(4, result[1], EPSILON);
                    assertEquals(5, result[2], EPSILON);

                    // Batch 0, seq 2: [6, 7, 8]
                    assertEquals(6, result[3], EPSILON);
                }
            }
        }
    }

    @Nested
    @DisplayName("Concatenate Operations")
    class CatTests {

        @Test
        @DisplayName("catVectors concatenates vectors")
        void catVectorsConcatenatesVectors() {
            try (TypedTensor<Vector, F32, Cpu> v1 = TypedTensor.fromFloatArray(
                    new float[]{1, 2}, new Vector(2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> v2 = TypedTensor.fromFloatArray(
                    new float[]{3, 4, 5}, new Vector(3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Vector, F32, Cpu> v3 = TypedTensor.fromFloatArray(
                    new float[]{6}, new Vector(1), F32.INSTANCE, Cpu.INSTANCE)) {

                try (TypedTensor<Vector, F32, Cpu> result = SliceOps.catVectors(List.of(v1, v2, v3))) {
                    assertEquals(6, result.shapeType().length());
                    assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6},
                            result.underlying().toFloatArray(), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("catRows stacks matrices vertically")
        void catRowsStacksMatricesVertically() {
            try (TypedTensor<Matrix, F32, Cpu> m1 = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4}, new Matrix(2, 2), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> m2 = TypedTensor.fromFloatArray(
                    new float[]{5, 6, 7, 8, 9, 10}, new Matrix(3, 2), F32.INSTANCE, Cpu.INSTANCE)) {

                try (TypedTensor<Matrix, F32, Cpu> result = SliceOps.catRows(List.of(m1, m2))) {
                    assertArrayEquals(new int[]{5, 2}, result.dimensions());
                    assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                            result.underlying().toFloatArray(), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("catRows rejects column mismatch")
        void catRowsRejectsColumnMismatch() {
            try (TypedTensor<Matrix, F32, Cpu> m1 = TypedTensor.zeros(
                    new Matrix(2, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Matrix, F32, Cpu> m2 = TypedTensor.zeros(
                    new Matrix(2, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> SliceOps.catRows(List.of(m1, m2)));
            }
        }

        @Test
        @DisplayName("catHidden concatenates along hidden dimension")
        void catHiddenConcatenatesAlongHiddenDimension() {
            // Simulates combining attention heads
            try (TypedTensor<Rank3, F32, Cpu> h1 = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6},
                    new Rank3(2, 1, 3), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> h2 = TypedTensor.fromFloatArray(
                    new float[]{7, 8, 9, 10},
                    new Rank3(2, 1, 2), F32.INSTANCE, Cpu.INSTANCE)) {

                try (TypedTensor<Rank3, F32, Cpu> result = SliceOps.catHidden(List.of(h1, h2))) {
                    assertArrayEquals(new int[]{2, 1, 5}, result.dimensions());

                    // Batch 0: [1, 2, 3, 7, 8]
                    // Batch 1: [4, 5, 6, 9, 10]
                    float[] data = result.underlying().toFloatArray();
                    assertEquals(1, data[0], EPSILON);
                    assertEquals(2, data[1], EPSILON);
                    assertEquals(3, data[2], EPSILON);
                    assertEquals(7, data[3], EPSILON);
                    assertEquals(8, data[4], EPSILON);
                    assertEquals(4, data[5], EPSILON);
                    assertEquals(5, data[6], EPSILON);
                }
            }
        }

        @Test
        @DisplayName("catHidden rejects batch/seq mismatch")
        void catHiddenRejectsBatchSeqMismatch() {
            try (TypedTensor<Rank3, F32, Cpu> t1 = TypedTensor.zeros(
                    new Rank3(2, 3, 4), F32.INSTANCE, Cpu.INSTANCE);
                 TypedTensor<Rank3, F32, Cpu> t2 = TypedTensor.zeros(
                    new Rank3(2, 4, 4), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> SliceOps.catHidden(List.of(t1, t2)));
            }
        }

        @Test
        @DisplayName("catVectors rejects empty list")
        void catVectorsRejectsEmptyList() {
            assertThrows(IllegalArgumentException.class,
                    () -> SliceOps.catVectors(List.of()));
        }
    }

    @Nested
    @DisplayName("Split Operations")
    class SplitTests {

        @Test
        @DisplayName("splitVector divides vector evenly")
        void splitVectorDividesVectorEvenly() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Vector(6), F32.INSTANCE, Cpu.INSTANCE)) {

                List<TypedTensor<Vector, F32, Cpu>> chunks = SliceOps.splitVector(vec, 3);

                assertEquals(3, chunks.size());

                assertEquals(2, chunks.get(0).shapeType().length());
                assertArrayEquals(new float[]{1, 2}, chunks.get(0).underlying().toFloatArray(), EPSILON);

                assertArrayEquals(new float[]{3, 4}, chunks.get(1).underlying().toFloatArray(), EPSILON);

                assertArrayEquals(new float[]{5, 6}, chunks.get(2).underlying().toFloatArray(), EPSILON);

                // Clean up
                for (TypedTensor<Vector, F32, Cpu> chunk : chunks) {
                    chunk.close();
                }
            }
        }

        @Test
        @DisplayName("splitVector rejects non-divisible size")
        void splitVectorRejectsNonDivisibleSize() {
            try (TypedTensor<Vector, F32, Cpu> vec = TypedTensor.zeros(
                    new Vector(7), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> SliceOps.splitVector(vec, 3));
            }
        }

        @Test
        @DisplayName("splitHidden divides into attention heads")
        void splitHiddenDividesIntoAttentionHeads() {
            // Simulates splitting hidden dim into attention heads
            // [batch=2, seq=1, hidden=6] -> 3 heads of [batch=2, seq=1, head_dim=2]
            float[] data = new float[12];
            for (int i = 0; i < 12; i++) {
                data[i] = i;
            }

            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.fromFloatArray(
                    data, new Rank3(2, 1, 6), F32.INSTANCE, Cpu.INSTANCE)) {

                List<TypedTensor<Rank3, F32, Cpu>> heads = SliceOps.splitHidden(input, 3);

                assertEquals(3, heads.size());

                for (TypedTensor<Rank3, F32, Cpu> head : heads) {
                    assertArrayEquals(new int[]{2, 1, 2}, head.dimensions());
                }

                // Head 0 should have [0, 1] for batch 0 and [6, 7] for batch 1
                float[] head0 = heads.get(0).underlying().toFloatArray();
                assertEquals(0, head0[0], EPSILON);
                assertEquals(1, head0[1], EPSILON);
                assertEquals(6, head0[2], EPSILON);
                assertEquals(7, head0[3], EPSILON);

                // Head 1 should have [2, 3] for batch 0 and [8, 9] for batch 1
                float[] head1 = heads.get(1).underlying().toFloatArray();
                assertEquals(2, head1[0], EPSILON);
                assertEquals(3, head1[1], EPSILON);

                // Clean up
                for (TypedTensor<Rank3, F32, Cpu> head : heads) {
                    head.close();
                }
            }
        }

        @Test
        @DisplayName("splitHidden rejects non-divisible hidden dim")
        void splitHiddenRejectsNonDivisibleHiddenDim() {
            try (TypedTensor<Rank3, F32, Cpu> input = TypedTensor.zeros(
                    new Rank3(2, 4, 7), F32.INSTANCE, Cpu.INSTANCE)) {

                assertThrows(IllegalArgumentException.class,
                        () -> SliceOps.splitHidden(input, 3));
            }
        }
    }

    @Nested
    @DisplayName("Round-trip Operations")
    class RoundTripTests {

        @Test
        @DisplayName("split then cat recovers original")
        void splitThenCatRecoversOriginal() {
            try (TypedTensor<Vector, F32, Cpu> original = TypedTensor.fromFloatArray(
                    new float[]{1, 2, 3, 4, 5, 6}, new Vector(6), F32.INSTANCE, Cpu.INSTANCE)) {

                List<TypedTensor<Vector, F32, Cpu>> chunks = SliceOps.splitVector(original, 2);

                try (TypedTensor<Vector, F32, Cpu> recovered = SliceOps.catVectors(chunks)) {
                    assertArrayEquals(original.underlying().toFloatArray(),
                            recovered.underlying().toFloatArray(), EPSILON);
                }

                for (TypedTensor<Vector, F32, Cpu> chunk : chunks) {
                    chunk.close();
                }
            }
        }

        @Test
        @DisplayName("splitHidden then catHidden recovers original")
        void splitHiddenThenCatHiddenRecoversOriginal() {
            float[] data = new float[24];
            for (int i = 0; i < 24; i++) {
                data[i] = i;
            }

            try (TypedTensor<Rank3, F32, Cpu> original = TypedTensor.fromFloatArray(
                    data, new Rank3(2, 2, 6), F32.INSTANCE, Cpu.INSTANCE)) {

                List<TypedTensor<Rank3, F32, Cpu>> heads = SliceOps.splitHidden(original, 3);

                try (TypedTensor<Rank3, F32, Cpu> recovered = SliceOps.catHidden(heads)) {
                    assertArrayEquals(original.dimensions(), recovered.dimensions());
                    assertArrayEquals(original.underlying().toFloatArray(),
                            recovered.underlying().toFloatArray(), EPSILON);
                }

                for (TypedTensor<Rank3, F32, Cpu> head : heads) {
                    head.close();
                }
            }
        }
    }
}
