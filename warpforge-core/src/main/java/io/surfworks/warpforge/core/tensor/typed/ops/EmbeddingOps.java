package io.surfworks.warpforge.core.tensor.typed.ops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Type-safe embedding operations for tensors.
 *
 * <p>These operations implement embedding layers used in transformer models:
 * <ul>
 *   <li>Token embeddings - map token IDs to dense vectors</li>
 *   <li>Position embeddings - add positional information to sequences</li>
 *   <li>Combined embeddings - token + position for transformer input</li>
 * </ul>
 *
 * <p>Embeddings are essentially learned lookup tables. Given an embedding matrix
 * [vocab_size, hidden_dim] and token indices [batch, seq_len], we look up the
 * corresponding rows to produce [batch, seq_len, hidden_dim].
 *
 * <p>Example:
 * <pre>{@code
 * // Token embedding lookup
 * TypedTensor<Matrix, F32, Cpu> tokenEmbed = ...;  // [vocab_size, hidden_dim]
 * int[][] tokenIds = {{101, 2054, 2003, 2023}, {101, 7592, 102, 0}};  // [batch=2, seq=4]
 *
 * TypedTensor<Rank3, F32, Cpu> tokens = EmbeddingOps.embedding(tokenEmbed, tokenIds);
 * // Result: [2, 4, hidden_dim]
 *
 * // Add position embeddings
 * TypedTensor<Matrix, F32, Cpu> posEmbed = ...;  // [max_seq_len, hidden_dim]
 * TypedTensor<Rank3, F32, Cpu> output = EmbeddingOps.addPositionEmbedding(tokens, posEmbed);
 * }</pre>
 */
public final class EmbeddingOps {

    private EmbeddingOps() {
        // Utility class
    }

    // ==================== Token Embedding ====================

    /**
     * Looks up embeddings from an embedding table.
     *
     * <p>Given embedding table [vocab_size, hidden_dim] and indices [batch, seq],
     * produces output [batch, seq, hidden_dim].
     *
     * <p>This is equivalent to SliceOps.gather but with a more intuitive name
     * for the embedding use case.
     *
     * @param embeddingTable the embedding matrix [vocab_size, hidden_dim]
     * @param indices token IDs [batch, seq]
     * @param <V> device type
     * @return embedded tokens [batch, seq, hidden_dim]
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank3, F32, V> embedding(TypedTensor<Matrix, F32, V> embeddingTable, int[][] indices) {
        // Delegate to SliceOps.gather which does the actual work
        return SliceOps.gather(embeddingTable, indices);
    }

    /**
     * Looks up embeddings for a single sequence (no batch dimension).
     *
     * <p>Given embedding table [vocab_size, hidden_dim] and indices [seq],
     * produces output [seq, hidden_dim].
     *
     * @param embeddingTable the embedding matrix [vocab_size, hidden_dim]
     * @param indices token IDs [seq]
     * @param <V> device type
     * @return embedded tokens [seq, hidden_dim]
     */
    public static <V extends DeviceTag>
    TypedTensor<Matrix, F32, V> embeddingSingle(TypedTensor<Matrix, F32, V> embeddingTable, int[] indices) {
        return SliceOps.gatherRows(embeddingTable, indices);
    }

    // ==================== Position Embedding ====================

    /**
     * Creates position embeddings for a sequence.
     *
     * <p>Extracts rows [0, seqLen) from the position embedding table.
     *
     * @param positionTable the position embedding matrix [max_seq_len, hidden_dim]
     * @param seqLen the sequence length to extract
     * @param <V> device type
     * @return position embeddings [seqLen, hidden_dim]
     */
    public static <V extends DeviceTag>
    TypedTensor<Matrix, F32, V> positionEmbedding(TypedTensor<Matrix, F32, V> positionTable, int seqLen) {
        return SliceOps.sliceRows(positionTable, 0, seqLen);
    }

    /**
     * Adds position embeddings to token embeddings.
     *
     * <p>For input [batch, seq, hidden] and position [seq, hidden],
     * broadcasts position across batch and adds elementwise.
     *
     * @param tokens token embeddings [batch, seq, hidden]
     * @param positions position embeddings [seq, hidden]
     * @param <V> device type
     * @return combined embeddings [batch, seq, hidden]
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank3, F32, V> addPositionEmbedding(
            TypedTensor<Rank3, F32, V> tokens,
            TypedTensor<Matrix, F32, V> positions) {

        int[] tokenDims = tokens.dimensions();
        int[] posDims = positions.dimensions();

        int batch = tokenDims[0];
        int seq = tokenDims[1];
        int hidden = tokenDims[2];

        // Validate dimensions
        if (posDims[0] != seq || posDims[1] != hidden) {
            throw new IllegalArgumentException(
                    "Position embedding shape [" + posDims[0] + ", " + posDims[1] +
                    "] doesn't match token shape [*, " + seq + ", " + hidden + "]");
        }

        Tensor result = Tensor.zeros(ScalarType.F32, tokenDims);
        addPositionF32(
                tokens.underlying().data(),
                positions.underlying().data(),
                result.data(),
                batch, seq, hidden);

        return TypedTensor.from(result, new Rank3(batch, seq, hidden), F32.INSTANCE, tokens.deviceType());
    }

    // ==================== Combined Embedding ====================

    /**
     * Creates complete transformer input embeddings.
     *
     * <p>Combines token embeddings and position embeddings in one operation:
     * output = token_embed[token_ids] + position_embed[0:seq_len]
     *
     * @param tokenTable token embedding table [vocab_size, hidden_dim]
     * @param positionTable position embedding table [max_seq_len, hidden_dim]
     * @param tokenIds token IDs [batch, seq]
     * @param <V> device type
     * @return combined embeddings [batch, seq, hidden_dim]
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank3, F32, V> transformerEmbedding(
            TypedTensor<Matrix, F32, V> tokenTable,
            TypedTensor<Matrix, F32, V> positionTable,
            int[][] tokenIds) {

        // Get token embeddings
        TypedTensor<Rank3, F32, V> tokens = embedding(tokenTable, tokenIds);

        // Get position embeddings for sequence length
        int seqLen = tokenIds[0].length;
        TypedTensor<Matrix, F32, V> positions = positionEmbedding(positionTable, seqLen);

        // Add them together
        return addPositionEmbedding(tokens, positions);
    }

    /**
     * Creates learned position embeddings.
     *
     * <p>Generates position IDs [0, 1, 2, ..., seqLen-1] and looks them up
     * in the position embedding table.
     *
     * @param positionTable the position embedding matrix [max_seq_len, hidden_dim]
     * @param batchSize the batch size
     * @param seqLen the sequence length
     * @param <V> device type
     * @return position embeddings [batch, seq, hidden]
     */
    public static <V extends DeviceTag>
    TypedTensor<Rank3, F32, V> learnedPositionEmbedding(
            TypedTensor<Matrix, F32, V> positionTable,
            int batchSize,
            int seqLen) {

        int[] posDims = positionTable.dimensions();
        int hidden = posDims[1];

        if (seqLen > posDims[0]) {
            throw new IllegalArgumentException(
                    "Sequence length " + seqLen + " exceeds max position " + posDims[0]);
        }

        // Get position embeddings [seq, hidden]
        TypedTensor<Matrix, F32, V> positions = positionEmbedding(positionTable, seqLen);

        // Expand to [batch, seq, hidden]
        return ShapeOps.expandToRank3(positions, new Rank3(batchSize, seqLen, hidden));
    }

    // ==================== Sinusoidal Position Embedding ====================

    /**
     * Creates sinusoidal position embeddings (non-learned).
     *
     * <p>PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
     * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
     *
     * <p>This is the original positional encoding from "Attention Is All You Need".
     *
     * @param seqLen maximum sequence length
     * @param hiddenDim embedding dimension (must be even)
     * @param <V> device type
     * @return sinusoidal position embeddings [seqLen, hiddenDim]
     */
    public static <V extends DeviceTag>
    TypedTensor<Matrix, F32, V> sinusoidalPositionEmbedding(int seqLen, int hiddenDim, V deviceType) {
        if (hiddenDim % 2 != 0) {
            throw new IllegalArgumentException(
                    "Hidden dimension must be even for sinusoidal embeddings, got: " + hiddenDim);
        }

        Tensor result = Tensor.zeros(ScalarType.F32, seqLen, hiddenDim);
        sinusoidalPositionF32(result.data(), seqLen, hiddenDim);

        return TypedTensor.from(result, new Matrix(seqLen, hiddenDim), F32.INSTANCE, deviceType);
    }

    // ==================== Internal Implementation ====================

    private static void addPositionF32(MemorySegment tokens, MemorySegment positions,
                                       MemorySegment dst, int batch, int seq, int hidden) {
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                long tokenOffset = ((long) b * seq + s) * hidden;
                long posOffset = (long) s * hidden;

                for (int h = 0; h < hidden; h++) {
                    float tokenVal = tokens.getAtIndex(ValueLayout.JAVA_FLOAT, tokenOffset + h);
                    float posVal = positions.getAtIndex(ValueLayout.JAVA_FLOAT, posOffset + h);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, tokenOffset + h, tokenVal + posVal);
                }
            }
        }
    }

    private static void sinusoidalPositionF32(MemorySegment dst, int seqLen, int hiddenDim) {
        int halfDim = hiddenDim / 2;
        double logBase = Math.log(10000.0);

        for (int pos = 0; pos < seqLen; pos++) {
            long offset = (long) pos * hiddenDim;

            for (int i = 0; i < halfDim; i++) {
                double divTerm = Math.exp(-logBase * i / halfDim);
                double angle = pos * divTerm;

                // Even indices: sin
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + 2 * i, (float) Math.sin(angle));
                // Odd indices: cos
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + 2 * i + 1, (float) Math.cos(angle));
            }
        }
    }
}
