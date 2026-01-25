package io.surfworks.warpforge.data.tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Interface for tokenizers that convert text to token IDs and back.
 *
 * <p>Supports loading HuggingFace tokenizers and provides methods for
 * encoding text, decoding tokens, and handling special tokens.
 *
 * <p>Example usage:
 * <pre>{@code
 * Tokenizer tokenizer = Tokenizer.fromHuggingFace("bert-base-uncased");
 * EncodedInput encoded = tokenizer.encode("Hello, world!");
 * String decoded = tokenizer.decode(encoded.inputIds());
 * }</pre>
 */
public interface Tokenizer extends AutoCloseable {

    /**
     * Encode a single text string.
     *
     * @param text The text to encode
     * @return Encoded input with token IDs and attention mask
     */
    EncodedInput encode(String text);

    /**
     * Encode a single text with options.
     *
     * @param text The text to encode
     * @param options Encoding options
     * @return Encoded input
     */
    EncodedInput encode(String text, EncodeOptions options);

    /**
     * Encode a pair of texts (e.g., for question-answering or sentence pairs).
     *
     * @param text The first text
     * @param textPair The second text
     * @return Encoded input
     */
    EncodedInput encodePair(String text, String textPair);

    /**
     * Encode multiple texts in a batch.
     *
     * @param texts List of texts to encode
     * @return List of encoded inputs
     */
    List<EncodedInput> encodeBatch(List<String> texts);

    /**
     * Encode multiple texts with padding to same length.
     *
     * @param texts List of texts to encode
     * @param options Encoding options with padding settings
     * @return Batch encoded input with 2D arrays
     */
    BatchEncodedInput encodeBatchPadded(List<String> texts, EncodeOptions options);

    /**
     * Decode token IDs back to text.
     *
     * @param tokenIds Array of token IDs
     * @return Decoded text
     */
    String decode(int[] tokenIds);

    /**
     * Decode token IDs, optionally skipping special tokens.
     *
     * @param tokenIds Array of token IDs
     * @param skipSpecialTokens Whether to skip special tokens like [CLS], [SEP]
     * @return Decoded text
     */
    String decode(int[] tokenIds, boolean skipSpecialTokens);

    /**
     * Convert a single token to its ID.
     *
     * @param token The token string
     * @return Token ID, or unknown token ID if not found
     */
    int tokenToId(String token);

    /**
     * Convert a token ID to its string representation.
     *
     * @param id Token ID
     * @return Token string
     */
    String idToToken(int id);

    /**
     * Get the vocabulary size.
     */
    int vocabSize();

    /**
     * Get special token IDs.
     */
    SpecialTokens specialTokens();

    /**
     * Get the model's maximum sequence length.
     */
    int maxLength();

    @Override
    void close();

    // ========== Factory Methods ==========

    /**
     * Load a tokenizer from a HuggingFace model ID.
     *
     * @param modelId HuggingFace model ID (e.g., "bert-base-uncased")
     * @return Loaded tokenizer
     */
    static Tokenizer fromHuggingFace(String modelId) throws IOException {
        return HuggingFaceTokenizer.load(modelId);
    }

    /**
     * Load a tokenizer from a local directory.
     *
     * @param dir Directory containing tokenizer files
     * @return Loaded tokenizer
     */
    static Tokenizer fromDirectory(Path dir) throws IOException {
        return HuggingFaceTokenizer.loadFromDirectory(dir);
    }

    /**
     * Create a simple whitespace tokenizer for testing.
     *
     * @param vocab Map from tokens to IDs
     * @return Simple tokenizer
     */
    static Tokenizer whitespace(Map<String, Integer> vocab) {
        return new WhitespaceTokenizer(vocab);
    }

    // ========== Data Classes ==========

    /**
     * Encoded input from tokenizer.
     */
    record EncodedInput(
            int[] inputIds,
            int[] attentionMask,
            int[] tokenTypeIds,
            List<String> tokens,
            List<int[]> offsets
    ) {
        /**
         * Create encoded input with just IDs and mask.
         */
        public static EncodedInput of(int[] inputIds, int[] attentionMask) {
            return new EncodedInput(inputIds, attentionMask, null, null, null);
        }

        /**
         * Get the sequence length.
         */
        public int length() {
            return inputIds.length;
        }
    }

    /**
     * Batch encoded input with padding.
     */
    record BatchEncodedInput(
            int[][] inputIds,
            int[][] attentionMask,
            int[][] tokenTypeIds,
            int batchSize,
            int maxLength
    ) {
        /**
         * Create batch input with just IDs and mask.
         */
        public static BatchEncodedInput of(int[][] inputIds, int[][] attentionMask) {
            return new BatchEncodedInput(inputIds, attentionMask, null,
                    inputIds.length, inputIds.length > 0 ? inputIds[0].length : 0);
        }
    }

    /**
     * Special tokens used by the tokenizer.
     */
    record SpecialTokens(
            int padTokenId,
            int clsTokenId,
            int sepTokenId,
            int unkTokenId,
            int maskTokenId,
            int bosTokenId,
            int eosTokenId
    ) {
        /**
         * Create special tokens with common defaults.
         */
        public static SpecialTokens defaults() {
            return new SpecialTokens(0, 101, 102, 100, 103, -1, -1);
        }

        /**
         * Create special tokens for GPT-style models.
         */
        public static SpecialTokens gpt() {
            return new SpecialTokens(-1, -1, -1, -1, -1, 50256, 50256);
        }
    }

    /**
     * Options for encoding.
     */
    record EncodeOptions(
            int maxLength,
            boolean truncation,
            boolean padding,
            PaddingStrategy paddingStrategy,
            boolean addSpecialTokens,
            boolean returnTokens,
            boolean returnOffsets
    ) {
        /**
         * Default encoding options.
         */
        public static EncodeOptions defaults() {
            return new EncodeOptions(512, true, false, PaddingStrategy.DO_NOT_PAD, true, false, false);
        }

        /**
         * Create options with padding to max length.
         */
        public static EncodeOptions withPadding(int maxLength) {
            return new EncodeOptions(maxLength, true, true, PaddingStrategy.MAX_LENGTH, true, false, false);
        }

        /**
         * Create options for batch encoding with longest padding.
         */
        public static EncodeOptions forBatch() {
            return new EncodeOptions(512, true, true, PaddingStrategy.LONGEST, true, false, false);
        }

        public EncodeOptions withMaxLength(int maxLength) {
            return new EncodeOptions(maxLength, truncation, padding, paddingStrategy, addSpecialTokens, returnTokens, returnOffsets);
        }

        public EncodeOptions withTruncation(boolean truncation) {
            return new EncodeOptions(maxLength, truncation, padding, paddingStrategy, addSpecialTokens, returnTokens, returnOffsets);
        }

        public EncodeOptions withPadding(boolean padding) {
            return new EncodeOptions(maxLength, truncation, padding, paddingStrategy, addSpecialTokens, returnTokens, returnOffsets);
        }
    }

    /**
     * Padding strategy for batch encoding.
     */
    enum PaddingStrategy {
        /** Do not pad */
        DO_NOT_PAD,
        /** Pad to the longest sequence in the batch */
        LONGEST,
        /** Pad to the max_length parameter */
        MAX_LENGTH
    }
}
