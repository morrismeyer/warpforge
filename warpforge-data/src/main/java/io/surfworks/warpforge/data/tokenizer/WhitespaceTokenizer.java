package io.surfworks.warpforge.data.tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Simple whitespace-based tokenizer for testing.
 *
 * <p>Splits text on whitespace and looks up each token in a vocabulary.
 * Unknown tokens are mapped to the unknown token ID.
 */
final class WhitespaceTokenizer implements Tokenizer {

    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private final SpecialTokens specialTokens;
    private final int maxLength;

    WhitespaceTokenizer(Map<String, Integer> vocab) {
        this.vocab = new HashMap<>(vocab);
        this.reverseVocab = new HashMap<>();
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            reverseVocab.put(entry.getValue(), entry.getKey());
        }

        // Ensure special tokens exist
        ensureToken("[PAD]", 0);
        ensureToken("[UNK]", 1);
        ensureToken("[CLS]", 2);
        ensureToken("[SEP]", 3);
        ensureToken("[MASK]", 4);

        this.specialTokens = new SpecialTokens(
                vocab.getOrDefault("[PAD]", 0),
                vocab.getOrDefault("[CLS]", 2),
                vocab.getOrDefault("[SEP]", 3),
                vocab.getOrDefault("[UNK]", 1),
                vocab.getOrDefault("[MASK]", 4),
                -1, -1
        );
        this.maxLength = 512;
    }

    private void ensureToken(String token, int defaultId) {
        if (!vocab.containsKey(token)) {
            int id = vocab.containsValue(defaultId) ? vocab.size() : defaultId;
            vocab.put(token, id);
            reverseVocab.put(id, token);
        }
    }

    @Override
    public EncodedInput encode(String text) {
        return encode(text, EncodeOptions.defaults());
    }

    @Override
    public EncodedInput encode(String text, EncodeOptions options) {
        List<Integer> ids = new ArrayList<>();

        // Add [CLS] if requested
        if (options.addSpecialTokens()) {
            ids.add(specialTokens.clsTokenId());
        }

        // Tokenize
        String[] tokens = text.toLowerCase().split("\\s+");
        for (String token : tokens) {
            if (!token.isEmpty()) {
                ids.add(vocab.getOrDefault(token, specialTokens.unkTokenId()));
            }
        }

        // Add [SEP] if requested
        if (options.addSpecialTokens()) {
            ids.add(specialTokens.sepTokenId());
        }

        // Truncate if needed
        if (options.truncation() && ids.size() > options.maxLength()) {
            ids = ids.subList(0, options.maxLength());
        }

        // Pad if needed
        int targetLength = ids.size();
        if (options.padding() && options.paddingStrategy() == PaddingStrategy.MAX_LENGTH) {
            targetLength = options.maxLength();
        }

        int[] inputIds = new int[targetLength];
        int[] attentionMask = new int[targetLength];

        for (int i = 0; i < ids.size(); i++) {
            inputIds[i] = ids.get(i);
            attentionMask[i] = 1;
        }

        // Pad remaining positions
        for (int i = ids.size(); i < targetLength; i++) {
            inputIds[i] = specialTokens.padTokenId();
            attentionMask[i] = 0;
        }

        return EncodedInput.of(inputIds, attentionMask);
    }

    @Override
    public EncodedInput encodePair(String text, String textPair) {
        List<Integer> ids = new ArrayList<>();
        ids.add(specialTokens.clsTokenId());

        // First text
        for (String token : text.toLowerCase().split("\\s+")) {
            if (!token.isEmpty()) {
                ids.add(vocab.getOrDefault(token, specialTokens.unkTokenId()));
            }
        }

        ids.add(specialTokens.sepTokenId());

        // Second text
        for (String token : textPair.toLowerCase().split("\\s+")) {
            if (!token.isEmpty()) {
                ids.add(vocab.getOrDefault(token, specialTokens.unkTokenId()));
            }
        }

        ids.add(specialTokens.sepTokenId());

        int[] inputIds = ids.stream().mapToInt(i -> i).toArray();
        int[] attentionMask = new int[inputIds.length];
        Arrays.fill(attentionMask, 1);

        return EncodedInput.of(inputIds, attentionMask);
    }

    @Override
    public List<EncodedInput> encodeBatch(List<String> texts) {
        List<EncodedInput> results = new ArrayList<>();
        for (String text : texts) {
            results.add(encode(text));
        }
        return results;
    }

    @Override
    public BatchEncodedInput encodeBatchPadded(List<String> texts, EncodeOptions options) {
        List<EncodedInput> encoded = new ArrayList<>();
        int maxLen = 0;

        for (String text : texts) {
            EncodedInput enc = encode(text, options.withPadding(false));
            encoded.add(enc);
            maxLen = Math.max(maxLen, enc.length());
        }

        // Determine target length
        int targetLength = switch (options.paddingStrategy()) {
            case MAX_LENGTH -> options.maxLength();
            case LONGEST -> maxLen;
            case DO_NOT_PAD -> maxLen;
        };

        int[][] inputIds = new int[texts.size()][targetLength];
        int[][] attentionMask = new int[texts.size()][targetLength];

        for (int i = 0; i < encoded.size(); i++) {
            EncodedInput enc = encoded.get(i);
            int len = Math.min(enc.length(), targetLength);

            System.arraycopy(enc.inputIds(), 0, inputIds[i], 0, len);
            System.arraycopy(enc.attentionMask(), 0, attentionMask[i], 0, len);

            // Pad remaining
            for (int j = len; j < targetLength; j++) {
                inputIds[i][j] = specialTokens.padTokenId();
                attentionMask[i][j] = 0;
            }
        }

        return BatchEncodedInput.of(inputIds, attentionMask);
    }

    @Override
    public String decode(int[] tokenIds) {
        return decode(tokenIds, true);
    }

    @Override
    public String decode(int[] tokenIds, boolean skipSpecialTokens) {
        StringBuilder sb = new StringBuilder();
        for (int id : tokenIds) {
            if (skipSpecialTokens && isSpecialToken(id)) {
                continue;
            }
            String token = reverseVocab.getOrDefault(id, "[UNK]");
            if (sb.length() > 0) {
                sb.append(" ");
            }
            sb.append(token);
        }
        return sb.toString();
    }

    private boolean isSpecialToken(int id) {
        return id == specialTokens.padTokenId() ||
                id == specialTokens.clsTokenId() ||
                id == specialTokens.sepTokenId() ||
                id == specialTokens.unkTokenId() ||
                id == specialTokens.maskTokenId();
    }

    @Override
    public int tokenToId(String token) {
        return vocab.getOrDefault(token.toLowerCase(), specialTokens.unkTokenId());
    }

    @Override
    public String idToToken(int id) {
        return reverseVocab.getOrDefault(id, "[UNK]");
    }

    @Override
    public int vocabSize() {
        return vocab.size();
    }

    @Override
    public SpecialTokens specialTokens() {
        return specialTokens;
    }

    @Override
    public int maxLength() {
        return maxLength;
    }

    @Override
    public void close() {
        // No resources to close
    }
}
