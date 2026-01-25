package io.surfworks.warpforge.data.tokenizer;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import io.surfworks.warpforge.data.WarpForge;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * HuggingFace tokenizer implementation.
 *
 * <p>Supports loading tokenizers from HuggingFace Hub or local directories.
 * Handles common tokenizer types including:
 * <ul>
 *   <li>WordPiece (BERT)</li>
 *   <li>BPE (GPT-2, RoBERTa)</li>
 *   <li>SentencePiece (T5, ALBERT)</li>
 * </ul>
 */
final class HuggingFaceTokenizer implements Tokenizer {

    private static final Gson GSON = new Gson();

    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private final SpecialTokens specialTokens;
    private final int maxLength;
    private final TokenizerType type;
    private final Map<String, String> merges;
    private final Pattern tokenPattern;

    private HuggingFaceTokenizer(Builder builder) {
        this.vocab = builder.vocab;
        this.reverseVocab = new HashMap<>();
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            reverseVocab.put(entry.getValue(), entry.getKey());
        }
        this.specialTokens = builder.specialTokens;
        this.maxLength = builder.maxLength;
        this.type = builder.type;
        this.merges = builder.merges;
        this.tokenPattern = builder.tokenPattern;
    }

    /**
     * Load tokenizer from HuggingFace Hub.
     */
    static HuggingFaceTokenizer load(String modelId) throws IOException {
        Path cacheDir = WarpForge.cacheDir().resolve("tokenizers").resolve(modelId.replace("/", "_"));

        if (!Files.exists(cacheDir)) {
            downloadTokenizer(modelId, cacheDir);
        }

        return loadFromDirectory(cacheDir);
    }

    /**
     * Load tokenizer from a local directory.
     */
    static HuggingFaceTokenizer loadFromDirectory(Path dir) throws IOException {
        Builder builder = new Builder();

        // Try to load tokenizer.json first (new format)
        Path tokenizerJson = dir.resolve("tokenizer.json");
        if (Files.exists(tokenizerJson)) {
            loadFromTokenizerJson(tokenizerJson, builder);
        } else {
            // Fall back to vocab.txt or vocab.json
            Path vocabTxt = dir.resolve("vocab.txt");
            Path vocabJson = dir.resolve("vocab.json");

            if (Files.exists(vocabTxt)) {
                loadWordPieceVocab(vocabTxt, builder);
            } else if (Files.exists(vocabJson)) {
                loadJsonVocab(vocabJson, builder);
            } else {
                throw new IOException("No vocabulary file found in: " + dir);
            }

            // Load merges for BPE
            Path mergesFile = dir.resolve("merges.txt");
            if (Files.exists(mergesFile)) {
                loadMerges(mergesFile, builder);
            }
        }

        // Load tokenizer config for special tokens
        Path configPath = dir.resolve("tokenizer_config.json");
        if (Files.exists(configPath)) {
            loadConfig(configPath, builder);
        }

        // Load special tokens map
        Path specialTokensPath = dir.resolve("special_tokens_map.json");
        if (Files.exists(specialTokensPath)) {
            loadSpecialTokensMap(specialTokensPath, builder);
        }

        return builder.build();
    }

    private static void downloadTokenizer(String modelId, Path cacheDir) throws IOException {
        Files.createDirectories(cacheDir);

        String[] files = {"tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json",
                "merges.txt", "special_tokens_map.json"};

        String baseUrl = "https://huggingface.co/" + modelId + "/resolve/main/";

        for (String file : files) {
            try {
                Path target = cacheDir.resolve(file);
                java.net.URL url = new java.net.URL(baseUrl + file);
                java.net.HttpURLConnection conn = (java.net.HttpURLConnection) url.openConnection();
                conn.setRequestProperty("User-Agent", "warpforge/1.0");
                conn.setConnectTimeout(10000);
                conn.setReadTimeout(30000);

                if (conn.getResponseCode() == 200) {
                    try (java.io.InputStream in = conn.getInputStream()) {
                        Files.copy(in, target);
                    }
                }
                conn.disconnect();
            } catch (Exception e) {
                // File might not exist, that's OK
            }
        }
    }

    private static void loadFromTokenizerJson(Path path, Builder builder) throws IOException {
        String content = Files.readString(path);
        JsonObject root = GSON.fromJson(content, JsonObject.class);

        // Extract vocabulary from model.vocab
        if (root.has("model")) {
            JsonObject model = root.getAsJsonObject("model");

            if (model.has("vocab")) {
                JsonObject vocab = model.getAsJsonObject("vocab");
                for (Map.Entry<String, JsonElement> entry : vocab.entrySet()) {
                    builder.vocab.put(entry.getKey(), entry.getValue().getAsInt());
                }
            }

            // Detect type
            if (model.has("type")) {
                String type = model.get("type").getAsString();
                builder.type = switch (type.toLowerCase()) {
                    case "wordpiece" -> TokenizerType.WORDPIECE;
                    case "bpe" -> TokenizerType.BPE;
                    case "unigram" -> TokenizerType.SENTENCEPIECE;
                    default -> TokenizerType.WORDPIECE;
                };
            }

            // Load merges for BPE
            if (model.has("merges")) {
                for (JsonElement merge : model.getAsJsonArray("merges")) {
                    String mergeStr = merge.getAsString();
                    String[] parts = mergeStr.split(" ");
                    if (parts.length == 2) {
                        builder.merges.put(parts[0] + " " + parts[1], parts[0] + parts[1]);
                    }
                }
            }
        }
    }

    private static void loadWordPieceVocab(Path path, Builder builder) throws IOException {
        List<String> lines = Files.readAllLines(path);
        for (int i = 0; i < lines.size(); i++) {
            String token = lines.get(i).trim();
            if (!token.isEmpty()) {
                builder.vocab.put(token, i);
            }
        }
        builder.type = TokenizerType.WORDPIECE;
    }

    private static void loadJsonVocab(Path path, Builder builder) throws IOException {
        String content = Files.readString(path);
        JsonObject vocab = GSON.fromJson(content, JsonObject.class);
        for (Map.Entry<String, JsonElement> entry : vocab.entrySet()) {
            builder.vocab.put(entry.getKey(), entry.getValue().getAsInt());
        }
        builder.type = TokenizerType.BPE;
    }

    private static void loadMerges(Path path, Builder builder) throws IOException {
        List<String> lines = Files.readAllLines(path);
        for (String line : lines) {
            if (line.startsWith("#") || line.trim().isEmpty()) continue;
            String[] parts = line.split(" ");
            if (parts.length == 2) {
                builder.merges.put(parts[0] + " " + parts[1], parts[0] + parts[1]);
            }
        }
    }

    private static void loadConfig(Path path, Builder builder) throws IOException {
        String content = Files.readString(path);
        JsonObject config = GSON.fromJson(content, JsonObject.class);

        if (config.has("model_max_length")) {
            builder.maxLength = config.get("model_max_length").getAsInt();
        }
    }

    private static void loadSpecialTokensMap(Path path, Builder builder) throws IOException {
        String content = Files.readString(path);
        JsonObject tokens = GSON.fromJson(content, JsonObject.class);

        int pad = getTokenId(tokens, "pad_token", builder.vocab, 0);
        int cls = getTokenId(tokens, "cls_token", builder.vocab, -1);
        int sep = getTokenId(tokens, "sep_token", builder.vocab, -1);
        int unk = getTokenId(tokens, "unk_token", builder.vocab, 0);
        int mask = getTokenId(tokens, "mask_token", builder.vocab, -1);
        int bos = getTokenId(tokens, "bos_token", builder.vocab, -1);
        int eos = getTokenId(tokens, "eos_token", builder.vocab, -1);

        builder.specialTokens = new SpecialTokens(pad, cls, sep, unk, mask, bos, eos);
    }

    private static int getTokenId(JsonObject tokens, String key, Map<String, Integer> vocab, int defaultId) {
        if (!tokens.has(key)) return defaultId;

        JsonElement elem = tokens.get(key);
        String tokenStr;
        if (elem.isJsonObject()) {
            tokenStr = elem.getAsJsonObject().get("content").getAsString();
        } else {
            tokenStr = elem.getAsString();
        }

        return vocab.getOrDefault(tokenStr, defaultId);
    }

    @Override
    public EncodedInput encode(String text) {
        return encode(text, EncodeOptions.defaults());
    }

    @Override
    public EncodedInput encode(String text, EncodeOptions options) {
        List<Integer> ids = new ArrayList<>();

        // Add special tokens
        if (options.addSpecialTokens() && specialTokens.clsTokenId() >= 0) {
            ids.add(specialTokens.clsTokenId());
        }

        // Tokenize based on type
        List<String> tokens = tokenize(text);
        for (String token : tokens) {
            Integer id = vocab.get(token);
            if (id != null) {
                ids.add(id);
            } else if (type == TokenizerType.WORDPIECE) {
                // WordPiece subword tokenization
                ids.addAll(wordPieceTokenize(token));
            } else {
                ids.add(specialTokens.unkTokenId());
            }
        }

        // Add SEP token
        if (options.addSpecialTokens() && specialTokens.sepTokenId() >= 0) {
            ids.add(specialTokens.sepTokenId());
        }

        // Truncate
        if (options.truncation() && ids.size() > options.maxLength()) {
            ids = ids.subList(0, options.maxLength());
        }

        // Determine target length
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

        for (int i = ids.size(); i < targetLength; i++) {
            inputIds[i] = specialTokens.padTokenId();
            attentionMask[i] = 0;
        }

        return EncodedInput.of(inputIds, attentionMask);
    }

    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();

        if (tokenPattern != null) {
            Matcher matcher = tokenPattern.matcher(text);
            while (matcher.find()) {
                tokens.add(matcher.group());
            }
        } else {
            // Default: split on whitespace and punctuation
            String[] parts = text.toLowerCase().split("(?=[\\p{Punct}])|(?<=[\\p{Punct}])|\\s+");
            for (String part : parts) {
                if (!part.isEmpty()) {
                    tokens.add(part);
                }
            }
        }

        return tokens;
    }

    private List<Integer> wordPieceTokenize(String word) {
        List<Integer> ids = new ArrayList<>();
        String remaining = word.toLowerCase();

        while (!remaining.isEmpty()) {
            String found = null;
            int foundLen = 0;

            // Find longest matching token
            for (int end = remaining.length(); end > 0; end--) {
                String candidate = remaining.substring(0, end);
                if (ids.isEmpty()) {
                    if (vocab.containsKey(candidate)) {
                        found = candidate;
                        foundLen = end;
                        break;
                    }
                } else {
                    String subword = "##" + candidate;
                    if (vocab.containsKey(subword)) {
                        found = subword;
                        foundLen = end;
                        break;
                    }
                }
            }

            if (found != null) {
                ids.add(vocab.get(found));
                remaining = remaining.substring(foundLen);
            } else {
                // Unknown token for remaining
                ids.add(specialTokens.unkTokenId());
                break;
            }
        }

        return ids;
    }

    @Override
    public EncodedInput encodePair(String text, String textPair) {
        List<Integer> ids = new ArrayList<>();

        if (specialTokens.clsTokenId() >= 0) {
            ids.add(specialTokens.clsTokenId());
        }

        for (String token : tokenize(text)) {
            ids.addAll(wordPieceTokenize(token));
        }

        if (specialTokens.sepTokenId() >= 0) {
            ids.add(specialTokens.sepTokenId());
        }

        for (String token : tokenize(textPair)) {
            ids.addAll(wordPieceTokenize(token));
        }

        if (specialTokens.sepTokenId() >= 0) {
            ids.add(specialTokens.sepTokenId());
        }

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

            // Handle WordPiece continuation tokens
            if (token.startsWith("##")) {
                sb.append(token.substring(2));
            } else {
                if (sb.length() > 0) {
                    sb.append(" ");
                }
                sb.append(token);
            }
        }

        return sb.toString();
    }

    private boolean isSpecialToken(int id) {
        return id == specialTokens.padTokenId() ||
                id == specialTokens.clsTokenId() ||
                id == specialTokens.sepTokenId() ||
                id == specialTokens.unkTokenId() ||
                id == specialTokens.maskTokenId() ||
                id == specialTokens.bosTokenId() ||
                id == specialTokens.eosTokenId();
    }

    @Override
    public int tokenToId(String token) {
        return vocab.getOrDefault(token, specialTokens.unkTokenId());
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

    /**
     * Tokenizer type.
     */
    enum TokenizerType {
        WORDPIECE,
        BPE,
        SENTENCEPIECE
    }

    /**
     * Builder for HuggingFaceTokenizer.
     */
    private static class Builder {
        Map<String, Integer> vocab = new HashMap<>();
        SpecialTokens specialTokens = SpecialTokens.defaults();
        int maxLength = 512;
        TokenizerType type = TokenizerType.WORDPIECE;
        Map<String, String> merges = new HashMap<>();
        Pattern tokenPattern = null;

        HuggingFaceTokenizer build() {
            return new HuggingFaceTokenizer(this);
        }
    }
}
