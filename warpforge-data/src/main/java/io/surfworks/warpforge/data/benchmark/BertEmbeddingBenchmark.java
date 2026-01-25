package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.WarpForge;
import io.surfworks.warpforge.data.hub.ProgressListener;
import io.surfworks.warpforge.data.model.ModelSource;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Benchmark for BERT embedding lookup and basic operations.
 *
 * <p>This benchmark:
 * <ul>
 *   <li>Loads a BERT model (tiny version for fast testing)</li>
 *   <li>Performs embedding lookup from model weights</li>
 *   <li>Combines word, position, and token type embeddings</li>
 *   <li>Applies layer normalization</li>
 * </ul>
 *
 * <p>This tests a realistic subset of BERT computation without requiring
 * the full StableHLO pipeline.
 *
 * <p>Example:
 * <pre>{@code
 * BertEmbeddingBenchmark benchmark = BertEmbeddingBenchmark.builder("bert-embed")
 *     .modelId("hf-internal-testing/tiny-random-bert")
 *     .build();
 *
 * BenchmarkResult result = runner.run(benchmark, config);
 * }</pre>
 */
public final class BertEmbeddingBenchmark implements ModelBenchmark {

    private final String name;
    private final String modelId;

    private ModelSource model;
    private Arena arena;
    private MemorySegment inputIds;
    private MemorySegment attentionMask;
    private MemorySegment outputBuffer;

    // Model config (populated from config.json)
    private int vocabSize = 30522;
    private int hiddenSize = 768;
    private int maxPositionEmbeddings = 512;
    private int typeVocabSize = 2;
    private float layerNormEps = 1e-12f;

    private BertEmbeddingBenchmark(Builder builder) {
        this.name = builder.name;
        this.modelId = builder.modelId;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public String modelId() {
        return modelId;
    }

    @Override
    public void setup(BenchmarkConfig config) throws IOException {
        // Configure WarpForge
        WarpForge.configure()
                .progressListener(ProgressListener.NONE)
                .apply();

        // Load model
        model = WarpForge.model(modelId);

        // Extract config if available
        Map<String, Object> metadata = model.metadata();
        vocabSize = extractInt(metadata, "vocab_size", vocabSize);
        hiddenSize = extractInt(metadata, "hidden_size", hiddenSize);
        maxPositionEmbeddings = extractInt(metadata, "max_position_embeddings", maxPositionEmbeddings);
        typeVocabSize = extractInt(metadata, "type_vocab_size", typeVocabSize);

        // Allocate working memory
        arena = Arena.ofShared();

        int batchSize = config.batchSize();
        int seqLen = config.sequenceLength();

        // Input tensors (int32)
        inputIds = arena.allocate((long) batchSize * seqLen * 4);
        attentionMask = arena.allocate((long) batchSize * seqLen * 4);

        // Output buffer (float32)
        outputBuffer = arena.allocate((long) batchSize * seqLen * hiddenSize * 4);

        // Initialize input_ids with random token IDs
        Random random = new Random(42);
        for (int i = 0; i < batchSize * seqLen; i++) {
            inputIds.setAtIndex(ValueLayout.JAVA_INT, i, random.nextInt(vocabSize));
            attentionMask.setAtIndex(ValueLayout.JAVA_INT, i, 1); // All tokens attended
        }
    }

    @Override
    public Map<String, TensorView> prepareInputs(BenchmarkConfig config) {
        int batchSize = config.batchSize();
        int seqLen = config.sequenceLength();

        TensorInfo inputIdsInfo = new TensorInfo("input_ids", DType.I32,
                new long[]{batchSize, seqLen}, 0, inputIds.byteSize());
        TensorInfo attentionMaskInfo = new TensorInfo("attention_mask", DType.I32,
                new long[]{batchSize, seqLen}, 0, attentionMask.byteSize());

        return Map.of(
                "input_ids", new TensorView(inputIds, inputIdsInfo),
                "attention_mask", new TensorView(attentionMask, attentionMaskInfo)
        );
    }

    @Override
    public Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException {
        TensorView inputIdsTensor = inputs.get("input_ids");
        long batchSize = inputIdsTensor.shape()[0];
        long seqLen = inputIdsTensor.shape()[1];

        // Get embedding weights
        TensorView wordEmbed = getEmbeddingTensor("word_embeddings");
        TensorView posEmbed = getEmbeddingTensor("position_embeddings");
        TensorView tokenTypeEmbed = getEmbeddingTensor("token_type_embeddings");
        TensorView layerNormWeight = getLayerNormTensor("weight");
        TensorView layerNormBias = getLayerNormTensor("bias");

        // Perform embedding lookup and combination
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int tokenId = inputIdsTensor.getInt(b, s);
                int positionId = (int) s;
                int tokenTypeId = 0; // Assume single segment

                // Lookup embeddings and combine
                for (int h = 0; h < hiddenSize; h++) {
                    float wordVal = wordEmbed.getFloat(tokenId, h);
                    float posVal = posEmbed.getFloat(positionId, h);
                    float typeVal = tokenTypeEmbed.getFloat(tokenTypeId, h);

                    float combined = wordVal + posVal + typeVal;

                    // Store in output buffer
                    long outIdx = (b * seqLen + s) * hiddenSize + h;
                    outputBuffer.setAtIndex(ValueLayout.JAVA_FLOAT, outIdx, combined);
                }
            }
        }

        // Apply layer normalization
        applyLayerNorm(batchSize, seqLen, layerNormWeight, layerNormBias);

        // Return output
        TensorInfo outputInfo = new TensorInfo("embeddings", DType.F32,
                new long[]{batchSize, seqLen, hiddenSize}, 0, outputBuffer.byteSize());

        Map<String, TensorView> outputs = new HashMap<>();
        outputs.put("embeddings", new TensorView(outputBuffer, outputInfo));
        return outputs;
    }

    private void applyLayerNorm(long batchSize, long seqLen,
                                 TensorView weight, TensorView bias) {
        // Apply layer normalization to each position
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                long baseIdx = (b * seqLen + s) * hiddenSize;

                // Compute mean
                double sum = 0;
                for (int h = 0; h < hiddenSize; h++) {
                    sum += outputBuffer.getAtIndex(ValueLayout.JAVA_FLOAT, baseIdx + h);
                }
                float mean = (float) (sum / hiddenSize);

                // Compute variance
                double varSum = 0;
                for (int h = 0; h < hiddenSize; h++) {
                    float val = outputBuffer.getAtIndex(ValueLayout.JAVA_FLOAT, baseIdx + h);
                    float diff = val - mean;
                    varSum += diff * diff;
                }
                float variance = (float) (varSum / hiddenSize);
                float std = (float) Math.sqrt(variance + layerNormEps);

                // Normalize and apply weight/bias
                for (int h = 0; h < hiddenSize; h++) {
                    float val = outputBuffer.getAtIndex(ValueLayout.JAVA_FLOAT, baseIdx + h);
                    float normalized = (val - mean) / std;
                    float scaled = normalized * weight.getFloat(h) + bias.getFloat(h);
                    outputBuffer.setAtIndex(ValueLayout.JAVA_FLOAT, baseIdx + h, scaled);
                }
            }
        }
    }

    private TensorView getEmbeddingTensor(String name) {
        // Try different naming conventions used by BERT models
        String[] prefixes = {
                "bert.embeddings.", "embeddings.", "transformer.embeddings.",
                "encoder.embeddings.", ""
        };

        for (String prefix : prefixes) {
            String fullName = prefix + name;
            if (model.hasTensor(fullName)) {
                return model.tensor(fullName);
            }
            // Also try with .weight suffix
            if (model.hasTensor(fullName + ".weight")) {
                return model.tensor(fullName + ".weight");
            }
        }

        throw new IllegalStateException("Could not find embedding tensor: " + name +
                ". Available tensors: " + model.tensorNames());
    }

    /**
     * Extract an integer value from metadata, handling both Number and JsonPrimitive.
     */
    private static int extractInt(Map<String, Object> metadata, String key, int defaultValue) {
        Object value = metadata.get(key);
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof Number n) {
            return n.intValue();
        }
        // Handle Gson JsonPrimitive
        if (value instanceof com.google.gson.JsonPrimitive jp) {
            return jp.getAsInt();
        }
        // Try parsing as string
        try {
            return Integer.parseInt(value.toString());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private TensorView getLayerNormTensor(String param) {
        String[] prefixes = {
                "bert.embeddings.LayerNorm.", "embeddings.LayerNorm.",
                "transformer.embeddings.LayerNorm.", "embeddings.layer_norm.",
                "bert.embeddings.layer_norm."
        };

        for (String prefix : prefixes) {
            String fullName = prefix + param;
            if (model.hasTensor(fullName)) {
                return model.tensor(fullName);
            }
        }

        throw new IllegalStateException("Could not find LayerNorm " + param +
                ". Available tensors: " + model.tensorNames());
    }

    @Override
    public List<String> outputsToValidate() {
        return List.of("embeddings");
    }

    @Override
    public void teardown() {
        if (model != null) {
            model.close();
            model = null;
        }
        if (arena != null) {
            arena.close();
            arena = null;
        }
    }

    /**
     * Create a builder for BertEmbeddingBenchmark.
     */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Builder for BertEmbeddingBenchmark.
     */
    public static final class Builder {
        private final String name;
        private String modelId = "hf-internal-testing/tiny-random-bert";

        Builder(String name) {
            this.name = name;
        }

        /**
         * Set the BERT model ID.
         * Default: "hf-internal-testing/tiny-random-bert" (tiny model for fast testing)
         */
        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public BertEmbeddingBenchmark build() {
            return new BertEmbeddingBenchmark(this);
        }
    }
}
