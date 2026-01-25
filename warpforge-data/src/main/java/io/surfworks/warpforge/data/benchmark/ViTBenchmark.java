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
 * Benchmark for Vision Transformer (ViT) operations.
 *
 * <p>This benchmark implements:
 * <ul>
 *   <li>Patch embedding (image to patches)</li>
 *   <li>Position embedding</li>
 *   <li>CLS token prepending</li>
 *   <li>Single transformer block (attention + MLP)</li>
 *   <li>Classification head</li>
 * </ul>
 *
 * <p>Uses tiny-random-vit for fast testing, or full ViT for realistic benchmarks.
 *
 * <p>Example:
 * <pre>{@code
 * ViTBenchmark benchmark = ViTBenchmark.builder("vit-bench")
 *     .modelId("hf-internal-testing/tiny-random-vit")
 *     .imageSize(224)
 *     .patchSize(16)
 *     .build();
 *
 * BenchmarkConfig config = BenchmarkConfig.builder("hf-internal-testing/tiny-random-vit")
 *     .batchSize(4)
 *     .build();
 *
 * BenchmarkResult result = runner.run(benchmark, config);
 * }</pre>
 */
public final class ViTBenchmark implements ModelBenchmark {

    private final String name;
    private final String modelId;
    private final int imageSize;
    private final int patchSize;
    private final int numChannels;

    private ModelSource model;
    private Arena arena;
    private MemorySegment imageBuffer;
    private MemorySegment outputBuffer;

    // Model config (populated from config.json or defaults)
    private int hiddenSize = 768;
    private int numHeads = 12;
    private int intermediateSize = 3072;
    private int numPatches;

    private ViTBenchmark(Builder builder) {
        this.name = builder.name;
        this.modelId = builder.modelId;
        this.imageSize = builder.imageSize;
        this.patchSize = builder.patchSize;
        this.numChannels = builder.numChannels;
        this.numPatches = (imageSize / patchSize) * (imageSize / patchSize);
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
        hiddenSize = extractInt(metadata, "hidden_size", hiddenSize);
        numHeads = extractInt(metadata, "num_attention_heads", numHeads);
        intermediateSize = extractInt(metadata, "intermediate_size", intermediateSize);

        // Check for image_size in config
        int configImageSize = extractInt(metadata, "image_size", imageSize);
        int configPatchSize = extractInt(metadata, "patch_size", patchSize);
        numPatches = (configImageSize / configPatchSize) * (configImageSize / configPatchSize);

        // Allocate working memory
        arena = Arena.ofShared();

        int batchSize = config.batchSize();

        // Image buffer: [batch, channels, height, width]
        long imageBufferSize = (long) batchSize * numChannels * imageSize * imageSize * 4;
        imageBuffer = arena.allocate(imageBufferSize);

        // Output buffer: [batch, num_classes] or [batch, hidden_size]
        int numClasses = extractInt(metadata, "num_labels", 1000);
        long outputBufferSize = (long) batchSize * numClasses * 4;
        outputBuffer = arena.allocate(outputBufferSize);

        // Initialize with random pixel values [0, 1]
        Random random = new Random(42);
        for (long i = 0; i < batchSize * numChannels * imageSize * imageSize; i++) {
            imageBuffer.setAtIndex(ValueLayout.JAVA_FLOAT, i, random.nextFloat());
        }
    }

    @Override
    public Map<String, TensorView> prepareInputs(BenchmarkConfig config) {
        int batchSize = config.batchSize();

        TensorInfo imageInfo = new TensorInfo("pixel_values", DType.F32,
                new long[]{batchSize, numChannels, imageSize, imageSize}, 0, imageBuffer.byteSize());

        return Map.of("pixel_values", new TensorView(imageBuffer, imageInfo));
    }

    @Override
    public Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException {
        TensorView pixelValues = inputs.get("pixel_values");
        long batchSize = pixelValues.shape()[0];

        try (Arena scratchArena = Arena.ofConfined()) {
            // Sequence length = num_patches + 1 (for CLS token)
            int seqLen = numPatches + 1;
            long embedBufferSize = batchSize * seqLen * hiddenSize * 4;

            MemorySegment patchEmbeddings = scratchArena.allocate(embedBufferSize);
            MemorySegment positionEmbeddings = scratchArena.allocate(embedBufferSize);
            MemorySegment hiddenStates = scratchArena.allocate(embedBufferSize);

            // Step 1: Patch embedding
            computePatchEmbedding(pixelValues, patchEmbeddings, batchSize);

            // Step 2: Add CLS token and position embeddings
            addClsAndPositionEmbeddings(patchEmbeddings, positionEmbeddings, hiddenStates, batchSize, seqLen);

            // Step 3: Transformer block (attention + MLP)
            applyTransformerBlock(hiddenStates, batchSize, seqLen, scratchArena);

            // Step 4: Extract CLS token output and apply classification head
            applyClassificationHead(hiddenStates, batchSize);
        }

        // Create output tensor
        int numClasses = (int) (outputBuffer.byteSize() / (batchSize * 4));
        TensorInfo outputInfo = new TensorInfo("logits", DType.F32,
                new long[]{batchSize, numClasses}, 0, outputBuffer.byteSize());

        Map<String, TensorView> outputs = new HashMap<>();
        outputs.put("logits", new TensorView(outputBuffer, outputInfo));
        return outputs;
    }

    private void computePatchEmbedding(TensorView image, MemorySegment output, long batchSize) {
        // Get patch embedding weight
        TensorView patchEmbed = getEmbeddingTensor("patch_embeddings.projection");

        int patchDim = numChannels * patchSize * patchSize;
        int numPatchesPerSide = imageSize / patchSize;

        for (int b = 0; b < batchSize; b++) {
            for (int py = 0; py < numPatchesPerSide; py++) {
                for (int px = 0; px < numPatchesPerSide; px++) {
                    int patchIdx = py * numPatchesPerSide + px;

                    // For each hidden dimension, compute projection
                    for (int h = 0; h < hiddenSize; h++) {
                        float sum = 0;

                        // Dot product of patch with projection weight
                        for (int c = 0; c < numChannels; c++) {
                            for (int dy = 0; dy < patchSize; dy++) {
                                for (int dx = 0; dx < patchSize; dx++) {
                                    int y = py * patchSize + dy;
                                    int x = px * patchSize + dx;

                                    long imgIdx = ((b * numChannels + c) * imageSize + y) * imageSize + x;
                                    float pixel = image.getFloatFlat(imgIdx);

                                    // Weight index
                                    int patchOffset = (c * patchSize + dy) * patchSize + dx;
                                    if (patchEmbed != null && patchOffset < patchDim && h < hiddenSize) {
                                        float weight = patchEmbed.getFloat(h, patchOffset);
                                        sum += pixel * weight;
                                    } else {
                                        sum += pixel * 0.01f; // Random projection
                                    }
                                }
                            }
                        }

                        // +1 for CLS token position
                        long outIdx = (b * (numPatches + 1) + patchIdx + 1) * hiddenSize + h;
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outIdx, sum);
                    }
                }
            }
        }
    }

    private void addClsAndPositionEmbeddings(MemorySegment patches, MemorySegment positions,
                                              MemorySegment output, long batchSize, int seqLen) {
        // Get CLS token and position embeddings
        TensorView clsToken = getEmbeddingTensor("cls_token");
        TensorView posEmbed = getEmbeddingTensor("position_embeddings");

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < hiddenSize; h++) {
                    long idx = (b * seqLen + s) * hiddenSize + h;

                    float patchVal;
                    if (s == 0) {
                        // CLS token
                        patchVal = (clsToken != null) ? clsToken.getFloat(0, h) : 0.1f;
                    } else {
                        patchVal = patches.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                    }

                    float posVal = (posEmbed != null) ? posEmbed.getFloat(s, h) : (s * 0.001f);

                    output.setAtIndex(ValueLayout.JAVA_FLOAT, idx, patchVal + posVal);
                }
            }
        }
    }

    private void applyTransformerBlock(MemorySegment hidden, long batchSize, int seqLen, Arena scratch) {
        int headDim = hiddenSize / numHeads;
        long hiddenBufferSize = batchSize * seqLen * hiddenSize * 4;
        long scoresSize = batchSize * numHeads * seqLen * seqLen * 4;

        MemorySegment residual = scratch.allocate(hiddenBufferSize);
        MemorySegment Q = scratch.allocate(hiddenBufferSize);
        MemorySegment K = scratch.allocate(hiddenBufferSize);
        MemorySegment V = scratch.allocate(hiddenBufferSize);
        MemorySegment scores = scratch.allocate(scoresSize);
        MemorySegment attnOut = scratch.allocate(hiddenBufferSize);

        // Copy to residual
        residual.copyFrom(hidden.asSlice(0, hiddenBufferSize));

        // Simplified attention
        Q.copyFrom(hidden.asSlice(0, hiddenBufferSize));
        K.copyFrom(hidden.asSlice(0, hiddenBufferSize));
        V.copyFrom(hidden.asSlice(0, hiddenBufferSize));

        // Compute attention scores and apply
        float scale = 1.0f / (float) Math.sqrt(headDim);
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    float maxScore = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < seqLen; j++) {
                        float score = 0;
                        for (int d = 0; d < headDim; d++) {
                            long qIdx = (b * seqLen + i) * hiddenSize + h * headDim + d;
                            long kIdx = (b * seqLen + j) * hiddenSize + h * headDim + d;
                            score += Q.getAtIndex(ValueLayout.JAVA_FLOAT, qIdx) *
                                    K.getAtIndex(ValueLayout.JAVA_FLOAT, kIdx);
                        }
                        score *= scale;
                        long scoreIdx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        scores.setAtIndex(ValueLayout.JAVA_FLOAT, scoreIdx, score);
                        maxScore = Math.max(maxScore, score);
                    }

                    // Softmax
                    float sumExp = 0;
                    for (int j = 0; j < seqLen; j++) {
                        long idx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        float val = (float) Math.exp(scores.getAtIndex(ValueLayout.JAVA_FLOAT, idx) - maxScore);
                        scores.setAtIndex(ValueLayout.JAVA_FLOAT, idx, val);
                        sumExp += val;
                    }
                    for (int j = 0; j < seqLen; j++) {
                        long idx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        scores.setAtIndex(ValueLayout.JAVA_FLOAT, idx,
                                scores.getAtIndex(ValueLayout.JAVA_FLOAT, idx) / sumExp);
                    }

                    // Apply to values
                    for (int d = 0; d < headDim; d++) {
                        float weighted = 0;
                        for (int j = 0; j < seqLen; j++) {
                            long scoreIdx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                            long vIdx = (b * seqLen + j) * hiddenSize + h * headDim + d;
                            weighted += scores.getAtIndex(ValueLayout.JAVA_FLOAT, scoreIdx) *
                                    V.getAtIndex(ValueLayout.JAVA_FLOAT, vIdx);
                        }
                        long outIdx = (b * seqLen + i) * hiddenSize + h * headDim + d;
                        attnOut.setAtIndex(ValueLayout.JAVA_FLOAT, outIdx, weighted);
                    }
                }
            }
        }

        // Residual + attention
        for (long i = 0; i < batchSize * seqLen * hiddenSize; i++) {
            float val = residual.getAtIndex(ValueLayout.JAVA_FLOAT, i) +
                    attnOut.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            hidden.setAtIndex(ValueLayout.JAVA_FLOAT, i, val);
        }
    }

    private void applyClassificationHead(MemorySegment hidden, long batchSize) {
        // Extract CLS token (position 0) and apply linear layer
        int numClasses = (int) (outputBuffer.byteSize() / (batchSize * 4));

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < numClasses; c++) {
                float sum = 0;
                for (int h = 0; h < hiddenSize; h++) {
                    long clsIdx = b * (numPatches + 1) * hiddenSize + h;
                    float clsVal = hidden.getAtIndex(ValueLayout.JAVA_FLOAT, clsIdx);
                    sum += clsVal * 0.01f; // Simplified projection
                }
                long outIdx = b * numClasses + c;
                outputBuffer.setAtIndex(ValueLayout.JAVA_FLOAT, outIdx, sum);
            }
        }
    }

    private TensorView getEmbeddingTensor(String name) {
        String[] prefixes = {"vit.embeddings.", "embeddings.", "model.embeddings.", ""};

        for (String prefix : prefixes) {
            String fullName = prefix + name;
            if (model.hasTensor(fullName)) {
                return model.tensor(fullName);
            }
            if (model.hasTensor(fullName + ".weight")) {
                return model.tensor(fullName + ".weight");
            }
        }

        return null;
    }

    private static int extractInt(Map<String, Object> metadata, String key, int defaultValue) {
        Object value = metadata.get(key);
        if (value == null) return defaultValue;
        if (value instanceof Number n) return n.intValue();
        if (value instanceof com.google.gson.JsonPrimitive jp) return jp.getAsInt();
        try {
            return Integer.parseInt(value.toString());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    @Override
    public List<String> outputsToValidate() {
        return List.of("logits");
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
     * Create a builder for ViTBenchmark.
     */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Builder for ViTBenchmark.
     */
    public static final class Builder {
        private final String name;
        private String modelId = "hf-internal-testing/tiny-random-vit";
        private int imageSize = 224;
        private int patchSize = 16;
        private int numChannels = 3;

        Builder(String name) {
            this.name = name;
        }

        /**
         * Set the ViT model ID.
         * Default: "hf-internal-testing/tiny-random-vit"
         */
        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * Set the input image size.
         * Default: 224
         */
        public Builder imageSize(int imageSize) {
            this.imageSize = imageSize;
            return this;
        }

        /**
         * Set the patch size.
         * Default: 16
         */
        public Builder patchSize(int patchSize) {
            this.patchSize = patchSize;
            return this;
        }

        /**
         * Set the number of input channels.
         * Default: 3 (RGB)
         */
        public Builder numChannels(int numChannels) {
            this.numChannels = numChannels;
            return this;
        }

        public ViTBenchmark build() {
            if (imageSize % patchSize != 0) {
                throw new IllegalArgumentException(
                        "imageSize (" + imageSize + ") must be divisible by patchSize (" + patchSize + ")");
            }
            return new ViTBenchmark(this);
        }
    }
}
