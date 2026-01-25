package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Benchmark for standalone attention layer operations.
 *
 * <p>This benchmark implements multi-head self-attention with:
 * <ul>
 *   <li>Query, Key, Value projections</li>
 *   <li>Scaled dot-product attention</li>
 *   <li>Optional causal masking</li>
 *   <li>Output projection</li>
 * </ul>
 *
 * <p>This is a synthetic benchmark that does not load an actual model,
 * making it suitable for measuring pure attention computation performance.
 *
 * <p>Example:
 * <pre>{@code
 * AttentionBenchmark benchmark = AttentionBenchmark.builder("attn-bench")
 *     .hiddenSize(768)
 *     .numHeads(12)
 *     .causal(true)
 *     .build();
 *
 * BenchmarkConfig config = BenchmarkConfig.builder("synthetic")
 *     .batchSize(8)
 *     .sequenceLength(512)
 *     .build();
 *
 * BenchmarkResult result = runner.run(benchmark, config);
 * }</pre>
 */
public final class AttentionBenchmark implements ModelBenchmark {

    private final String name;
    private final int hiddenSize;
    private final int numHeads;
    private final boolean causal;
    private final boolean useFlashAttention;

    private Arena arena;
    private MemorySegment queryWeight;
    private MemorySegment keyWeight;
    private MemorySegment valueWeight;
    private MemorySegment outputWeight;
    private MemorySegment inputBuffer;
    private MemorySegment outputBuffer;

    private AttentionBenchmark(Builder builder) {
        this.name = builder.name;
        this.hiddenSize = builder.hiddenSize;
        this.numHeads = builder.numHeads;
        this.causal = builder.causal;
        this.useFlashAttention = builder.useFlashAttention;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public String modelId() {
        return "synthetic-attention";
    }

    @Override
    public void setup(BenchmarkConfig config) throws IOException {
        arena = Arena.ofShared();

        int batchSize = config.batchSize();
        int seqLen = config.sequenceLength();

        // Allocate weight matrices
        // Q, K, V projections: [hidden_size, hidden_size]
        long projSize = (long) hiddenSize * hiddenSize * 4;
        queryWeight = arena.allocate(projSize);
        keyWeight = arena.allocate(projSize);
        valueWeight = arena.allocate(projSize);
        outputWeight = arena.allocate(projSize);

        // Initialize weights with random values
        Random random = new Random(42);
        initializeWeights(queryWeight, random);
        initializeWeights(keyWeight, random);
        initializeWeights(valueWeight, random);
        initializeWeights(outputWeight, random);

        // Allocate input and output buffers
        long bufferSize = (long) batchSize * seqLen * hiddenSize * 4;
        inputBuffer = arena.allocate(bufferSize);
        outputBuffer = arena.allocate(bufferSize);

        // Initialize input with random values
        for (long i = 0; i < batchSize * seqLen * hiddenSize; i++) {
            inputBuffer.setAtIndex(ValueLayout.JAVA_FLOAT, i, random.nextFloat() * 0.02f - 0.01f);
        }
    }

    private void initializeWeights(MemorySegment weights, Random random) {
        float scale = (float) Math.sqrt(2.0 / hiddenSize);
        long count = weights.byteSize() / 4;
        for (long i = 0; i < count; i++) {
            weights.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) random.nextGaussian() * scale);
        }
    }

    @Override
    public Map<String, TensorView> prepareInputs(BenchmarkConfig config) {
        int batchSize = config.batchSize();
        int seqLen = config.sequenceLength();

        TensorInfo inputInfo = new TensorInfo("hidden_states", DType.F32,
                new long[]{batchSize, seqLen, hiddenSize}, 0, inputBuffer.byteSize());

        return Map.of("hidden_states", new TensorView(inputBuffer, inputInfo));
    }

    @Override
    public Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException {
        TensorView input = inputs.get("hidden_states");
        long batchSize = input.shape()[0];
        long seqLen = input.shape()[1];
        int headDim = hiddenSize / numHeads;

        // Allocate scratch space for Q, K, V and attention scores
        try (Arena scratchArena = Arena.ofConfined()) {
            long projBufferSize = batchSize * seqLen * hiddenSize * 4;
            MemorySegment Q = scratchArena.allocate(projBufferSize);
            MemorySegment K = scratchArena.allocate(projBufferSize);
            MemorySegment V = scratchArena.allocate(projBufferSize);
            MemorySegment attnOutput = scratchArena.allocate(projBufferSize);

            // Attention scores: [batch, num_heads, seq_len, seq_len]
            long scoresSize = batchSize * numHeads * seqLen * seqLen * 4;
            MemorySegment scores = scratchArena.allocate(scoresSize);

            // Step 1: Compute Q, K, V projections
            computeProjection(inputBuffer, queryWeight, Q, batchSize, seqLen);
            computeProjection(inputBuffer, keyWeight, K, batchSize, seqLen);
            computeProjection(inputBuffer, valueWeight, V, batchSize, seqLen);

            // Step 2: Compute attention scores
            if (useFlashAttention) {
                computeFlashAttention(Q, K, V, attnOutput, batchSize, seqLen, headDim, scores);
            } else {
                computeStandardAttention(Q, K, V, attnOutput, batchSize, seqLen, headDim, scores);
            }

            // Step 3: Output projection
            computeProjection(attnOutput, outputWeight, outputBuffer, batchSize, seqLen);
        }

        // Create output tensor
        TensorInfo outputInfo = new TensorInfo("attention_output", DType.F32,
                new long[]{batchSize, seqLen, hiddenSize}, 0, outputBuffer.byteSize());

        Map<String, TensorView> outputs = new HashMap<>();
        outputs.put("attention_output", new TensorView(outputBuffer, outputInfo));
        return outputs;
    }

    private void computeProjection(MemorySegment input, MemorySegment weight,
                                    MemorySegment output, long batchSize, long seqLen) {
        // Matrix multiply: output[b,s,:] = input[b,s,:] @ weight
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int o = 0; o < hiddenSize; o++) {
                    float sum = 0;
                    for (int i = 0; i < hiddenSize; i++) {
                        long inputIdx = (b * seqLen + s) * hiddenSize + i;
                        long weightIdx = (long) i * hiddenSize + o;
                        sum += input.getAtIndex(ValueLayout.JAVA_FLOAT, inputIdx) *
                                weight.getAtIndex(ValueLayout.JAVA_FLOAT, weightIdx);
                    }
                    long outIdx = (b * seqLen + s) * hiddenSize + o;
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outIdx, sum);
                }
            }
        }
    }

    private void computeStandardAttention(MemorySegment Q, MemorySegment K, MemorySegment V,
                                           MemorySegment output, long batchSize, long seqLen,
                                           int headDim, MemorySegment scores) {
        float scale = 1.0f / (float) Math.sqrt(headDim);

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Compute attention scores: Q @ K^T
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        float score = 0;
                        for (int d = 0; d < headDim; d++) {
                            long qIdx = (b * seqLen + i) * hiddenSize + h * headDim + d;
                            long kIdx = (b * seqLen + j) * hiddenSize + h * headDim + d;
                            score += Q.getAtIndex(ValueLayout.JAVA_FLOAT, qIdx) *
                                    K.getAtIndex(ValueLayout.JAVA_FLOAT, kIdx);
                        }
                        score *= scale;

                        // Apply causal mask
                        if (causal && j > i) {
                            score = Float.NEGATIVE_INFINITY;
                        }

                        long scoreIdx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        scores.setAtIndex(ValueLayout.JAVA_FLOAT, scoreIdx, score);
                    }
                }

                // Softmax over j dimension
                for (int i = 0; i < seqLen; i++) {
                    int validLen = causal ? (int) (i + 1) : (int) seqLen;

                    // Find max
                    float maxVal = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < validLen; j++) {
                        long idx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        maxVal = Math.max(maxVal, scores.getAtIndex(ValueLayout.JAVA_FLOAT, idx));
                    }

                    // Exp and sum
                    float sum = 0;
                    for (int j = 0; j < validLen; j++) {
                        long idx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        float val = (float) Math.exp(scores.getAtIndex(ValueLayout.JAVA_FLOAT, idx) - maxVal);
                        scores.setAtIndex(ValueLayout.JAVA_FLOAT, idx, val);
                        sum += val;
                    }

                    // Normalize
                    for (int j = 0; j < validLen; j++) {
                        long idx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        float val = scores.getAtIndex(ValueLayout.JAVA_FLOAT, idx) / sum;
                        scores.setAtIndex(ValueLayout.JAVA_FLOAT, idx, val);
                    }
                }

                // Compute attention output: scores @ V
                for (int i = 0; i < seqLen; i++) {
                    int validLen = causal ? (int) (i + 1) : (int) seqLen;

                    for (int d = 0; d < headDim; d++) {
                        float weighted = 0;
                        for (int j = 0; j < validLen; j++) {
                            long scoreIdx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                            long vIdx = (b * seqLen + j) * hiddenSize + h * headDim + d;
                            weighted += scores.getAtIndex(ValueLayout.JAVA_FLOAT, scoreIdx) *
                                    V.getAtIndex(ValueLayout.JAVA_FLOAT, vIdx);
                        }
                        long outIdx = (b * seqLen + i) * hiddenSize + h * headDim + d;
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outIdx, weighted);
                    }
                }
            }
        }
    }

    private void computeFlashAttention(MemorySegment Q, MemorySegment K, MemorySegment V,
                                        MemorySegment output, long batchSize, long seqLen,
                                        int headDim, MemorySegment scores) {
        // Flash attention is a memory-efficient attention algorithm that computes
        // attention in blocks to reduce memory usage. This is a simplified version
        // that demonstrates the block-based approach.

        int blockSize = 64; // Block size for tiling
        float scale = 1.0f / (float) Math.sqrt(headDim);

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Initialize output and normalization accumulators
                float[] outAccum = new float[(int) seqLen * headDim];
                float[] maxAccum = new float[(int) seqLen];
                float[] sumAccum = new float[(int) seqLen];
                java.util.Arrays.fill(maxAccum, Float.NEGATIVE_INFINITY);

                // Process K, V in blocks
                for (int kStart = 0; kStart < seqLen; kStart += blockSize) {
                    int kEnd = (int) Math.min(kStart + blockSize, seqLen);

                    // Process Q in blocks
                    for (int qStart = 0; qStart < seqLen; qStart += blockSize) {
                        int qEnd = (int) Math.min(qStart + blockSize, seqLen);

                        // Skip if causal mask would block all
                        if (causal && kStart > qEnd - 1) {
                            continue;
                        }

                        // Compute attention scores for this block
                        for (int i = qStart; i < qEnd; i++) {
                            float localMax = Float.NEGATIVE_INFINITY;

                            // Compute scores and find max
                            for (int j = kStart; j < kEnd; j++) {
                                if (causal && j > i) break;

                                float score = 0;
                                for (int d = 0; d < headDim; d++) {
                                    long qIdx = (b * seqLen + i) * hiddenSize + h * headDim + d;
                                    long kIdx = (b * seqLen + j) * hiddenSize + h * headDim + d;
                                    score += Q.getAtIndex(ValueLayout.JAVA_FLOAT, qIdx) *
                                            K.getAtIndex(ValueLayout.JAVA_FLOAT, kIdx);
                                }
                                score *= scale;

                                int scoreIdx = (j - kStart);
                                scores.setAtIndex(ValueLayout.JAVA_FLOAT, scoreIdx, score);
                                localMax = Math.max(localMax, score);
                            }

                            // Update running max and rescale previous accumulator
                            float prevMax = maxAccum[i];
                            float newMax = Math.max(prevMax, localMax);

                            if (prevMax != Float.NEGATIVE_INFINITY) {
                                float rescale = (float) Math.exp(prevMax - newMax);
                                sumAccum[i] *= rescale;
                                for (int d = 0; d < headDim; d++) {
                                    outAccum[i * headDim + d] *= rescale;
                                }
                            }
                            maxAccum[i] = newMax;

                            // Compute exp and accumulate
                            for (int j = kStart; j < kEnd; j++) {
                                if (causal && j > i) break;

                                int scoreIdx = (j - kStart);
                                float score = scores.getAtIndex(ValueLayout.JAVA_FLOAT, scoreIdx);
                                float expScore = (float) Math.exp(score - newMax);
                                sumAccum[i] += expScore;

                                for (int d = 0; d < headDim; d++) {
                                    long vIdx = (b * seqLen + j) * hiddenSize + h * headDim + d;
                                    outAccum[i * headDim + d] += expScore *
                                            V.getAtIndex(ValueLayout.JAVA_FLOAT, vIdx);
                                }
                            }
                        }
                    }
                }

                // Normalize and write output
                for (int i = 0; i < seqLen; i++) {
                    for (int d = 0; d < headDim; d++) {
                        float val = outAccum[i * headDim + d] / sumAccum[i];
                        long outIdx = (b * seqLen + i) * hiddenSize + h * headDim + d;
                        output.setAtIndex(ValueLayout.JAVA_FLOAT, outIdx, val);
                    }
                }
            }
        }
    }

    @Override
    public List<String> outputsToValidate() {
        return List.of("attention_output");
    }

    @Override
    public void teardown() {
        if (arena != null) {
            arena.close();
            arena = null;
        }
    }

    /**
     * Create a builder for AttentionBenchmark.
     */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Builder for AttentionBenchmark.
     */
    public static final class Builder {
        private final String name;
        private int hiddenSize = 768;
        private int numHeads = 12;
        private boolean causal = true;
        private boolean useFlashAttention = false;

        Builder(String name) {
            this.name = name;
        }

        /**
         * Set the hidden size (embedding dimension).
         * Default: 768 (BERT/GPT-2 base)
         */
        public Builder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        /**
         * Set the number of attention heads.
         * Default: 12 (BERT/GPT-2 base)
         */
        public Builder numHeads(int numHeads) {
            this.numHeads = numHeads;
            return this;
        }

        /**
         * Whether to use causal masking (for autoregressive models).
         * Default: true
         */
        public Builder causal(boolean causal) {
            this.causal = causal;
            return this;
        }

        /**
         * Whether to use flash attention algorithm.
         * Default: false
         */
        public Builder useFlashAttention(boolean useFlashAttention) {
            this.useFlashAttention = useFlashAttention;
            return this;
        }

        public AttentionBenchmark build() {
            if (hiddenSize % numHeads != 0) {
                throw new IllegalArgumentException(
                        "hiddenSize (" + hiddenSize + ") must be divisible by numHeads (" + numHeads + ")");
            }
            return new AttentionBenchmark(this);
        }
    }
}
