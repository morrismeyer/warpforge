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
 * Benchmark for a complete transformer block.
 *
 * <p>This benchmark implements a full transformer block consisting of:
 * <ul>
 *   <li>Pre-LayerNorm (optional, for pre-norm architecture)</li>
 *   <li>Multi-head self-attention</li>
 *   <li>Residual connection</li>
 *   <li>LayerNorm</li>
 *   <li>Feed-forward network (MLP with GELU activation)</li>
 *   <li>Residual connection</li>
 *   <li>Post-LayerNorm (optional, for post-norm architecture)</li>
 * </ul>
 *
 * <p>This is a synthetic benchmark that uses random weights, making it suitable
 * for measuring pure transformer computation performance without model loading overhead.
 *
 * <p>Example:
 * <pre>{@code
 * TransformerBlockBenchmark benchmark = TransformerBlockBenchmark.builder("transformer-block")
 *     .hiddenSize(768)
 *     .intermediateSize(3072)
 *     .numHeads(12)
 *     .preNorm(true)
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
public final class TransformerBlockBenchmark implements ModelBenchmark {

    private final String name;
    private final int hiddenSize;
    private final int intermediateSize;
    private final int numHeads;
    private final boolean preNorm;
    private final boolean causal;
    private final float layerNormEps;

    private Arena arena;

    // Attention weights
    private MemorySegment queryWeight;
    private MemorySegment keyWeight;
    private MemorySegment valueWeight;
    private MemorySegment outputProjWeight;

    // MLP weights
    private MemorySegment mlpUpWeight;
    private MemorySegment mlpDownWeight;

    // LayerNorm parameters
    private MemorySegment attnLnWeight;
    private MemorySegment attnLnBias;
    private MemorySegment mlpLnWeight;
    private MemorySegment mlpLnBias;

    // Buffers
    private MemorySegment inputBuffer;
    private MemorySegment outputBuffer;

    private TransformerBlockBenchmark(Builder builder) {
        this.name = builder.name;
        this.hiddenSize = builder.hiddenSize;
        this.intermediateSize = builder.intermediateSize;
        this.numHeads = builder.numHeads;
        this.preNorm = builder.preNorm;
        this.causal = builder.causal;
        this.layerNormEps = builder.layerNormEps;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public String modelId() {
        return "synthetic-transformer-block";
    }

    @Override
    public void setup(BenchmarkConfig config) throws IOException {
        arena = Arena.ofShared();
        Random random = new Random(42);

        int batchSize = config.batchSize();
        int seqLen = config.sequenceLength();

        // Allocate attention weights [hidden, hidden]
        long attnProjSize = (long) hiddenSize * hiddenSize * 4;
        queryWeight = allocateAndInit(attnProjSize, random);
        keyWeight = allocateAndInit(attnProjSize, random);
        valueWeight = allocateAndInit(attnProjSize, random);
        outputProjWeight = allocateAndInit(attnProjSize, random);

        // Allocate MLP weights
        // Up projection: [hidden, intermediate]
        long upSize = (long) hiddenSize * intermediateSize * 4;
        mlpUpWeight = allocateAndInit(upSize, random);
        // Down projection: [intermediate, hidden]
        long downSize = (long) intermediateSize * hiddenSize * 4;
        mlpDownWeight = allocateAndInit(downSize, random);

        // LayerNorm parameters [hidden]
        long lnSize = (long) hiddenSize * 4;
        attnLnWeight = allocateLayerNormWeight(lnSize);
        attnLnBias = allocateAndZero(lnSize);
        mlpLnWeight = allocateLayerNormWeight(lnSize);
        mlpLnBias = allocateAndZero(lnSize);

        // Input/output buffers [batch, seq, hidden]
        long bufferSize = (long) batchSize * seqLen * hiddenSize * 4;
        inputBuffer = arena.allocate(bufferSize);
        outputBuffer = arena.allocate(bufferSize);

        // Initialize input with random values
        initializeBuffer(inputBuffer, random, 0.02f);
    }

    private MemorySegment allocateAndInit(long size, Random random) {
        MemorySegment segment = arena.allocate(size);
        float scale = (float) Math.sqrt(2.0 / hiddenSize);
        long count = size / 4;
        for (long i = 0; i < count; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) random.nextGaussian() * scale);
        }
        return segment;
    }

    private MemorySegment allocateAndZero(long size) {
        MemorySegment segment = arena.allocate(size);
        segment.fill((byte) 0);
        return segment;
    }

    private MemorySegment allocateLayerNormWeight(long size) {
        MemorySegment segment = arena.allocate(size);
        long count = size / 4;
        for (long i = 0; i < count; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, 1.0f);
        }
        return segment;
    }

    private void initializeBuffer(MemorySegment buffer, Random random, float scale) {
        long count = buffer.byteSize() / 4;
        for (long i = 0; i < count; i++) {
            buffer.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) random.nextGaussian() * scale);
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

        try (Arena scratchArena = Arena.ofConfined()) {
            long hiddenBufferSize = batchSize * seqLen * hiddenSize * 4;
            long intermediateBufferSize = batchSize * seqLen * intermediateSize * 4;

            // Scratch buffers for attention
            MemorySegment residual = scratchArena.allocate(hiddenBufferSize);
            MemorySegment normed = scratchArena.allocate(hiddenBufferSize);
            MemorySegment Q = scratchArena.allocate(hiddenBufferSize);
            MemorySegment K = scratchArena.allocate(hiddenBufferSize);
            MemorySegment V = scratchArena.allocate(hiddenBufferSize);
            MemorySegment attnOutput = scratchArena.allocate(hiddenBufferSize);

            // Scratch buffers for MLP
            MemorySegment mlpIntermediate = scratchArena.allocate(intermediateBufferSize);
            MemorySegment mlpOutput = scratchArena.allocate(hiddenBufferSize);

            // Attention scores [batch, heads, seq, seq]
            long scoresSize = batchSize * numHeads * seqLen * seqLen * 4;
            MemorySegment scores = scratchArena.allocate(scoresSize);

            // Copy input to residual
            copyBuffer(inputBuffer, residual, hiddenBufferSize);

            // === ATTENTION SUBLAYER ===

            // Step 1: Pre-norm (if pre-norm architecture)
            if (preNorm) {
                layerNorm(inputBuffer, normed, attnLnWeight, attnLnBias, batchSize, seqLen);
            } else {
                copyBuffer(inputBuffer, normed, hiddenBufferSize);
            }

            // Step 2: Q, K, V projections
            matmul(normed, queryWeight, Q, batchSize * seqLen, hiddenSize, hiddenSize);
            matmul(normed, keyWeight, K, batchSize * seqLen, hiddenSize, hiddenSize);
            matmul(normed, valueWeight, V, batchSize * seqLen, hiddenSize, hiddenSize);

            // Step 3: Attention
            computeAttention(Q, K, V, attnOutput, scores, batchSize, seqLen, headDim);

            // Step 4: Output projection
            matmul(attnOutput, outputProjWeight, attnOutput, batchSize * seqLen, hiddenSize, hiddenSize);

            // Step 5: Residual connection
            addBuffers(residual, attnOutput, residual, hiddenBufferSize);

            // Step 6: Post-norm (if post-norm architecture)
            if (!preNorm) {
                layerNorm(residual, residual, attnLnWeight, attnLnBias, batchSize, seqLen);
            }

            // === MLP SUBLAYER ===

            // Step 7: Copy for MLP residual
            copyBuffer(residual, normed, hiddenBufferSize);

            // Step 8: Pre-norm for MLP (if pre-norm)
            if (preNorm) {
                layerNorm(residual, normed, mlpLnWeight, mlpLnBias, batchSize, seqLen);
            }

            // Step 9: Up projection with GELU
            matmul(normed, mlpUpWeight, mlpIntermediate, batchSize * seqLen, hiddenSize, intermediateSize);
            applyGelu(mlpIntermediate, intermediateBufferSize);

            // Step 10: Down projection
            matmul(mlpIntermediate, mlpDownWeight, mlpOutput, batchSize * seqLen, intermediateSize, hiddenSize);

            // Step 11: Residual connection
            addBuffers(residual, mlpOutput, outputBuffer, hiddenBufferSize);

            // Step 12: Post-norm for MLP (if post-norm)
            if (!preNorm) {
                layerNorm(outputBuffer, outputBuffer, mlpLnWeight, mlpLnBias, batchSize, seqLen);
            }
        }

        // Create output tensor
        TensorInfo outputInfo = new TensorInfo("hidden_states", DType.F32,
                new long[]{batchSize, seqLen, hiddenSize}, 0, outputBuffer.byteSize());

        Map<String, TensorView> outputs = new HashMap<>();
        outputs.put("hidden_states", new TensorView(outputBuffer, outputInfo));
        return outputs;
    }

    private void copyBuffer(MemorySegment src, MemorySegment dst, long size) {
        dst.copyFrom(src.asSlice(0, size));
    }

    private void addBuffers(MemorySegment a, MemorySegment b, MemorySegment out, long size) {
        long count = size / 4;
        for (long i = 0; i < count; i++) {
            float va = a.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            float vb = b.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            out.setAtIndex(ValueLayout.JAVA_FLOAT, i, va + vb);
        }
    }

    private void layerNorm(MemorySegment input, MemorySegment output,
                           MemorySegment weight, MemorySegment bias,
                           long batchSize, long seqLen) {
        for (long b = 0; b < batchSize; b++) {
            for (long s = 0; s < seqLen; s++) {
                long baseIdx = (b * seqLen + s) * hiddenSize;

                // Compute mean
                double sum = 0;
                for (int h = 0; h < hiddenSize; h++) {
                    sum += input.getAtIndex(ValueLayout.JAVA_FLOAT, baseIdx + h);
                }
                float mean = (float) (sum / hiddenSize);

                // Compute variance
                double varSum = 0;
                for (int h = 0; h < hiddenSize; h++) {
                    float val = input.getAtIndex(ValueLayout.JAVA_FLOAT, baseIdx + h);
                    float diff = val - mean;
                    varSum += diff * diff;
                }
                float variance = (float) (varSum / hiddenSize);
                float invStd = 1.0f / (float) Math.sqrt(variance + layerNormEps);

                // Normalize and apply weight/bias
                for (int h = 0; h < hiddenSize; h++) {
                    float val = input.getAtIndex(ValueLayout.JAVA_FLOAT, baseIdx + h);
                    float normalized = (val - mean) * invStd;
                    float w = weight.getAtIndex(ValueLayout.JAVA_FLOAT, h);
                    float bi = bias.getAtIndex(ValueLayout.JAVA_FLOAT, h);
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, baseIdx + h, normalized * w + bi);
                }
            }
        }
    }

    private void matmul(MemorySegment input, MemorySegment weight, MemorySegment output,
                        long rows, int inDim, int outDim) {
        // output[i, j] = sum_k(input[i, k] * weight[k, j])
        for (long i = 0; i < rows; i++) {
            for (int j = 0; j < outDim; j++) {
                float sum = 0;
                for (int k = 0; k < inDim; k++) {
                    float inVal = input.getAtIndex(ValueLayout.JAVA_FLOAT, i * inDim + k);
                    float wVal = weight.getAtIndex(ValueLayout.JAVA_FLOAT, (long) k * outDim + j);
                    sum += inVal * wVal;
                }
                output.setAtIndex(ValueLayout.JAVA_FLOAT, i * outDim + j, sum);
            }
        }
    }

    private void computeAttention(MemorySegment Q, MemorySegment K, MemorySegment V,
                                   MemorySegment output, MemorySegment scores,
                                   long batchSize, long seqLen, int headDim) {
        float scale = 1.0f / (float) Math.sqrt(headDim);

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                // Compute attention scores
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

                        if (causal && j > i) {
                            score = Float.NEGATIVE_INFINITY;
                        }

                        long scoreIdx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        scores.setAtIndex(ValueLayout.JAVA_FLOAT, scoreIdx, score);
                    }
                }

                // Softmax
                for (int i = 0; i < seqLen; i++) {
                    int validLen = causal ? (int) (i + 1) : (int) seqLen;

                    float maxVal = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < validLen; j++) {
                        long idx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        maxVal = Math.max(maxVal, scores.getAtIndex(ValueLayout.JAVA_FLOAT, idx));
                    }

                    float sumExp = 0;
                    for (int j = 0; j < validLen; j++) {
                        long idx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        float val = (float) Math.exp(scores.getAtIndex(ValueLayout.JAVA_FLOAT, idx) - maxVal);
                        scores.setAtIndex(ValueLayout.JAVA_FLOAT, idx, val);
                        sumExp += val;
                    }

                    for (int j = 0; j < validLen; j++) {
                        long idx = ((b * numHeads + h) * seqLen + i) * seqLen + j;
                        float val = scores.getAtIndex(ValueLayout.JAVA_FLOAT, idx) / sumExp;
                        scores.setAtIndex(ValueLayout.JAVA_FLOAT, idx, val);
                    }
                }

                // Attention output
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

    private void applyGelu(MemorySegment buffer, long size) {
        // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        long count = size / 4;
        double sqrt2OverPi = Math.sqrt(2.0 / Math.PI);

        for (long i = 0; i < count; i++) {
            float x = buffer.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            double inner = sqrt2OverPi * (x + 0.044715 * x * x * x);
            float gelu = (float) (0.5 * x * (1.0 + Math.tanh(inner)));
            buffer.setAtIndex(ValueLayout.JAVA_FLOAT, i, gelu);
        }
    }

    @Override
    public List<String> outputsToValidate() {
        return List.of("hidden_states");
    }

    @Override
    public void teardown() {
        if (arena != null) {
            arena.close();
            arena = null;
        }
    }

    /**
     * Create a builder for TransformerBlockBenchmark.
     */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Builder for TransformerBlockBenchmark.
     */
    public static final class Builder {
        private final String name;
        private int hiddenSize = 768;
        private int intermediateSize = 3072;
        private int numHeads = 12;
        private boolean preNorm = true;
        private boolean causal = true;
        private float layerNormEps = 1e-5f;

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
         * Set the intermediate size (MLP hidden dimension).
         * Default: 3072 (4x hidden size, typical for transformers)
         */
        public Builder intermediateSize(int intermediateSize) {
            this.intermediateSize = intermediateSize;
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
         * Whether to use pre-norm architecture (LN before attention/MLP).
         * Default: true (GPT-2 style)
         *
         * <p>Pre-norm: LN -> Attention -> Add -> LN -> MLP -> Add
         * <p>Post-norm: Attention -> Add -> LN -> MLP -> Add -> LN
         */
        public Builder preNorm(boolean preNorm) {
            this.preNorm = preNorm;
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
         * LayerNorm epsilon for numerical stability.
         * Default: 1e-5
         */
        public Builder layerNormEps(float eps) {
            this.layerNormEps = eps;
            return this;
        }

        public TransformerBlockBenchmark build() {
            if (hiddenSize % numHeads != 0) {
                throw new IllegalArgumentException(
                        "hiddenSize (" + hiddenSize + ") must be divisible by numHeads (" + numHeads + ")");
            }
            return new TransformerBlockBenchmark(this);
        }
    }
}
