package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.dataset.Dataset;
import io.surfworks.warpforge.data.dataset.SQuADDataset;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * End-to-end benchmark combining SQuAD dataset with BERT question answering.
 *
 * <p>Measures the full pipeline:
 * <ul>
 *   <li>SQuAD data loading (context + question)</li>
 *   <li>Tokenization and encoding</li>
 *   <li>BERT inference</li>
 *   <li>Answer span prediction validation</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * SQuADBertBenchmark benchmark = SQuADBertBenchmark.builder()
 *     .datasetPath(Path.of("/data/squad"))
 *     .version(SQuADDataset.Version.V2)
 *     .maxLength(384)
 *     .build();
 *
 * BenchmarkRunner runner = new BenchmarkRunner();
 * BenchmarkResult result = runner.run(benchmark, config);
 * }</pre>
 */
public final class SQuADBertBenchmark extends EndToEndBenchmark<SQuADDataset.QASample> {

    private final SQuADDataset.Version version;
    private final int maxLength;
    private final int hiddenSize;
    private final Dataset.Split split;

    private SQuADBertBenchmark(Builder builder) {
        super(builder.name, builder.modelId, builder.datasetPath);
        this.version = builder.version;
        this.maxLength = builder.maxLength;
        this.hiddenSize = builder.hiddenSize;
        this.split = builder.split;
    }

    /**
     * Create a new builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    @Override
    protected Dataset<SQuADDataset.QASample> loadDataset(BenchmarkConfig config) throws IOException {
        return SQuADDataset.load(version, datasetPath, split);
    }

    @Override
    protected Map<String, TensorView> transformBatchForModel(
            Dataset.Batch<SQuADDataset.QASample> batch, BenchmarkConfig config) {

        Map<String, TensorView> inputs = new HashMap<>();
        int batchSize = batch.size();

        // Collect tokenized sequences
        List<int[]> inputIdsList = new ArrayList<>();
        List<int[]> attentionMaskList = new ArrayList<>();
        List<int[]> tokenTypeIdsList = new ArrayList<>();
        List<int[]> startPositions = new ArrayList<>();
        List<int[]> endPositions = new ArrayList<>();

        for (SQuADDataset.QASample sample : batch.samples()) {
            // Simple mock tokenization (in real impl, use actual tokenizer)
            String combined = sample.question() + " [SEP] " + sample.context();
            int[] inputIds = mockTokenize(combined, maxLength);
            int[] attentionMask = new int[maxLength];
            int[] tokenTypeIds = new int[maxLength];

            // Set attention mask
            int questionLen = Math.min(sample.question().length(), maxLength / 2);
            for (int i = 0; i < inputIds.length && inputIds[i] != 0; i++) {
                attentionMask[i] = 1;
                // Token type: 0 for question, 1 for context
                tokenTypeIds[i] = i < questionLen + 2 ? 0 : 1;
            }

            inputIdsList.add(inputIds);
            attentionMaskList.add(attentionMask);
            tokenTypeIdsList.add(tokenTypeIds);

            // Answer positions
            int start = sample.isImpossible() ? 0 : Math.max(0, sample.answerStart());
            int end = sample.isImpossible() ? 0 : start + sample.answerText().length();
            startPositions.add(new int[]{Math.min(start, maxLength - 1)});
            endPositions.add(new int[]{Math.min(end, maxLength - 1)});
        }

        // Create tensors
        inputs.put("input_ids", createBatchTensor(inputIdsList, "input_ids", DType.I32));
        inputs.put("attention_mask", createBatchTensor(attentionMaskList, "attention_mask", DType.I32));
        inputs.put("token_type_ids", createBatchTensor(tokenTypeIdsList, "token_type_ids", DType.I32));

        // Start/end positions for training
        inputs.put("start_positions", createLabelTensor(startPositions, "start_positions"));
        inputs.put("end_positions", createLabelTensor(endPositions, "end_positions"));

        return inputs;
    }

    @Override
    public Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException {
        TensorView inputIds = inputs.get("input_ids");
        long[] inputShape = inputIds.info().shape();
        int batchSize = (int) inputShape[0];
        int seqLen = (int) inputShape[1];

        Map<String, TensorView> outputs = new HashMap<>();

        // Start logits [batch, seqLen]
        long[] logitsShape = {batchSize, seqLen};
        long logitsBytes = batchSize * seqLen * 4L;

        MemorySegment startLogits = arena.allocate(logitsBytes);
        MemorySegment endLogits = arena.allocate(logitsBytes);

        // Simulate QA output
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                float startLogit = (float) (Math.random() * 2 - 1);
                float endLogit = (float) (Math.random() * 2 - 1);
                startLogits.setAtIndex(ValueLayout.JAVA_FLOAT, b * seqLen + s, startLogit);
                endLogits.setAtIndex(ValueLayout.JAVA_FLOAT, b * seqLen + s, endLogit);
            }
        }

        TensorInfo startInfo = new TensorInfo("start_logits", DType.F32, logitsShape, 0, logitsBytes);
        TensorInfo endInfo = new TensorInfo("end_logits", DType.F32, logitsShape, 0, logitsBytes);
        outputs.put("start_logits", new TensorView(startLogits, startInfo));
        outputs.put("end_logits", new TensorView(endLogits, endInfo));

        // Hidden states [batch, seqLen, hiddenSize]
        long[] hiddenShape = {batchSize, seqLen, hiddenSize};
        long hiddenBytes = batchSize * seqLen * (long) hiddenSize * 4;
        MemorySegment hiddenStates = arena.allocate(hiddenBytes);

        for (long i = 0; i < batchSize * seqLen * hiddenSize; i++) {
            hiddenStates.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.random());
        }

        TensorInfo hiddenInfo = new TensorInfo("last_hidden_state", DType.F32, hiddenShape, 0, hiddenBytes);
        outputs.put("last_hidden_state", new TensorView(hiddenStates, hiddenInfo));

        return outputs;
    }

    @Override
    public List<String> outputsToValidate() {
        return List.of("start_logits", "end_logits");
    }

    @Override
    protected long maxSamples() {
        return 1000;
    }

    private int[] mockTokenize(String text, int maxLen) {
        int[] tokens = new int[maxLen];
        byte[] bytes = text.getBytes();
        // Simple byte-level tokenization (mock)
        tokens[0] = 101; // [CLS]
        for (int i = 0; i < Math.min(bytes.length, maxLen - 2); i++) {
            tokens[i + 1] = (bytes[i] & 0xFF) + 1000; // Offset to avoid special tokens
        }
        int endIdx = Math.min(bytes.length + 1, maxLen - 1);
        tokens[endIdx] = 102; // [SEP]
        return tokens;
    }

    private TensorView createBatchTensor(List<int[]> sequences, String name, DType dtype) {
        int batchSize = sequences.size();
        int seqLen = sequences.get(0).length;
        long[] shape = {batchSize, seqLen};
        long byteSize = batchSize * seqLen * 4L;

        MemorySegment segment = arena.allocate(byteSize);
        for (int b = 0; b < batchSize; b++) {
            int[] seq = sequences.get(b);
            for (int s = 0; s < seqLen; s++) {
                segment.setAtIndex(ValueLayout.JAVA_INT, b * seqLen + s, seq[s]);
            }
        }

        TensorInfo info = new TensorInfo(name, dtype, shape, 0, byteSize);
        return new TensorView(segment, info);
    }

    private TensorView createLabelTensor(List<int[]> labels, String name) {
        int batchSize = labels.size();
        long[] shape = {batchSize};
        long byteSize = batchSize * 8L;

        MemorySegment segment = arena.allocate(byteSize);
        for (int b = 0; b < batchSize; b++) {
            segment.setAtIndex(ValueLayout.JAVA_LONG, b, labels.get(b)[0]);
        }

        TensorInfo info = new TensorInfo(name, DType.I64, shape, 0, byteSize);
        return new TensorView(segment, info);
    }

    /**
     * Builder for SQuADBertBenchmark.
     */
    public static final class Builder {
        private String name = "squad-bert-endtoend";
        private String modelId = "bert-large-uncased-whole-word-masking-finetuned-squad";
        private Path datasetPath;
        private SQuADDataset.Version version = SQuADDataset.Version.V2;
        private int maxLength = 384;
        private int hiddenSize = 1024;
        private Dataset.Split split = Dataset.Split.VALIDATION;

        private Builder() {}

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder datasetPath(Path datasetPath) {
            this.datasetPath = datasetPath;
            return this;
        }

        public Builder version(SQuADDataset.Version version) {
            this.version = version;
            return this;
        }

        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }

        public Builder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        public Builder split(Dataset.Split split) {
            this.split = split;
            return this;
        }

        /**
         * Configure for BERT-Base.
         */
        public Builder bertBase() {
            return modelId("bert-base-uncased")
                    .hiddenSize(768)
                    .maxLength(384);
        }

        /**
         * Configure for BERT-Large.
         */
        public Builder bertLarge() {
            return modelId("bert-large-uncased-whole-word-masking-finetuned-squad")
                    .hiddenSize(1024)
                    .maxLength(384);
        }

        public SQuADBertBenchmark build() {
            if (datasetPath == null) {
                throw new IllegalStateException("datasetPath must be set");
            }
            return new SQuADBertBenchmark(this);
        }
    }
}
