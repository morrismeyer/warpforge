package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.dataset.Dataset;
import io.surfworks.warpforge.data.dataset.GLUEDataset;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * End-to-end benchmark combining GLUE dataset with text classification/regression.
 *
 * <p>Supports all GLUE tasks:
 * <ul>
 *   <li>CoLA - Linguistic acceptability</li>
 *   <li>SST-2 - Sentiment analysis</li>
 *   <li>MRPC, QQP - Paraphrase detection</li>
 *   <li>STS-B - Semantic similarity (regression)</li>
 *   <li>MNLI, QNLI, RTE, WNLI - Natural language inference</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * GLUEBenchmark benchmark = GLUEBenchmark.builder()
 *     .datasetPath(Path.of("/data/glue"))
 *     .task(GLUEDataset.Task.SST2)
 *     .build();
 *
 * BenchmarkRunner runner = new BenchmarkRunner();
 * BenchmarkResult result = runner.run(benchmark, config);
 * }</pre>
 */
public final class GLUEBenchmark extends EndToEndBenchmark<GLUEDataset.TextSample> {

    private final GLUEDataset.Task task;
    private final int maxLength;
    private final int hiddenSize;
    private final Dataset.Split split;

    private GLUEBenchmark(Builder builder) {
        super(builder.name, builder.modelId, builder.datasetPath);
        this.task = builder.task;
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
    protected Dataset<GLUEDataset.TextSample> loadDataset(BenchmarkConfig config) throws IOException {
        return GLUEDataset.load(task, datasetPath, split);
    }

    @Override
    protected Map<String, TensorView> transformBatchForModel(
            Dataset.Batch<GLUEDataset.TextSample> batch, BenchmarkConfig config) {

        Map<String, TensorView> inputs = new HashMap<>();
        int batchSize = batch.size();

        List<int[]> inputIdsList = new ArrayList<>();
        List<int[]> attentionMaskList = new ArrayList<>();
        List<int[]> tokenTypeIdsList = new ArrayList<>();

        for (GLUEDataset.TextSample sample : batch.samples()) {
            String text;
            if (task.isSentencePair() && sample.sentence2() != null) {
                text = sample.sentence1() + " [SEP] " + sample.sentence2();
            } else {
                text = sample.sentence1();
            }

            int[] inputIds = mockTokenize(text, maxLength);
            int[] attentionMask = new int[maxLength];
            int[] tokenTypeIds = new int[maxLength];

            // Set attention mask
            int sent1Len = task.isSentencePair() ?
                    Math.min(sample.sentence1().length(), maxLength / 2) :
                    Math.min(sample.sentence1().length(), maxLength - 2);

            for (int i = 0; i < inputIds.length && inputIds[i] != 0; i++) {
                attentionMask[i] = 1;
                // Token type for sentence pair tasks
                if (task.isSentencePair()) {
                    tokenTypeIds[i] = i < sent1Len + 2 ? 0 : 1;
                }
            }

            inputIdsList.add(inputIds);
            attentionMaskList.add(attentionMask);
            tokenTypeIdsList.add(tokenTypeIds);
        }

        inputs.put("input_ids", createBatchTensor(inputIdsList, "input_ids"));
        inputs.put("attention_mask", createBatchTensor(attentionMaskList, "attention_mask"));
        if (task.isSentencePair()) {
            inputs.put("token_type_ids", createBatchTensor(tokenTypeIdsList, "token_type_ids"));
        }

        // Labels
        if (task.isRegression()) {
            // Regression labels (float)
            long[] labelShape = {batchSize};
            long labelBytes = batchSize * 4L;
            MemorySegment labels = arena.allocate(labelBytes);
            int idx = 0;
            for (GLUEDataset.TextSample sample : batch.samples()) {
                labels.setAtIndex(ValueLayout.JAVA_FLOAT, idx++, sample.score());
            }
            inputs.put("labels", new TensorView(labels,
                    new TensorInfo("labels", DType.F32, labelShape, 0, labelBytes)));
        } else {
            // Classification labels (long)
            long[] labelShape = {batchSize};
            long labelBytes = batchSize * 8L;
            MemorySegment labels = arena.allocate(labelBytes);
            int idx = 0;
            for (GLUEDataset.TextSample sample : batch.samples()) {
                labels.setAtIndex(ValueLayout.JAVA_LONG, idx++, sample.label());
            }
            inputs.put("labels", new TensorView(labels,
                    new TensorInfo("labels", DType.I64, labelShape, 0, labelBytes)));
        }

        return inputs;
    }

    @Override
    public Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException {
        TensorView inputIds = inputs.get("input_ids");
        long[] inputShape = inputIds.info().shape();
        int batchSize = (int) inputShape[0];
        int seqLen = (int) inputShape[1];

        Map<String, TensorView> outputs = new HashMap<>();

        if (task.isRegression()) {
            // Regression output [batch]
            long[] logitsShape = {batchSize};
            long logitsBytes = batchSize * 4L;
            MemorySegment logits = arena.allocate(logitsBytes);

            for (int b = 0; b < batchSize; b++) {
                // STS-B scores are 0-5
                logits.setAtIndex(ValueLayout.JAVA_FLOAT, b, (float) (Math.random() * 5));
            }

            outputs.put("logits", new TensorView(logits,
                    new TensorInfo("logits", DType.F32, logitsShape, 0, logitsBytes)));
        } else {
            // Classification output [batch, numLabels]
            int numLabels = task.numLabels();
            long[] logitsShape = {batchSize, numLabels};
            long logitsBytes = batchSize * numLabels * 4L;
            MemorySegment logits = arena.allocate(logitsBytes);

            for (int b = 0; b < batchSize; b++) {
                for (int l = 0; l < numLabels; l++) {
                    logits.setAtIndex(ValueLayout.JAVA_FLOAT, b * numLabels + l,
                            (float) (Math.random() * 2 - 1));
                }
            }

            outputs.put("logits", new TensorView(logits,
                    new TensorInfo("logits", DType.F32, logitsShape, 0, logitsBytes)));
        }

        // Hidden states [batch, seqLen, hiddenSize]
        long[] hiddenShape = {batchSize, seqLen, hiddenSize};
        long hiddenBytes = batchSize * seqLen * (long) hiddenSize * 4;
        MemorySegment hiddenStates = arena.allocate(hiddenBytes);

        for (long i = 0; i < batchSize * seqLen * hiddenSize; i++) {
            hiddenStates.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.random());
        }

        outputs.put("last_hidden_state", new TensorView(hiddenStates,
                new TensorInfo("last_hidden_state", DType.F32, hiddenShape, 0, hiddenBytes)));

        return outputs;
    }

    @Override
    public List<String> outputsToValidate() {
        return List.of("logits");
    }

    @Override
    protected long maxSamples() {
        return 2000;
    }

    private int[] mockTokenize(String text, int maxLen) {
        int[] tokens = new int[maxLen];
        byte[] bytes = text.getBytes();
        tokens[0] = 101; // [CLS]
        for (int i = 0; i < Math.min(bytes.length, maxLen - 2); i++) {
            tokens[i + 1] = (bytes[i] & 0xFF) + 1000;
        }
        int endIdx = Math.min(bytes.length + 1, maxLen - 1);
        tokens[endIdx] = 102; // [SEP]
        return tokens;
    }

    private TensorView createBatchTensor(List<int[]> sequences, String name) {
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

        return new TensorView(segment, new TensorInfo(name, DType.I32, shape, 0, byteSize));
    }

    /**
     * Get the GLUE task for this benchmark.
     */
    public GLUEDataset.Task task() {
        return task;
    }

    /**
     * Builder for GLUEBenchmark.
     */
    public static final class Builder {
        private String name;
        private String modelId = "bert-base-uncased";
        private Path datasetPath;
        private GLUEDataset.Task task = GLUEDataset.Task.SST2;
        private int maxLength = 128;
        private int hiddenSize = 768;
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

        public Builder task(GLUEDataset.Task task) {
            this.task = task;
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
                    .maxLength(128);
        }

        /**
         * Configure for RoBERTa-Base.
         */
        public Builder robertaBase() {
            return modelId("roberta-base")
                    .hiddenSize(768)
                    .maxLength(128);
        }

        public GLUEBenchmark build() {
            if (datasetPath == null) {
                throw new IllegalStateException("datasetPath must be set");
            }
            if (name == null) {
                name = "glue-" + task.name().toLowerCase() + "-endtoend";
            }
            return new GLUEBenchmark(this);
        }
    }
}
