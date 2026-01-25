package io.surfworks.warpforge.data.dataset;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * GLUE (General Language Understanding Evaluation) benchmark dataset loader.
 *
 * <p>Supports all GLUE tasks:
 * <ul>
 *   <li>CoLA - Corpus of Linguistic Acceptability</li>
 *   <li>SST-2 - Stanford Sentiment Treebank</li>
 *   <li>MRPC - Microsoft Research Paraphrase Corpus</li>
 *   <li>QQP - Quora Question Pairs</li>
 *   <li>STS-B - Semantic Textual Similarity Benchmark</li>
 *   <li>MNLI - Multi-Genre Natural Language Inference</li>
 *   <li>QNLI - Question Natural Language Inference</li>
 *   <li>RTE - Recognizing Textual Entailment</li>
 *   <li>WNLI - Winograd Natural Language Inference</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * GLUEDataset dataset = GLUEDataset.load(Task.SST2, Path.of("/data/glue"))
 *     .split(Split.TRAIN);
 *
 * for (GLUEDataset.TextSample sample : dataset) {
 *     String sentence = sample.sentence();
 *     int label = sample.label();
 * }
 * }</pre>
 */
public class GLUEDataset extends AbstractDataset<GLUEDataset.TextSample> {

    private static final String GLUE_URL = "https://dl.fbaipublicfiles.com/glue/data/";

    private final Task task;
    private final Path dataPath;
    private final List<TextEntry> entries;
    private final Map<String, Integer> labelMapping;
    private final Arena arena;

    private GLUEDataset(Task task, Path dataPath, Split split, List<TextEntry> entries,
                        Map<String, Integer> labelMapping) {
        super("glue-" + task.name().toLowerCase(), split);
        this.task = task;
        this.dataPath = dataPath;
        this.entries = entries;
        this.labelMapping = labelMapping;
        this.arena = Arena.ofShared();
    }

    /**
     * GLUE task types.
     */
    public enum Task {
        COLA("CoLA", true, false, 2),
        SST2("SST-2", true, false, 2),
        MRPC("MRPC", false, true, 2),
        QQP("QQP", false, true, 2),
        STSB("STS-B", false, true, -1),  // Regression
        MNLI("MNLI", false, true, 3),
        QNLI("QNLI", false, true, 2),
        RTE("RTE", false, true, 2),
        WNLI("WNLI", false, true, 2);

        private final String dirName;
        private final boolean singleSentence;
        private final boolean sentencePair;
        private final int numLabels;

        Task(String dirName, boolean singleSentence, boolean sentencePair, int numLabels) {
            this.dirName = dirName;
            this.singleSentence = singleSentence;
            this.sentencePair = sentencePair;
            this.numLabels = numLabels;
        }

        public String dirName() { return dirName; }
        public boolean isSingleSentence() { return singleSentence; }
        public boolean isSentencePair() { return sentencePair; }
        public int numLabels() { return numLabels; }
        public boolean isRegression() { return numLabels < 0; }
    }

    /**
     * Load a GLUE task from the specified data directory.
     */
    public static GLUEDataset load(Task task, Path dataPath) throws IOException {
        return load(task, dataPath, Split.TRAIN);
    }

    /**
     * Load a GLUE task with specific split.
     */
    public static GLUEDataset load(Task task, Path dataPath, Split split) throws IOException {
        Path taskPath = dataPath.resolve(task.dirName());

        // Download if not present
        if (!Files.isDirectory(taskPath)) {
            download(task, dataPath);
        }

        // Load data
        String filename = getSplitFilename(task, split);
        Path dataFile = taskPath.resolve(filename);

        if (!Files.exists(dataFile)) {
            throw new IOException("Data file not found: " + dataFile);
        }

        List<TextEntry> entries = parseFile(task, dataFile);
        Map<String, Integer> labelMapping = buildLabelMapping(task);

        return new GLUEDataset(task, dataPath, split, entries, labelMapping);
    }

    /**
     * Download GLUE task data.
     */
    public static void download(Task task, Path dataPath) throws IOException {
        Files.createDirectories(dataPath);

        String url = GLUE_URL + task.dirName() + ".zip";
        Path zipPath = dataPath.resolve(task.dirName() + ".zip");

        // Download
        downloadFile(url, zipPath);

        // Extract
        extractZip(zipPath, dataPath);

        // Cleanup
        Files.deleteIfExists(zipPath);
    }

    private static void downloadFile(String url, Path destination) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) URI.create(url).toURL().openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(30000);
        conn.setReadTimeout(60000);

        try (var in = conn.getInputStream()) {
            Files.copy(in, destination);
        } finally {
            conn.disconnect();
        }
    }

    private static void extractZip(Path zipPath, Path destDir) throws IOException {
        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(zipPath))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                Path entryPath = destDir.resolve(entry.getName());
                if (entry.isDirectory()) {
                    Files.createDirectories(entryPath);
                } else {
                    Files.createDirectories(entryPath.getParent());
                    Files.copy(zis, entryPath);
                }
                zis.closeEntry();
            }
        }
    }

    private static String getSplitFilename(Task task, Split split) {
        String base = switch (split) {
            case TRAIN -> "train";
            case VALIDATION -> "dev";
            case TEST -> "test";
        };

        // MNLI has matched/mismatched variants
        if (task == Task.MNLI && split != Split.TRAIN) {
            base = base + "_matched";
        }

        return base + ".tsv";
    }

    private static List<TextEntry> parseFile(Task task, Path file) throws IOException {
        List<TextEntry> entries = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(file)) {
            String header = reader.readLine();  // Skip header
            if (header == null) return entries;

            String[] columns = header.split("\t");
            int sentence1Col = findColumn(columns, task.isSingleSentence() ? "sentence" : "sentence1",
                    task.isSingleSentence() ? "sentence" : "question");
            int sentence2Col = task.isSentencePair() ?
                    findColumn(columns, "sentence2", "sentence", "question") : -1;
            int labelCol = findColumn(columns, "label", "quality", "is_duplicate");

            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("\t", -1);
                if (parts.length <= Math.max(sentence1Col, Math.max(sentence2Col, labelCol))) {
                    continue;  // Skip malformed lines
                }

                String sentence1 = sentence1Col >= 0 && sentence1Col < parts.length ?
                        parts[sentence1Col] : "";
                String sentence2 = sentence2Col >= 0 && sentence2Col < parts.length ?
                        parts[sentence2Col] : null;
                String labelStr = labelCol >= 0 && labelCol < parts.length ?
                        parts[labelCol] : "0";

                entries.add(new TextEntry(sentence1, sentence2, labelStr));
            }
        }

        return entries;
    }

    private static int findColumn(String[] columns, String... names) {
        for (String name : names) {
            for (int i = 0; i < columns.length; i++) {
                if (columns[i].equalsIgnoreCase(name)) {
                    return i;
                }
            }
        }
        return -1;
    }

    private static Map<String, Integer> buildLabelMapping(Task task) {
        Map<String, Integer> mapping = new HashMap<>();
        switch (task) {
            case COLA, SST2, MRPC, QQP, QNLI, WNLI -> {
                mapping.put("0", 0);
                mapping.put("1", 1);
            }
            case RTE -> {
                mapping.put("not_entailment", 0);
                mapping.put("entailment", 1);
            }
            case MNLI -> {
                mapping.put("contradiction", 0);
                mapping.put("neutral", 1);
                mapping.put("entailment", 2);
            }
            case STSB -> {
                // Regression - no label mapping needed
            }
        }
        return mapping;
    }

    @Override
    protected long totalSize() {
        return entries.size();
    }

    @Override
    protected TextSample getRaw(long index) {
        TextEntry entry = entries.get((int) index);
        return new TextSample(entry, task, labelMapping, arena);
    }

    @Override
    public Dataset<TextSample> split(Split split) {
        try {
            return load(task, dataPath, split);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load split: " + split, e);
        }
    }

    @Override
    public DatasetInfo info() {
        return DatasetInfo.builder("glue-" + task.name().toLowerCase())
                .description("GLUE " + task.dirName() + " task")
                .totalSamples(entries.size())
                .splitSize(currentSplit, entries.size())
                .feature("sentence1")
                .feature(task.isSentencePair() ? "sentence2" : null)
                .feature("label")
                .extra("task", task.name())
                .extra("num_labels", task.numLabels())
                .extra("is_regression", task.isRegression())
                .build();
    }

    /**
     * Get the GLUE task.
     */
    public Task task() {
        return task;
    }

    // Internal entry class
    private record TextEntry(String sentence1, String sentence2, String label) {}

    /**
     * A single GLUE text sample.
     */
    public static class TextSample implements Dataset.Sample {
        private final TextEntry entry;
        private final Task task;
        private final Map<String, Integer> labelMapping;
        private final Arena arena;

        TextSample(TextEntry entry, Task task, Map<String, Integer> labelMapping, Arena arena) {
            this.entry = entry;
            this.task = task;
            this.labelMapping = labelMapping;
            this.arena = arena;
        }

        /**
         * Get the first sentence.
         */
        public String sentence() {
            return entry.sentence1;
        }

        /**
         * Get the first sentence (alias).
         */
        public String sentence1() {
            return entry.sentence1;
        }

        /**
         * Get the second sentence (for sentence pair tasks).
         */
        public String sentence2() {
            return entry.sentence2;
        }

        /**
         * Get the label string.
         */
        public String labelString() {
            return entry.label;
        }

        /**
         * Get the label as integer (for classification tasks).
         */
        public int label() {
            return labelMapping.getOrDefault(entry.label, 0);
        }

        /**
         * Get the label as float (for regression tasks like STS-B).
         */
        public float score() {
            try {
                return Float.parseFloat(entry.label);
            } catch (NumberFormatException e) {
                return 0.0f;
            }
        }

        @Override
        public Map<String, TensorView> toTensors() {
            Map<String, TensorView> tensors = new HashMap<>();

            // For actual usage, text would be tokenized here
            // For benchmarking, we create placeholder tensors

            // Create mock input_ids tensor
            int maxLen = 128;
            long[] shape = {maxLen};
            long byteSize = maxLen * 4L;  // I32

            MemorySegment inputIds = arena.allocate(byteSize);
            // Simple hash-based mock tokenization
            byte[] bytes = entry.sentence1.getBytes();
            for (int i = 0; i < Math.min(maxLen, bytes.length); i++) {
                inputIds.setAtIndex(ValueLayout.JAVA_INT, i, bytes[i] & 0xFF);
            }

            TensorInfo inputInfo = new TensorInfo("input_ids", DType.I32, shape, 0, byteSize);
            tensors.put("input_ids", new TensorView(inputIds, inputInfo));

            // Create attention_mask
            MemorySegment attentionMask = arena.allocate(byteSize);
            int seqLen = Math.min(maxLen, bytes.length);
            for (int i = 0; i < seqLen; i++) {
                attentionMask.setAtIndex(ValueLayout.JAVA_INT, i, 1);
            }
            TensorInfo maskInfo = new TensorInfo("attention_mask", DType.I32, shape, 0, byteSize);
            tensors.put("attention_mask", new TensorView(attentionMask, maskInfo));

            // Create label tensor
            if (task.isRegression()) {
                MemorySegment labelSegment = arena.allocate(4);
                labelSegment.set(ValueLayout.JAVA_FLOAT, 0, score());
                TensorInfo labelInfo = new TensorInfo("label", DType.F32, new long[]{}, 0, 4);
                tensors.put("label", new TensorView(labelSegment, labelInfo));
            } else {
                MemorySegment labelSegment = arena.allocate(8);
                labelSegment.set(ValueLayout.JAVA_LONG, 0, label());
                TensorInfo labelInfo = new TensorInfo("label", DType.I64, new long[]{}, 0, 8);
                tensors.put("label", new TensorView(labelSegment, labelInfo));
            }

            return tensors;
        }
    }
}
