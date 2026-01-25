package io.surfworks.warpforge.data.dataset;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

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

/**
 * SQuAD (Stanford Question Answering Dataset) loader.
 *
 * <p>Supports both SQuAD v1.1 and SQuAD v2.0 formats.
 *
 * <p>Example usage:
 * <pre>{@code
 * SQuADDataset dataset = SQuADDataset.load(Version.V2, Path.of("/data/squad"))
 *     .split(Split.TRAIN);
 *
 * for (SQuADDataset.QASample sample : dataset) {
 *     String context = sample.context();
 *     String question = sample.question();
 *     List<Answer> answers = sample.answers();
 * }
 * }</pre>
 */
public class SQuADDataset extends AbstractDataset<SQuADDataset.QASample> {

    private static final String SQUAD_V1_TRAIN = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json";
    private static final String SQUAD_V1_DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json";
    private static final String SQUAD_V2_TRAIN = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json";
    private static final String SQUAD_V2_DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json";

    private final Version version;
    private final Path dataPath;
    private final List<QAEntry> entries;
    private final Arena arena;

    private SQuADDataset(Version version, Path dataPath, Split split, List<QAEntry> entries) {
        super("squad-" + version.name().toLowerCase(), split);
        this.version = version;
        this.dataPath = dataPath;
        this.entries = entries;
        this.arena = Arena.ofShared();
    }

    /**
     * SQuAD version.
     */
    public enum Version {
        V1("1.1"),
        V2("2.0");

        private final String versionStr;

        Version(String versionStr) {
            this.versionStr = versionStr;
        }

        public String versionString() {
            return versionStr;
        }
    }

    /**
     * Load SQuAD dataset.
     */
    public static SQuADDataset load(Version version, Path dataPath) throws IOException {
        return load(version, dataPath, Split.TRAIN);
    }

    /**
     * Load SQuAD dataset with specific split.
     */
    public static SQuADDataset load(Version version, Path dataPath, Split split) throws IOException {
        Files.createDirectories(dataPath);

        String filename = getFilename(version, split);
        Path dataFile = dataPath.resolve(filename);

        // Download if not present
        if (!Files.exists(dataFile)) {
            download(version, split, dataFile);
        }

        // Parse JSON
        List<QAEntry> entries = parseFile(dataFile);

        return new SQuADDataset(version, dataPath, split, entries);
    }

    /**
     * Download SQuAD data.
     */
    public static void download(Version version, Split split, Path destination) throws IOException {
        String url = switch (version) {
            case V1 -> split == Split.TRAIN ? SQUAD_V1_TRAIN : SQUAD_V1_DEV;
            case V2 -> split == Split.TRAIN ? SQUAD_V2_TRAIN : SQUAD_V2_DEV;
        };

        Files.createDirectories(destination.getParent());

        HttpURLConnection conn = (HttpURLConnection) URI.create(url).toURL().openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(30000);
        conn.setReadTimeout(120000);

        try (var in = conn.getInputStream()) {
            Files.copy(in, destination);
        } finally {
            conn.disconnect();
        }
    }

    private static String getFilename(Version version, Split split) {
        String splitStr = split == Split.TRAIN ? "train" : "dev";
        return splitStr + "-v" + version.versionString() + ".json";
    }

    private static List<QAEntry> parseFile(Path file) throws IOException {
        List<QAEntry> entries = new ArrayList<>();

        try (var reader = Files.newBufferedReader(file)) {
            JsonObject root = JsonParser.parseReader(reader).getAsJsonObject();
            JsonArray data = root.getAsJsonArray("data");

            for (JsonElement article : data) {
                JsonObject articleObj = article.getAsJsonObject();
                String title = articleObj.has("title") ? articleObj.get("title").getAsString() : "";

                JsonArray paragraphs = articleObj.getAsJsonArray("paragraphs");
                for (JsonElement paragraph : paragraphs) {
                    JsonObject paraObj = paragraph.getAsJsonObject();
                    String context = paraObj.get("context").getAsString();

                    JsonArray qas = paraObj.getAsJsonArray("qas");
                    for (JsonElement qa : qas) {
                        JsonObject qaObj = qa.getAsJsonObject();
                        String id = qaObj.get("id").getAsString();
                        String question = qaObj.get("question").getAsString();
                        boolean isImpossible = qaObj.has("is_impossible") &&
                                qaObj.get("is_impossible").getAsBoolean();

                        List<Answer> answers = new ArrayList<>();
                        JsonArray answersArray = qaObj.getAsJsonArray("answers");
                        if (answersArray != null) {
                            for (JsonElement ans : answersArray) {
                                JsonObject ansObj = ans.getAsJsonObject();
                                String text = ansObj.get("text").getAsString();
                                int start = ansObj.get("answer_start").getAsInt();
                                answers.add(new Answer(text, start));
                            }
                        }

                        // SQuAD v2 has plausible_answers for impossible questions
                        if (qaObj.has("plausible_answers")) {
                            JsonArray plausible = qaObj.getAsJsonArray("plausible_answers");
                            for (JsonElement ans : plausible) {
                                JsonObject ansObj = ans.getAsJsonObject();
                                String text = ansObj.get("text").getAsString();
                                int start = ansObj.get("answer_start").getAsInt();
                                answers.add(new Answer(text, start));
                            }
                        }

                        entries.add(new QAEntry(id, title, context, question, answers, isImpossible));
                    }
                }
            }
        }

        return entries;
    }

    @Override
    protected long totalSize() {
        return entries.size();
    }

    @Override
    protected QASample getRaw(long index) {
        QAEntry entry = entries.get((int) index);
        return new QASample(entry, arena);
    }

    @Override
    public Dataset<QASample> split(Split split) {
        try {
            return load(version, dataPath, split);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load split: " + split, e);
        }
    }

    @Override
    public DatasetInfo info() {
        return DatasetInfo.builder("squad-" + version.name().toLowerCase())
                .description("Stanford Question Answering Dataset " + version.versionString())
                .totalSamples(entries.size())
                .splitSize(currentSplit, entries.size())
                .feature("context")
                .feature("question")
                .feature("answers")
                .extra("version", version.versionString())
                .extra("has_impossible", version == Version.V2)
                .build();
    }

    /**
     * Get the SQuAD version.
     */
    public Version version() {
        return version;
    }

    /**
     * An answer span.
     */
    public record Answer(String text, int startPosition) {}

    // Internal entry class
    private record QAEntry(
            String id,
            String title,
            String context,
            String question,
            List<Answer> answers,
            boolean isImpossible
    ) {}

    /**
     * A single SQuAD question-answer sample.
     */
    public static class QASample implements Dataset.Sample {
        private final QAEntry entry;
        private final Arena arena;

        QASample(QAEntry entry, Arena arena) {
            this.entry = entry;
            this.arena = arena;
        }

        /**
         * Get the sample ID.
         */
        public String id() {
            return entry.id;
        }

        /**
         * Get the article title.
         */
        public String title() {
            return entry.title;
        }

        /**
         * Get the context paragraph.
         */
        public String context() {
            return entry.context;
        }

        /**
         * Get the question.
         */
        public String question() {
            return entry.question;
        }

        /**
         * Get the answers.
         */
        public List<Answer> answers() {
            return entry.answers;
        }

        /**
         * Whether this question is impossible to answer (SQuAD v2).
         */
        public boolean isImpossible() {
            return entry.isImpossible;
        }

        /**
         * Get the first answer text, or empty string if no answers.
         */
        public String answerText() {
            return entry.answers.isEmpty() ? "" : entry.answers.get(0).text;
        }

        /**
         * Get the first answer start position, or -1 if no answers.
         */
        public int answerStart() {
            return entry.answers.isEmpty() ? -1 : entry.answers.get(0).startPosition;
        }

        @Override
        public Map<String, TensorView> toTensors() {
            Map<String, TensorView> tensors = new HashMap<>();

            // For actual usage, text would be tokenized
            // For benchmarking, create placeholder tensors
            int maxLen = 384;
            long[] shape = {maxLen};
            long byteSize = maxLen * 4L;  // I32

            // Input IDs (mock tokenization)
            MemorySegment inputIds = arena.allocate(byteSize);
            String combined = entry.question + " [SEP] " + entry.context;
            byte[] bytes = combined.getBytes();
            for (int i = 0; i < Math.min(maxLen, bytes.length); i++) {
                inputIds.setAtIndex(ValueLayout.JAVA_INT, i, bytes[i] & 0xFF);
            }
            TensorInfo inputInfo = new TensorInfo("input_ids", DType.I32, shape, 0, byteSize);
            tensors.put("input_ids", new TensorView(inputIds, inputInfo));

            // Attention mask
            MemorySegment attentionMask = arena.allocate(byteSize);
            int seqLen = Math.min(maxLen, bytes.length);
            for (int i = 0; i < seqLen; i++) {
                attentionMask.setAtIndex(ValueLayout.JAVA_INT, i, 1);
            }
            TensorInfo maskInfo = new TensorInfo("attention_mask", DType.I32, shape, 0, byteSize);
            tensors.put("attention_mask", new TensorView(attentionMask, maskInfo));

            // Start and end positions
            MemorySegment startPos = arena.allocate(8);
            MemorySegment endPos = arena.allocate(8);
            int start = entry.isImpossible ? 0 : Math.max(0, answerStart());
            int end = entry.isImpossible ? 0 : start + answerText().length();
            startPos.set(ValueLayout.JAVA_LONG, 0, start);
            endPos.set(ValueLayout.JAVA_LONG, 0, end);

            TensorInfo startInfo = new TensorInfo("start_positions", DType.I64, new long[]{}, 0, 8);
            TensorInfo endInfo = new TensorInfo("end_positions", DType.I64, new long[]{}, 0, 8);
            tensors.put("start_positions", new TensorView(startPos, startInfo));
            tensors.put("end_positions", new TensorView(endPos, endInfo));

            return tensors;
        }
    }
}
