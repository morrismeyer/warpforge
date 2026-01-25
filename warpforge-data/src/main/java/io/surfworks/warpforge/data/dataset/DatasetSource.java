package io.surfworks.warpforge.data.dataset;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Interface for accessing datasets.
 *
 * <p>Supports common formats like JSON (SQuAD), Parquet, and Arrow.
 */
public interface DatasetSource extends AutoCloseable, Iterable<Map<String, Object>> {

    /**
     * Dataset identifier.
     */
    String id();

    /**
     * Number of examples in the dataset.
     */
    long size();

    /**
     * Get a single example by index.
     */
    Map<String, Object> get(long index);

    /**
     * Get a batch of examples.
     */
    List<Map<String, Object>> getBatch(long startIndex, int batchSize);

    /**
     * Stream all examples.
     */
    Stream<Map<String, Object>> stream();

    /**
     * Get split names (e.g., "train", "validation", "test").
     */
    List<String> splits();

    /**
     * Open a specific split.
     */
    DatasetSource split(String splitName) throws IOException;

    @Override
    void close();

    /**
     * Open a dataset from a local directory.
     */
    static DatasetSource open(String id, Path dir) throws IOException {
        // Try to detect format

        // SQuAD format
        if (Files.exists(dir.resolve("train-v2.0.json")) ||
                Files.exists(dir.resolve("dev-v2.0.json"))) {
            return new SquadDataset(id, dir);
        }

        // COCO format (has annotations/ subdirectory)
        Path annotationsDir = dir.resolve("annotations");
        if (Files.isDirectory(annotationsDir)) {
            try (Stream<Path> files = Files.list(annotationsDir)) {
                boolean hasCoco = files.anyMatch(p -> {
                    String name = p.getFileName().toString();
                    return name.startsWith("instances_") ||
                            name.startsWith("captions_") ||
                            name.startsWith("person_keypoints_");
                });
                if (hasCoco) {
                    return CocoDataset.load(id, dir);
                }
            }
        }

        // Parquet files
        try (Stream<Path> files = Files.list(dir)) {
            List<Path> parquetFiles = files
                    .filter(p -> p.toString().endsWith(".parquet"))
                    .toList();
            if (!parquetFiles.isEmpty()) {
                return ParquetDataset.load(id, parquetFiles.get(0));
            }
        }

        // Look for any JSON files
        try (Stream<Path> files = Files.list(dir)) {
            List<Path> jsonFiles = files
                    .filter(p -> p.toString().endsWith(".json"))
                    .toList();
            if (!jsonFiles.isEmpty()) {
                return new JsonDataset(id, jsonFiles.get(0));
            }
        }

        throw new IOException("Unknown dataset format in: " + dir);
    }

    /**
     * Open a single Parquet file as a dataset.
     */
    static DatasetSource openParquet(String id, Path file) throws IOException {
        return ParquetDataset.load(id, file);
    }

    /**
     * Open a COCO dataset from a directory.
     *
     * @param id Dataset identifier
     * @param dir Directory containing annotations/ and images/ subdirectories
     * @param split Split name (e.g., "val2017", "train2017")
     * @param annotationType Type of annotations to load
     */
    static DatasetSource openCoco(String id, Path dir, String split,
                                   CocoDataset.CocoAnnotationType annotationType) throws IOException {
        return CocoDataset.load(id, dir, split, annotationType);
    }

    /**
     * Simple JSON dataset implementation.
     */
    class JsonDataset implements DatasetSource {
        private static final Gson GSON = new Gson();

        private final String id;
        private final List<Map<String, Object>> data;

        @SuppressWarnings("unchecked")
        public JsonDataset(String id, Path jsonFile) throws IOException {
            this.id = id;
            String content = Files.readString(jsonFile);
            JsonElement root = GSON.fromJson(content, JsonElement.class);

            this.data = new ArrayList<>();
            if (root.isJsonArray()) {
                for (JsonElement elem : root.getAsJsonArray()) {
                    data.add(GSON.fromJson(elem, Map.class));
                }
            } else if (root.isJsonObject()) {
                // Single object or nested structure
                JsonObject obj = root.getAsJsonObject();
                if (obj.has("data")) {
                    for (JsonElement elem : obj.getAsJsonArray("data")) {
                        data.add(GSON.fromJson(elem, Map.class));
                    }
                } else {
                    data.add(GSON.fromJson(obj, Map.class));
                }
            }
        }

        @Override
        public String id() {
            return id;
        }

        @Override
        public long size() {
            return data.size();
        }

        @Override
        public Map<String, Object> get(long index) {
            return data.get((int) index);
        }

        @Override
        public List<Map<String, Object>> getBatch(long startIndex, int batchSize) {
            int start = (int) startIndex;
            int end = Math.min(start + batchSize, data.size());
            return data.subList(start, end);
        }

        @Override
        public Stream<Map<String, Object>> stream() {
            return data.stream();
        }

        @Override
        public List<String> splits() {
            return List.of("default");
        }

        @Override
        public DatasetSource split(String splitName) {
            return this;
        }

        @Override
        public Iterator<Map<String, Object>> iterator() {
            return data.iterator();
        }

        @Override
        public void close() {
            // No resources to close
        }
    }

    /**
     * SQuAD dataset implementation.
     */
    class SquadDataset implements DatasetSource {
        private static final Gson GSON = new Gson();

        private final String id;
        private final Path dir;
        private final List<Map<String, Object>> data;
        private final String currentSplit;

        public SquadDataset(String id, Path dir) throws IOException {
            this(id, dir, "dev"); // Default to dev split
        }

        private SquadDataset(String id, Path dir, String split) throws IOException {
            this.id = id;
            this.dir = dir;
            this.currentSplit = split;
            this.data = loadSplit(split);
        }

        @SuppressWarnings("unchecked")
        private List<Map<String, Object>> loadSplit(String split) throws IOException {
            Path file = dir.resolve(split + "-v2.0.json");
            if (!Files.exists(file)) {
                file = dir.resolve(split + "-v1.1.json");
            }
            if (!Files.exists(file)) {
                throw new IOException("Split not found: " + split);
            }

            String content = Files.readString(file);
            JsonObject root = GSON.fromJson(content, JsonObject.class);

            List<Map<String, Object>> examples = new ArrayList<>();
            JsonArray articles = root.getAsJsonArray("data");

            for (JsonElement article : articles) {
                JsonObject articleObj = article.getAsJsonObject();
                String title = articleObj.has("title") ? articleObj.get("title").getAsString() : "";

                for (JsonElement para : articleObj.getAsJsonArray("paragraphs")) {
                    JsonObject paraObj = para.getAsJsonObject();
                    String context = paraObj.get("context").getAsString();

                    for (JsonElement qa : paraObj.getAsJsonArray("qas")) {
                        JsonObject qaObj = qa.getAsJsonObject();
                        String question = qaObj.get("question").getAsString();
                        String qid = qaObj.get("id").getAsString();

                        List<String> answers = new ArrayList<>();
                        if (qaObj.has("answers")) {
                            for (JsonElement ans : qaObj.getAsJsonArray("answers")) {
                                answers.add(ans.getAsJsonObject().get("text").getAsString());
                            }
                        }

                        boolean isImpossible = qaObj.has("is_impossible") &&
                                qaObj.get("is_impossible").getAsBoolean();

                        examples.add(Map.of(
                                "id", qid,
                                "title", title,
                                "context", context,
                                "question", question,
                                "answers", answers,
                                "is_impossible", isImpossible
                        ));
                    }
                }
            }

            return examples;
        }

        @Override
        public String id() {
            return id;
        }

        @Override
        public long size() {
            return data.size();
        }

        @Override
        public Map<String, Object> get(long index) {
            return data.get((int) index);
        }

        @Override
        public List<Map<String, Object>> getBatch(long startIndex, int batchSize) {
            int start = (int) startIndex;
            int end = Math.min(start + batchSize, data.size());
            return data.subList(start, end);
        }

        @Override
        public Stream<Map<String, Object>> stream() {
            return data.stream();
        }

        @Override
        public List<String> splits() {
            List<String> available = new ArrayList<>();
            if (Files.exists(dir.resolve("train-v2.0.json")) ||
                    Files.exists(dir.resolve("train-v1.1.json"))) {
                available.add("train");
            }
            if (Files.exists(dir.resolve("dev-v2.0.json")) ||
                    Files.exists(dir.resolve("dev-v1.1.json"))) {
                available.add("dev");
            }
            return available;
        }

        @Override
        public DatasetSource split(String splitName) throws IOException {
            return new SquadDataset(id, dir, splitName);
        }

        @Override
        public Iterator<Map<String, Object>> iterator() {
            return data.iterator();
        }

        @Override
        public void close() {
            // No resources to close
        }
    }
}
