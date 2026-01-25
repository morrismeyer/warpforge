package io.surfworks.warpforge.data.dataset;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * COCO (Common Objects in Context) dataset loader.
 */
public class COCODataset extends AbstractDataset<COCODataset.COCOSample> {

    private final Task task;
    private final Path rootPath;
    private final List<ImageEntry> entries;
    private final Map<Long, CategoryInfo> categories;
    private final int imageSize;
    private final Arena arena;

    private COCODataset(Task task, Path rootPath, Split split, List<ImageEntry> entries,
                        Map<Long, CategoryInfo> categories, int imageSize) {
        super("coco-" + task.name().toLowerCase(), split);
        this.task = task;
        this.rootPath = rootPath;
        this.entries = entries;
        this.categories = categories;
        this.imageSize = imageSize;
        this.arena = Arena.ofShared();
    }

    public enum Task {
        DETECTION("instances"),
        SEGMENTATION("instances"),
        KEYPOINTS("person_keypoints"),
        CAPTIONS("captions");

        private final String annotationPrefix;
        Task(String annotationPrefix) { this.annotationPrefix = annotationPrefix; }
        public String annotationPrefix() { return annotationPrefix; }
    }

    public static COCODataset load(Task task, Path rootPath) throws IOException {
        return load(task, rootPath, Split.TRAIN, 640);
    }

    public static COCODataset load(Task task, Path rootPath, Split split) throws IOException {
        return load(task, rootPath, split, 640);
    }

    public static COCODataset load(Task task, Path rootPath, Split split, int imageSize) throws IOException {
        String splitName = split == Split.TRAIN ? "train" : "val";
        String annotationFile = task.annotationPrefix() + "_" + splitName + "2017.json";
        Path annotationPath = rootPath.resolve("annotations").resolve(annotationFile);
        Path imagePath = rootPath.resolve(splitName + "2017");

        if (!Files.exists(annotationPath)) {
            throw new IOException("Annotation file not found: " + annotationPath);
        }

        var result = parseAnnotations(annotationPath, imagePath);
        return new COCODataset(task, rootPath, split, result.entries, result.categories, imageSize);
    }

    private record ParseResult(List<ImageEntry> entries, Map<Long, CategoryInfo> categories) {}

    private static ParseResult parseAnnotations(Path annotationPath, Path imageDir) throws IOException {
        Map<Long, CategoryInfo> categories = new HashMap<>();
        Map<Long, ImageInfo> images = new HashMap<>();
        Map<Long, List<Annotation>> annotations = new HashMap<>();

        try (var reader = Files.newBufferedReader(annotationPath)) {
            JsonObject root = JsonParser.parseReader(reader).getAsJsonObject();

            if (root.has("categories")) {
                for (JsonElement cat : root.getAsJsonArray("categories")) {
                    JsonObject c = cat.getAsJsonObject();
                    categories.put(c.get("id").getAsLong(),
                        new CategoryInfo(c.get("id").getAsLong(), c.get("name").getAsString(),
                            c.has("supercategory") ? c.get("supercategory").getAsString() : ""));
                }
            }

            for (JsonElement img : root.getAsJsonArray("images")) {
                JsonObject i = img.getAsJsonObject();
                images.put(i.get("id").getAsLong(),
                    new ImageInfo(i.get("id").getAsLong(), imageDir.resolve(i.get("file_name").getAsString()),
                        i.get("width").getAsInt(), i.get("height").getAsInt()));
            }

            if (root.has("annotations")) {
                for (JsonElement ann : root.getAsJsonArray("annotations")) {
                    JsonObject a = ann.getAsJsonObject();
                    long imageId = a.get("image_id").getAsLong();
                    float[] bbox = null;
                    if (a.has("bbox")) {
                        JsonArray b = a.getAsJsonArray("bbox");
                        bbox = new float[]{b.get(0).getAsFloat(), b.get(1).getAsFloat(),
                            b.get(2).getAsFloat(), b.get(3).getAsFloat()};
                    }
                    annotations.computeIfAbsent(imageId, k -> new ArrayList<>())
                        .add(new Annotation(a.has("category_id") ? a.get("category_id").getAsLong() : 0, bbox));
                }
            }
        }

        List<ImageEntry> entries = new ArrayList<>();
        for (ImageInfo info : images.values()) {
            entries.add(new ImageEntry(info, annotations.getOrDefault(info.id, List.of())));
        }
        return new ParseResult(entries, categories);
    }

    @Override protected long totalSize() { return entries.size(); }

    @Override
    protected COCOSample getRaw(long index) {
        return new COCOSample(entries.get((int) index), categories, imageSize, arena);
    }

    @Override
    public Dataset<COCOSample> split(Split split) {
        try { return load(task, rootPath, split, imageSize); }
        catch (IOException e) { throw new RuntimeException(e); }
    }

    @Override
    public DatasetInfo info() {
        return DatasetInfo.builder("coco-" + task.name().toLowerCase())
            .description("COCO " + task.name() + " dataset")
            .totalSamples(entries.size())
            .splitSize(currentSplit, entries.size())
            .feature("image").feature("boxes").feature("labels")
            .extra("num_classes", categories.size())
            .build();
    }

    public int numClasses() { return categories.size(); }

    private record ImageInfo(long id, Path path, int width, int height) {}
    private record Annotation(long categoryId, float[] bbox) {}
    private record ImageEntry(ImageInfo image, List<Annotation> annotations) {}
    public record CategoryInfo(long id, String name, String supercategory) {}
    public record BoundingBox(float x, float y, float width, float height, long categoryId) {}

    public static class COCOSample implements Dataset.Sample {
        private final ImageEntry entry;
        private final Map<Long, CategoryInfo> categories;
        private final int imageSize;
        private final Arena arena;

        COCOSample(ImageEntry entry, Map<Long, CategoryInfo> categories, int imageSize, Arena arena) {
            this.entry = entry; this.categories = categories; this.imageSize = imageSize; this.arena = arena;
        }

        public long imageId() { return entry.image.id; }
        public Path imagePath() { return entry.image.path; }
        public int width() { return entry.image.width; }
        public int height() { return entry.image.height; }
        public int numObjects() { return entry.annotations.size(); }

        public List<BoundingBox> boundingBoxes() {
            List<BoundingBox> boxes = new ArrayList<>();
            for (Annotation a : entry.annotations) {
                if (a.bbox != null) boxes.add(new BoundingBox(a.bbox[0], a.bbox[1], a.bbox[2], a.bbox[3], a.categoryId));
            }
            return boxes;
        }

        public List<Long> labels() {
            return entry.annotations.stream().map(a -> a.categoryId).toList();
        }

        @Override
        public Map<String, TensorView> toTensors() {
            Map<String, TensorView> tensors = new HashMap<>();
            long[] imageShape = {3, imageSize, imageSize};
            long imageBytes = 3L * imageSize * imageSize * 4;
            MemorySegment imageSeg = arena.allocate(imageBytes);
            TensorInfo imageInfo = new TensorInfo("image", DType.F32, imageShape, 0, imageBytes);
            tensors.put("image", new TensorView(imageSeg, imageInfo));

            int n = entry.annotations.size();
            if (n > 0) {
                MemorySegment boxSeg = arena.allocate(n * 4L * 4);
                MemorySegment labelSeg = arena.allocate(n * 8L);
                int idx = 0;
                for (int i = 0; i < n; i++) {
                    Annotation a = entry.annotations.get(i);
                    for (int j = 0; j < 4; j++) boxSeg.setAtIndex(ValueLayout.JAVA_FLOAT, idx++, a.bbox != null ? a.bbox[j] : 0);
                    labelSeg.setAtIndex(ValueLayout.JAVA_LONG, i, a.categoryId);
                }
                tensors.put("boxes", new TensorView(boxSeg, new TensorInfo("boxes", DType.F32, new long[]{n, 4}, 0, n * 16L)));
                tensors.put("labels", new TensorView(labelSeg, new TensorInfo("labels", DType.I64, new long[]{n}, 0, n * 8L)));
            }
            return tensors;
        }
    }
}
