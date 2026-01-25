package io.surfworks.warpforge.data.dataset;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * COCO dataset implementation for object detection and image captioning.
 *
 * <p>Supports the standard COCO annotation format with:
 * <ul>
 *   <li>Object detection annotations (bounding boxes, categories)</li>
 *   <li>Image captions</li>
 *   <li>Keypoint annotations</li>
 *   <li>Panoptic segmentation</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * CocoDataset dataset = CocoDataset.load("coco-2017", Path.of("/path/to/coco"));
 * for (Map<String, Object> example : dataset) {
 *     String imageFile = (String) example.get("file_name");
 *     List<Map<String, Object>> annotations = (List) example.get("annotations");
 *     // Process image and annotations...
 * }
 * }</pre>
 */
public final class CocoDataset implements DatasetSource {

    private static final Gson GSON = new Gson();

    private final String id;
    private final Path baseDir;
    private final String currentSplit;
    private final CocoAnnotationType annotationType;
    private final List<Map<String, Object>> data;

    // Category mappings
    private final Map<Long, String> categoryIdToName;
    private final Map<Long, String> categoryIdToSupercategory;

    /**
     * Type of COCO annotations to load.
     */
    public enum CocoAnnotationType {
        /** Object detection with bounding boxes */
        INSTANCES,
        /** Image captions */
        CAPTIONS,
        /** Keypoint detection */
        KEYPOINTS,
        /** Panoptic segmentation */
        PANOPTIC
    }

    private CocoDataset(String id, Path baseDir, String split, CocoAnnotationType annotationType) throws IOException {
        this.id = id;
        this.baseDir = baseDir;
        this.currentSplit = split;
        this.annotationType = annotationType;
        this.categoryIdToName = new HashMap<>();
        this.categoryIdToSupercategory = new HashMap<>();
        this.data = loadAnnotations(split, annotationType);
    }

    /**
     * Load a COCO dataset from a directory.
     *
     * @param id Dataset identifier
     * @param baseDir Base directory containing annotations/ and images/ subdirectories
     * @return The loaded dataset
     */
    public static CocoDataset load(String id, Path baseDir) throws IOException {
        return new CocoDataset(id, baseDir, "val2017", CocoAnnotationType.INSTANCES);
    }

    /**
     * Load a COCO dataset with specific split and annotation type.
     *
     * @param id Dataset identifier
     * @param baseDir Base directory
     * @param split Split name (e.g., "train2017", "val2017")
     * @param annotationType Type of annotations to load
     * @return The loaded dataset
     */
    public static CocoDataset load(String id, Path baseDir, String split,
                                    CocoAnnotationType annotationType) throws IOException {
        return new CocoDataset(id, baseDir, split, annotationType);
    }

    private List<Map<String, Object>> loadAnnotations(String split, CocoAnnotationType type) throws IOException {
        String annotationPrefix = switch (type) {
            case INSTANCES -> "instances_";
            case CAPTIONS -> "captions_";
            case KEYPOINTS -> "person_keypoints_";
            case PANOPTIC -> "panoptic_";
        };

        Path annotationFile = baseDir.resolve("annotations")
                .resolve(annotationPrefix + split + ".json");

        if (!Files.exists(annotationFile)) {
            throw new IOException("Annotation file not found: " + annotationFile);
        }

        String content = Files.readString(annotationFile);
        JsonObject root = GSON.fromJson(content, JsonObject.class);

        // Load categories
        if (root.has("categories")) {
            for (JsonElement cat : root.getAsJsonArray("categories")) {
                JsonObject catObj = cat.getAsJsonObject();
                long catId = catObj.get("id").getAsLong();
                String name = catObj.get("name").getAsString();
                categoryIdToName.put(catId, name);
                if (catObj.has("supercategory")) {
                    categoryIdToSupercategory.put(catId, catObj.get("supercategory").getAsString());
                }
            }
        }

        // Build image ID to image info map
        Map<Long, JsonObject> imageIdToInfo = new HashMap<>();
        if (root.has("images")) {
            for (JsonElement img : root.getAsJsonArray("images")) {
                JsonObject imgObj = img.getAsJsonObject();
                long imgId = imgObj.get("id").getAsLong();
                imageIdToInfo.put(imgId, imgObj);
            }
        }

        // Build image ID to annotations map
        Map<Long, List<JsonObject>> imageIdToAnnotations = new HashMap<>();
        if (root.has("annotations")) {
            for (JsonElement ann : root.getAsJsonArray("annotations")) {
                JsonObject annObj = ann.getAsJsonObject();
                long imgId = annObj.get("image_id").getAsLong();
                imageIdToAnnotations.computeIfAbsent(imgId, k -> new ArrayList<>()).add(annObj);
            }
        }

        // Build examples: one per image with all its annotations
        List<Map<String, Object>> examples = new ArrayList<>();
        for (Map.Entry<Long, JsonObject> entry : imageIdToInfo.entrySet()) {
            long imageId = entry.getKey();
            JsonObject imageInfo = entry.getValue();

            Map<String, Object> example = new HashMap<>();
            example.put("image_id", imageId);
            example.put("file_name", imageInfo.get("file_name").getAsString());
            example.put("width", imageInfo.get("width").getAsInt());
            example.put("height", imageInfo.get("height").getAsInt());

            if (imageInfo.has("coco_url")) {
                example.put("coco_url", imageInfo.get("coco_url").getAsString());
            }

            // Build image path
            Path imagePath = baseDir.resolve("images").resolve(split).resolve(imageInfo.get("file_name").getAsString());
            example.put("image_path", imagePath.toString());

            // Add annotations for this image
            List<JsonObject> annotations = imageIdToAnnotations.getOrDefault(imageId, List.of());
            List<Map<String, Object>> annotationList = new ArrayList<>();

            for (JsonObject ann : annotations) {
                Map<String, Object> annotation = parseAnnotation(ann, type);
                annotationList.add(annotation);
            }

            example.put("annotations", annotationList);
            example.put("num_annotations", annotationList.size());

            examples.add(example);
        }

        return examples;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> parseAnnotation(JsonObject ann, CocoAnnotationType type) {
        Map<String, Object> annotation = new HashMap<>();

        annotation.put("id", ann.get("id").getAsLong());
        annotation.put("image_id", ann.get("image_id").getAsLong());

        switch (type) {
            case INSTANCES -> {
                // Bounding box: [x, y, width, height]
                if (ann.has("bbox")) {
                    JsonArray bbox = ann.getAsJsonArray("bbox");
                    annotation.put("bbox", List.of(
                            bbox.get(0).getAsDouble(),
                            bbox.get(1).getAsDouble(),
                            bbox.get(2).getAsDouble(),
                            bbox.get(3).getAsDouble()
                    ));
                }

                // Category
                if (ann.has("category_id")) {
                    long catId = ann.get("category_id").getAsLong();
                    annotation.put("category_id", catId);
                    annotation.put("category_name", categoryIdToName.getOrDefault(catId, "unknown"));
                    annotation.put("supercategory", categoryIdToSupercategory.getOrDefault(catId, ""));
                }

                // Area and crowd flag
                if (ann.has("area")) {
                    annotation.put("area", ann.get("area").getAsDouble());
                }
                if (ann.has("iscrowd")) {
                    annotation.put("iscrowd", ann.get("iscrowd").getAsInt() == 1);
                }

                // Segmentation (polygon or RLE)
                if (ann.has("segmentation")) {
                    annotation.put("segmentation", GSON.fromJson(ann.get("segmentation"), Object.class));
                }
            }

            case CAPTIONS -> {
                if (ann.has("caption")) {
                    annotation.put("caption", ann.get("caption").getAsString());
                }
            }

            case KEYPOINTS -> {
                // Keypoints: [x1, y1, v1, x2, y2, v2, ...]
                if (ann.has("keypoints")) {
                    JsonArray kpts = ann.getAsJsonArray("keypoints");
                    List<Double> keypoints = new ArrayList<>();
                    for (JsonElement kpt : kpts) {
                        keypoints.add(kpt.getAsDouble());
                    }
                    annotation.put("keypoints", keypoints);
                }
                if (ann.has("num_keypoints")) {
                    annotation.put("num_keypoints", ann.get("num_keypoints").getAsInt());
                }

                // Also include bbox and category for keypoint annotations
                if (ann.has("bbox")) {
                    JsonArray bbox = ann.getAsJsonArray("bbox");
                    annotation.put("bbox", List.of(
                            bbox.get(0).getAsDouble(),
                            bbox.get(1).getAsDouble(),
                            bbox.get(2).getAsDouble(),
                            bbox.get(3).getAsDouble()
                    ));
                }
                if (ann.has("category_id")) {
                    long catId = ann.get("category_id").getAsLong();
                    annotation.put("category_id", catId);
                    annotation.put("category_name", categoryIdToName.getOrDefault(catId, "unknown"));
                }
            }

            case PANOPTIC -> {
                if (ann.has("segments_info")) {
                    annotation.put("segments_info", GSON.fromJson(ann.get("segments_info"), Object.class));
                }
                if (ann.has("file_name")) {
                    annotation.put("file_name", ann.get("file_name").getAsString());
                }
            }
        }

        return annotation;
    }

    /**
     * Get the category name for a category ID.
     */
    public String getCategoryName(long categoryId) {
        return categoryIdToName.getOrDefault(categoryId, "unknown");
    }

    /**
     * Get all category names.
     */
    public Map<Long, String> getCategories() {
        return Map.copyOf(categoryIdToName);
    }

    /**
     * Get the annotation type.
     */
    public CocoAnnotationType annotationType() {
        return annotationType;
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
        Path annotationsDir = baseDir.resolve("annotations");

        // Check for common COCO splits
        String[] possibleSplits = {"train2017", "val2017", "test2017", "train2014", "val2014"};
        String prefix = switch (annotationType) {
            case INSTANCES -> "instances_";
            case CAPTIONS -> "captions_";
            case KEYPOINTS -> "person_keypoints_";
            case PANOPTIC -> "panoptic_";
        };

        for (String split : possibleSplits) {
            if (Files.exists(annotationsDir.resolve(prefix + split + ".json"))) {
                available.add(split);
            }
        }

        return available;
    }

    @Override
    public DatasetSource split(String splitName) throws IOException {
        return new CocoDataset(id, baseDir, splitName, annotationType);
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
