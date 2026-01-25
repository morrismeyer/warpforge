package io.surfworks.warpforge.data.dataset;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CocoDatasetTest {

    @TempDir
    Path tempDir;

    private static final Gson GSON = new Gson();

    @BeforeEach
    void setUp() throws IOException {
        // Create COCO directory structure
        Files.createDirectories(tempDir.resolve("annotations"));
        Files.createDirectories(tempDir.resolve("images/val2017"));
    }

    @Nested
    class InstancesTests {

        @Test
        void testLoadInstanceAnnotations() throws IOException {
            // Create test instances annotation file
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_val2017.json"),
                    GSON.toJson(root)
            );

            CocoDataset dataset = CocoDataset.load("coco-test", tempDir, "val2017",
                    CocoDataset.CocoAnnotationType.INSTANCES);

            assertEquals("coco-test", dataset.id());
            assertEquals(2, dataset.size()); // 2 images
            assertEquals(CocoDataset.CocoAnnotationType.INSTANCES, dataset.annotationType());
        }

        @Test
        void testInstanceAnnotationFields() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_val2017.json"),
                    GSON.toJson(root)
            );

            CocoDataset dataset = CocoDataset.load("coco-test", tempDir, "val2017",
                    CocoDataset.CocoAnnotationType.INSTANCES);

            Map<String, Object> example = dataset.get(0);

            assertNotNull(example.get("image_id"));
            assertNotNull(example.get("file_name"));
            assertNotNull(example.get("width"));
            assertNotNull(example.get("height"));
            assertNotNull(example.get("annotations"));
            assertNotNull(example.get("num_annotations"));

            @SuppressWarnings("unchecked")
            List<Map<String, Object>> annotations = (List<Map<String, Object>>) example.get("annotations");
            assertFalse(annotations.isEmpty());

            Map<String, Object> annotation = annotations.get(0);
            assertNotNull(annotation.get("bbox"));
            assertNotNull(annotation.get("category_id"));
            assertNotNull(annotation.get("category_name"));
        }

        @Test
        void testBoundingBoxFormat() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_val2017.json"),
                    GSON.toJson(root)
            );

            CocoDataset dataset = CocoDataset.load("coco-test", tempDir, "val2017",
                    CocoDataset.CocoAnnotationType.INSTANCES);

            Map<String, Object> example = dataset.get(0);
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> annotations = (List<Map<String, Object>>) example.get("annotations");
            Map<String, Object> annotation = annotations.get(0);

            @SuppressWarnings("unchecked")
            List<Double> bbox = (List<Double>) annotation.get("bbox");
            assertEquals(4, bbox.size()); // [x, y, width, height]
        }

        private JsonObject createInstancesAnnotation() {
            JsonObject root = new JsonObject();

            // Categories
            JsonArray categories = new JsonArray();
            JsonObject cat1 = new JsonObject();
            cat1.addProperty("id", 1);
            cat1.addProperty("name", "person");
            cat1.addProperty("supercategory", "human");
            categories.add(cat1);

            JsonObject cat2 = new JsonObject();
            cat2.addProperty("id", 2);
            cat2.addProperty("name", "car");
            cat2.addProperty("supercategory", "vehicle");
            categories.add(cat2);

            root.add("categories", categories);

            // Images
            JsonArray images = new JsonArray();
            JsonObject img1 = new JsonObject();
            img1.addProperty("id", 1);
            img1.addProperty("file_name", "000001.jpg");
            img1.addProperty("width", 640);
            img1.addProperty("height", 480);
            images.add(img1);

            JsonObject img2 = new JsonObject();
            img2.addProperty("id", 2);
            img2.addProperty("file_name", "000002.jpg");
            img2.addProperty("width", 800);
            img2.addProperty("height", 600);
            images.add(img2);

            root.add("images", images);

            // Annotations
            JsonArray annotations = new JsonArray();

            JsonObject ann1 = new JsonObject();
            ann1.addProperty("id", 1);
            ann1.addProperty("image_id", 1);
            ann1.addProperty("category_id", 1);
            JsonArray bbox1 = new JsonArray();
            bbox1.add(100); bbox1.add(100); bbox1.add(50); bbox1.add(100);
            ann1.add("bbox", bbox1);
            ann1.addProperty("area", 5000);
            ann1.addProperty("iscrowd", 0);
            annotations.add(ann1);

            JsonObject ann2 = new JsonObject();
            ann2.addProperty("id", 2);
            ann2.addProperty("image_id", 1);
            ann2.addProperty("category_id", 2);
            JsonArray bbox2 = new JsonArray();
            bbox2.add(200); bbox2.add(150); bbox2.add(100); bbox2.add(80);
            ann2.add("bbox", bbox2);
            ann2.addProperty("area", 8000);
            ann2.addProperty("iscrowd", 0);
            annotations.add(ann2);

            JsonObject ann3 = new JsonObject();
            ann3.addProperty("id", 3);
            ann3.addProperty("image_id", 2);
            ann3.addProperty("category_id", 1);
            JsonArray bbox3 = new JsonArray();
            bbox3.add(50); bbox3.add(50); bbox3.add(200); bbox3.add(300);
            ann3.add("bbox", bbox3);
            ann3.addProperty("area", 60000);
            ann3.addProperty("iscrowd", 0);
            annotations.add(ann3);

            root.add("annotations", annotations);

            return root;
        }
    }

    @Nested
    class CaptionsTests {

        @Test
        void testLoadCaptionAnnotations() throws IOException {
            JsonObject root = createCaptionsAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/captions_val2017.json"),
                    GSON.toJson(root)
            );

            CocoDataset dataset = CocoDataset.load("coco-captions", tempDir, "val2017",
                    CocoDataset.CocoAnnotationType.CAPTIONS);

            assertEquals(1, dataset.size());

            Map<String, Object> example = dataset.get(0);
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> annotations = (List<Map<String, Object>>) example.get("annotations");
            assertEquals(2, annotations.size()); // 2 captions for the image

            assertTrue(annotations.get(0).containsKey("caption"));
        }

        private JsonObject createCaptionsAnnotation() {
            JsonObject root = new JsonObject();

            JsonArray images = new JsonArray();
            JsonObject img = new JsonObject();
            img.addProperty("id", 1);
            img.addProperty("file_name", "000001.jpg");
            img.addProperty("width", 640);
            img.addProperty("height", 480);
            images.add(img);
            root.add("images", images);

            JsonArray annotations = new JsonArray();

            JsonObject ann1 = new JsonObject();
            ann1.addProperty("id", 1);
            ann1.addProperty("image_id", 1);
            ann1.addProperty("caption", "A person standing in front of a car.");
            annotations.add(ann1);

            JsonObject ann2 = new JsonObject();
            ann2.addProperty("id", 2);
            ann2.addProperty("image_id", 1);
            ann2.addProperty("caption", "Someone next to a vehicle on the street.");
            annotations.add(ann2);

            root.add("annotations", annotations);
            root.add("categories", new JsonArray());

            return root;
        }
    }

    @Nested
    class SplitTests {

        @Test
        void testAvailableSplits() throws IOException {
            // Create both train and val annotations
            JsonObject annotation = createMinimalAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(annotation)
            );
            Files.writeString(
                    tempDir.resolve("annotations/instances_val2017.json"),
                    GSON.toJson(annotation)
            );

            CocoDataset dataset = CocoDataset.load("coco-test", tempDir, "val2017",
                    CocoDataset.CocoAnnotationType.INSTANCES);

            List<String> splits = dataset.splits();
            assertTrue(splits.contains("train2017"));
            assertTrue(splits.contains("val2017"));
        }

        @Test
        void testChangeSplit() throws IOException {
            JsonObject annotation = createMinimalAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(annotation)
            );
            Files.writeString(
                    tempDir.resolve("annotations/instances_val2017.json"),
                    GSON.toJson(annotation)
            );

            CocoDataset valDataset = CocoDataset.load("coco-test", tempDir, "val2017",
                    CocoDataset.CocoAnnotationType.INSTANCES);
            DatasetSource trainDataset = valDataset.split("train2017");

            assertNotNull(trainDataset);
            assertEquals("coco-test", trainDataset.id());
        }

        private JsonObject createMinimalAnnotation() {
            JsonObject root = new JsonObject();
            root.add("images", new JsonArray());
            root.add("annotations", new JsonArray());
            root.add("categories", new JsonArray());
            return root;
        }
    }

    @Nested
    class CategoryTests {

        @Test
        void testGetCategories() throws IOException {
            JsonObject root = new JsonObject();

            JsonArray categories = new JsonArray();
            JsonObject cat = new JsonObject();
            cat.addProperty("id", 1);
            cat.addProperty("name", "dog");
            cat.addProperty("supercategory", "animal");
            categories.add(cat);
            root.add("categories", categories);

            root.add("images", new JsonArray());
            root.add("annotations", new JsonArray());

            Files.writeString(
                    tempDir.resolve("annotations/instances_val2017.json"),
                    GSON.toJson(root)
            );

            CocoDataset dataset = CocoDataset.load("coco-test", tempDir, "val2017",
                    CocoDataset.CocoAnnotationType.INSTANCES);

            assertEquals("dog", dataset.getCategoryName(1));
            assertEquals("unknown", dataset.getCategoryName(999));

            Map<Long, String> allCategories = dataset.getCategories();
            assertEquals(1, allCategories.size());
            assertEquals("dog", allCategories.get(1L));
        }
    }

    @Nested
    class ErrorTests {

        @Test
        void testMissingAnnotationFile() {
            assertThrows(IOException.class, () ->
                    CocoDataset.load("missing", tempDir, "nonexistent",
                            CocoDataset.CocoAnnotationType.INSTANCES));
        }
    }

    @Nested
    class IterationTests {

        @Test
        void testIteration() throws IOException {
            JsonObject root = new JsonObject();

            JsonArray images = new JsonArray();
            for (int i = 1; i <= 5; i++) {
                JsonObject img = new JsonObject();
                img.addProperty("id", i);
                img.addProperty("file_name", String.format("%06d.jpg", i));
                img.addProperty("width", 640);
                img.addProperty("height", 480);
                images.add(img);
            }
            root.add("images", images);
            root.add("annotations", new JsonArray());
            root.add("categories", new JsonArray());

            Files.writeString(
                    tempDir.resolve("annotations/instances_val2017.json"),
                    GSON.toJson(root)
            );

            CocoDataset dataset = CocoDataset.load("coco-test", tempDir);

            int count = 0;
            for (Map<String, Object> example : dataset) {
                assertNotNull(example.get("image_id"));
                count++;
            }
            assertEquals(5, count);
        }

        @Test
        void testStreamAndBatch() throws IOException {
            JsonObject root = new JsonObject();

            JsonArray images = new JsonArray();
            for (int i = 1; i <= 10; i++) {
                JsonObject img = new JsonObject();
                img.addProperty("id", i);
                img.addProperty("file_name", String.format("%06d.jpg", i));
                img.addProperty("width", 640);
                img.addProperty("height", 480);
                images.add(img);
            }
            root.add("images", images);
            root.add("annotations", new JsonArray());
            root.add("categories", new JsonArray());

            Files.writeString(
                    tempDir.resolve("annotations/instances_val2017.json"),
                    GSON.toJson(root)
            );

            CocoDataset dataset = CocoDataset.load("coco-test", tempDir);

            // Test stream
            assertEquals(10, dataset.stream().count());

            // Test batch
            List<Map<String, Object>> batch = dataset.getBatch(0, 3);
            assertEquals(3, batch.size());

            batch = dataset.getBatch(8, 5);
            assertEquals(2, batch.size()); // Only 2 remaining
        }
    }
}
