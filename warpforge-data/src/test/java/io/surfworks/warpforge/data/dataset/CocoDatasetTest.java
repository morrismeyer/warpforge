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
        Files.createDirectories(tempDir.resolve("train2017"));
        Files.createDirectories(tempDir.resolve("val2017"));
    }

    @Nested
    class TaskEnumTests {

        @Test
        void testTaskAnnotationPrefixes() {
            assertEquals("instances", COCODataset.Task.DETECTION.annotationPrefix());
            assertEquals("instances", COCODataset.Task.SEGMENTATION.annotationPrefix());
            assertEquals("person_keypoints", COCODataset.Task.KEYPOINTS.annotationPrefix());
            assertEquals("captions", COCODataset.Task.CAPTIONS.annotationPrefix());
        }
    }

    @Nested
    class LoadTests {

        @Test
        void testLoadDetectionDataset() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(COCODataset.Task.DETECTION, tempDir);

            assertEquals("coco-detection", dataset.name());
            assertEquals(2, dataset.size()); // 2 images
        }

        @Test
        void testLoadWithSplit() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_val2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(
                    COCODataset.Task.DETECTION, tempDir, Dataset.Split.VALIDATION);

            assertEquals(2, dataset.size());
        }

        @Test
        void testLoadWithImageSize() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(
                    COCODataset.Task.DETECTION, tempDir, Dataset.Split.TRAIN, 224);

            assertEquals(2, dataset.size());
        }
    }

    @Nested
    class SampleTests {

        @Test
        void testSampleProperties() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(COCODataset.Task.DETECTION, tempDir);
            COCODataset.COCOSample sample = dataset.get(0);

            assertEquals(1, sample.imageId());
            assertEquals(640, sample.width());
            assertEquals(480, sample.height());
            assertEquals(2, sample.numObjects()); // 2 annotations for image 1
        }

        @Test
        void testBoundingBoxes() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(COCODataset.Task.DETECTION, tempDir);
            COCODataset.COCOSample sample = dataset.get(0);

            List<COCODataset.BoundingBox> boxes = sample.boundingBoxes();
            assertEquals(2, boxes.size());

            COCODataset.BoundingBox box = boxes.get(0);
            assertEquals(100.0f, box.x());
            assertEquals(100.0f, box.y());
            assertEquals(50.0f, box.width());
            assertEquals(100.0f, box.height());
            assertEquals(1, box.categoryId());
        }

        @Test
        void testLabels() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(COCODataset.Task.DETECTION, tempDir);
            COCODataset.COCOSample sample = dataset.get(0);

            List<Long> labels = sample.labels();
            assertEquals(2, labels.size());
            assertTrue(labels.contains(1L));
            assertTrue(labels.contains(2L));
        }

        @Test
        void testToTensors() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(COCODataset.Task.DETECTION, tempDir);
            COCODataset.COCOSample sample = dataset.get(0);

            var tensors = sample.toTensors();
            assertNotNull(tensors.get("image"));
            assertNotNull(tensors.get("boxes"));
            assertNotNull(tensors.get("labels"));
        }
    }

    @Nested
    class InfoTests {

        @Test
        void testDatasetInfo() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(COCODataset.Task.DETECTION, tempDir);
            Dataset.DatasetInfo info = dataset.info();

            assertEquals("coco-detection", info.name());
            assertTrue(info.description().contains("COCO"));
            assertEquals(2, info.totalSamples());
            assertTrue(info.featureNames().contains("image"));
            assertTrue(info.featureNames().contains("boxes"));
            assertTrue(info.featureNames().contains("labels"));
        }

        @Test
        void testNumClasses() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(COCODataset.Task.DETECTION, tempDir);
            assertEquals(2, dataset.numClasses());
        }
    }

    @Nested
    class IterationTests {

        @Test
        void testIteration() throws IOException {
            JsonObject root = createInstancesAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(COCODataset.Task.DETECTION, tempDir);

            int count = 0;
            for (COCODataset.COCOSample sample : dataset) {
                assertNotNull(sample);
                assertTrue(sample.imageId() > 0);
                count++;
            }
            assertEquals(2, count);
        }

        @Test
        void testDatasetOperations() throws IOException {
            JsonObject root = createLargerAnnotation(10);
            Files.writeString(
                    tempDir.resolve("annotations/instances_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(COCODataset.Task.DETECTION, tempDir);

            // Test take
            Dataset<COCODataset.COCOSample> taken = dataset.take(3);
            assertEquals(3, taken.size());

            // Test skip
            Dataset<COCODataset.COCOSample> skipped = dataset.skip(7);
            assertEquals(3, skipped.size());
        }
    }

    @Nested
    class ErrorTests {

        @Test
        void testMissingAnnotationFile() {
            assertThrows(IOException.class, () ->
                    COCODataset.load(COCODataset.Task.DETECTION, tempDir, Dataset.Split.TEST));
        }
    }

    @Nested
    class CaptionsTests {

        @Test
        void testCaptionsTask() throws IOException {
            JsonObject root = createCaptionsAnnotation();
            Files.writeString(
                    tempDir.resolve("annotations/captions_train2017.json"),
                    GSON.toJson(root)
            );

            COCODataset dataset = COCODataset.load(COCODataset.Task.CAPTIONS, tempDir);
            assertEquals("coco-captions", dataset.name());
            assertEquals(1, dataset.size());
        }
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

    private JsonObject createLargerAnnotation(int numImages) {
        JsonObject root = new JsonObject();

        JsonArray categories = new JsonArray();
        JsonObject cat = new JsonObject();
        cat.addProperty("id", 1);
        cat.addProperty("name", "object");
        cat.addProperty("supercategory", "thing");
        categories.add(cat);
        root.add("categories", categories);

        JsonArray images = new JsonArray();
        for (int i = 1; i <= numImages; i++) {
            JsonObject img = new JsonObject();
            img.addProperty("id", i);
            img.addProperty("file_name", String.format("%06d.jpg", i));
            img.addProperty("width", 640);
            img.addProperty("height", 480);
            images.add(img);
        }
        root.add("images", images);

        root.add("annotations", new JsonArray());

        return root;
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
        root.add("annotations", annotations);

        root.add("categories", new JsonArray());

        return root;
    }
}
