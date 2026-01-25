package io.surfworks.warpforge.data.golden;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GoldenStoreTest {

    @TempDir
    Path tempDir;

    private GoldenStore store;
    private Arena arena;

    @BeforeEach
    void setUp() {
        store = GoldenStore.file(tempDir);
        arena = Arena.ofShared();
    }

    @Nested
    class BasicOperationTests {

        @Test
        void testSaveAndLoad() throws IOException {
            GoldenOutput original = createTestGolden("test-output");

            store.save(original);
            Optional<GoldenOutput> loaded = store.load("test-output");

            assertTrue(loaded.isPresent());
            assertEquals("test-output", loaded.get().id());
            assertEquals(DType.F32, loaded.get().tensorInfo().dtype());
            assertArrayEquals(new long[]{2, 3}, loaded.get().tensorInfo().shape());
        }

        @Test
        void testLoadNonExistent() throws IOException {
            Optional<GoldenOutput> loaded = store.load("does-not-exist");

            assertFalse(loaded.isPresent());
        }

        @Test
        void testExists() throws IOException {
            assertFalse(store.exists("test-output"));

            store.save(createTestGolden("test-output"));

            assertTrue(store.exists("test-output"));
        }

        @Test
        void testDelete() throws IOException {
            store.save(createTestGolden("test-output"));
            assertTrue(store.exists("test-output"));

            boolean deleted = store.delete("test-output");

            assertTrue(deleted);
            assertFalse(store.exists("test-output"));
        }

        @Test
        void testDeleteNonExistent() throws IOException {
            boolean deleted = store.delete("does-not-exist");

            assertFalse(deleted);
        }

        @Test
        void testOverwrite() throws IOException {
            // Save initial version
            GoldenOutput v1 = GoldenOutput.builder("test-output")
                    .tensor(createTensorView(new long[]{2}, new float[]{1, 2}))
                    .pytorchVersion("2.6.0")
                    .build();
            store.save(v1);

            // Overwrite with new version
            GoldenOutput v2 = GoldenOutput.builder("test-output")
                    .tensor(createTensorView(new long[]{2}, new float[]{3, 4}))
                    .pytorchVersion("2.7.0")
                    .build();
            store.save(v2);

            // Load and verify
            Optional<GoldenOutput> loaded = store.load("test-output");
            assertTrue(loaded.isPresent());
            assertEquals("2.7.0", loaded.get().pytorchVersion());

            TensorView view = loaded.get().toTensorView();
            assertEquals(3.0f, view.getFloat(0), 1e-6);
            assertEquals(4.0f, view.getFloat(1), 1e-6);
        }
    }

    @Nested
    class ListingTests {

        @Test
        void testList() throws IOException {
            store.save(createTestGolden("output-a"));
            store.save(createTestGolden("output-b"));
            store.save(createTestGolden("output-c"));

            List<String> ids = store.list();

            assertEquals(3, ids.size());
            assertTrue(ids.contains("output-a"));
            assertTrue(ids.contains("output-b"));
            assertTrue(ids.contains("output-c"));
        }

        @Test
        void testListEmpty() throws IOException {
            List<String> ids = store.list();

            assertTrue(ids.isEmpty());
        }

        @Test
        void testListByPrefix() throws IOException {
            store.save(createTestGolden("bert/pooler"));
            store.save(createTestGolden("bert/classifier"));
            store.save(createTestGolden("gpt2/output"));

            List<String> bertOutputs = store.listByPrefix("bert/");

            assertEquals(2, bertOutputs.size());
            assertTrue(bertOutputs.stream().allMatch(id -> id.startsWith("bert/")));
        }
    }

    @Nested
    class HierarchicalIdTests {

        @Test
        void testNestedId() throws IOException {
            store.save(createTestGolden("models/bert/base/pooler_output"));

            assertTrue(store.exists("models/bert/base/pooler_output"));

            Optional<GoldenOutput> loaded = store.load("models/bert/base/pooler_output");
            assertTrue(loaded.isPresent());
            assertEquals("models/bert/base/pooler_output", loaded.get().id());
        }

        @Test
        void testNestedIdCreatesDirectories() throws IOException {
            store.save(createTestGolden("level1/level2/level3/output"));

            Path expectedDir = tempDir.resolve("level1/level2/level3");
            assertTrue(Files.exists(expectedDir));
            assertTrue(Files.isDirectory(expectedDir));
        }

        @Test
        void testMixedDepthIds() throws IOException {
            store.save(createTestGolden("shallow"));
            store.save(createTestGolden("deep/nested/output"));

            List<String> all = store.list();

            assertEquals(2, all.size());
            assertTrue(all.contains("shallow"));
            assertTrue(all.contains("deep/nested/output"));
        }
    }

    @Nested
    class MetadataTests {

        @Test
        void testMetadataPreserved() throws IOException {
            Instant now = Instant.now();
            GoldenOutput original = GoldenOutput.builder("test")
                    .tensor(createTensorView(new long[]{2}, new float[]{1, 2}))
                    .pytorchVersion("2.7.0")
                    .modelId("bert-base-uncased")
                    .inputHash("abc123")
                    .description("Test golden output")
                    .tolerance(1e-4)
                    .metadata("custom_key", "custom_value")
                    .createdAt(now)
                    .build();

            store.save(original);
            GoldenOutput loaded = store.load("test").orElseThrow();

            assertEquals("2.7.0", loaded.pytorchVersion());
            assertEquals("bert-base-uncased", loaded.modelId());
            assertEquals("abc123", loaded.inputHash());
            assertEquals("Test golden output", loaded.metadata().get(GoldenOutput.KEY_DESCRIPTION));
            assertEquals(1e-4, loaded.tolerance(), 1e-10);
            assertEquals("custom_value", loaded.metadata().get("custom_key"));
        }

        @Test
        void testCreatedAtPreserved() throws IOException {
            Instant fixedTime = Instant.parse("2025-01-15T10:30:00Z");
            GoldenOutput original = GoldenOutput.builder("test")
                    .tensor(createTensorView(new long[]{2}, new float[]{1, 2}))
                    .createdAt(fixedTime)
                    .build();

            store.save(original);
            GoldenOutput loaded = store.load("test").orElseThrow();

            assertEquals(fixedTime, loaded.createdAt());
        }
    }

    @Nested
    class TensorDataTests {

        @Test
        void testTensorDataPreserved() throws IOException {
            float[] originalData = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
            GoldenOutput original = GoldenOutput.builder("test")
                    .tensor(createTensorView(new long[]{2, 3}, originalData))
                    .build();

            store.save(original);
            GoldenOutput loaded = store.load("test").orElseThrow();

            TensorView view = loaded.toTensorView();
            assertEquals(6, view.info().elementCount());

            // Check all values
            float[] loadedData = view.toFloatArray();
            assertArrayEquals(originalData, loadedData, 1e-6f);
        }

        @Test
        void testLargeTensor() throws IOException {
            // Create a larger tensor
            int size = 1000;
            float[] data = new float[size];
            for (int i = 0; i < size; i++) {
                data[i] = i * 0.1f;
            }

            GoldenOutput original = GoldenOutput.builder("large")
                    .tensor(createTensorView(new long[]{10, 100}, data))
                    .build();

            store.save(original);
            GoldenOutput loaded = store.load("large").orElseThrow();

            float[] loadedData = loaded.toTensorView().toFloatArray();
            assertArrayEquals(data, loadedData, 1e-6f);
        }

        @Test
        void testDifferentDTypes() throws IOException {
            // F32 tensor
            GoldenOutput f32 = GoldenOutput.builder("f32")
                    .tensor(DType.F32, new long[]{2}, createF32Segment(new float[]{1, 2}))
                    .build();
            store.save(f32);

            GoldenOutput loaded = store.load("f32").orElseThrow();
            assertEquals(DType.F32, loaded.tensorInfo().dtype());
        }
    }

    @Nested
    class ComparisonTests {

        @Test
        void testCompareWithinTolerance() throws IOException {
            GoldenOutput golden = GoldenOutput.builder("test")
                    .tensor(createTensorView(new long[]{3}, new float[]{1.0f, 2.0f, 3.0f}))
                    .tolerance(1e-4)
                    .build();

            store.save(golden);
            GoldenOutput loaded = store.load("test").orElseThrow();

            TensorView actual = createTensorView(new long[]{3},
                    new float[]{1.00001f, 2.00001f, 3.00001f});

            ComparisonResult result = loaded.compare(actual);

            assertTrue(result.matches());
        }

        @Test
        void testCompareExceedsTolerance() throws IOException {
            GoldenOutput golden = GoldenOutput.builder("test")
                    .tensor(createTensorView(new long[]{3}, new float[]{1.0f, 2.0f, 3.0f}))
                    .tolerance(1e-6)
                    .build();

            store.save(golden);
            GoldenOutput loaded = store.load("test").orElseThrow();

            TensorView actual = createTensorView(new long[]{3},
                    new float[]{1.0f, 2.1f, 3.0f});

            ComparisonResult result = loaded.compare(actual);

            assertFalse(result.matches());
            assertEquals(1, result.mismatchCount());
        }
    }

    // Helper methods

    private GoldenOutput createTestGolden(String id) {
        return GoldenOutput.builder(id)
                .tensor(createTensorView(new long[]{2, 3}, new float[]{1, 2, 3, 4, 5, 6}))
                .pytorchVersion("2.7.0")
                .build();
    }

    private TensorView createTensorView(long[] shape, float[] data) {
        MemorySegment segment = arena.allocate(data.length * 4L);
        for (int i = 0; i < data.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, data[i]);
        }
        TensorInfo info = new TensorInfo("test", DType.F32, shape, 0, segment.byteSize());
        return new TensorView(segment, info);
    }

    private MemorySegment createF32Segment(float[] data) {
        MemorySegment segment = arena.allocate(data.length * 4L);
        for (int i = 0; i < data.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, data[i]);
        }
        return segment;
    }
}
