package io.surfworks.warpforge.data.dataset;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DatasetTest {

    @Nested
    class SplitTests {

        @Test
        void testSplitFromString() {
            assertEquals(Dataset.Split.TRAIN, Dataset.Split.fromString("train"));
            assertEquals(Dataset.Split.VALIDATION, Dataset.Split.fromString("val"));
            assertEquals(Dataset.Split.VALIDATION, Dataset.Split.fromString("validation"));
            assertEquals(Dataset.Split.TEST, Dataset.Split.fromString("test"));
        }
    }

    @Nested
    class DatasetInfoTests {

        @Test
        void testDatasetInfoBuilder() {
            Dataset.DatasetInfo info = Dataset.DatasetInfo.builder("test-dataset")
                    .description("A test dataset")
                    .totalSamples(1000)
                    .splitSize(Dataset.Split.TRAIN, 800)
                    .splitSize(Dataset.Split.VALIDATION, 200)
                    .feature("input")
                    .feature("label")
                    .extra("num_classes", 10)
                    .build();

            assertEquals("test-dataset", info.name());
            assertEquals("A test dataset", info.description());
            assertEquals(1000, info.totalSamples());
            assertEquals(800L, info.splitSizes().get(Dataset.Split.TRAIN));
            assertEquals(200L, info.splitSizes().get(Dataset.Split.VALIDATION));
            assertTrue(info.featureNames().contains("input"));
            assertTrue(info.featureNames().contains("label"));
            assertEquals(10, info.extras().get("num_classes"));
        }
    }

    @Nested
    class InMemoryDatasetTests {

        @Test
        void testBasicIteration() {
            InMemoryDataset dataset = new InMemoryDataset(10);

            int count = 0;
            for (SimpleSample sample : dataset) {
                assertNotNull(sample);
                count++;
            }

            assertEquals(10, count);
        }

        @Test
        void testSize() {
            InMemoryDataset dataset = new InMemoryDataset(25);
            assertEquals(25, dataset.size());
        }

        @Test
        void testGet() {
            InMemoryDataset dataset = new InMemoryDataset(10);

            SimpleSample sample = dataset.get(5);
            assertEquals(5, sample.index);
        }

        @Test
        void testShuffle() {
            InMemoryDataset dataset = new InMemoryDataset(100);
            Dataset<SimpleSample> shuffled = dataset.shuffle(42);

            assertEquals(100, shuffled.size());

            // Check that order is different
            List<Integer> originalOrder = new ArrayList<>();
            List<Integer> shuffledOrder = new ArrayList<>();

            for (SimpleSample s : dataset) originalOrder.add(s.index);
            for (SimpleSample s : shuffled) shuffledOrder.add(s.index);

            assertNotEquals(originalOrder, shuffledOrder);
        }

        @Test
        void testTake() {
            InMemoryDataset dataset = new InMemoryDataset(100);
            Dataset<SimpleSample> taken = dataset.take(10);

            assertEquals(10, taken.size());

            List<Integer> indices = new ArrayList<>();
            for (SimpleSample s : taken) indices.add(s.index);

            assertEquals(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), indices);
        }

        @Test
        void testSkip() {
            InMemoryDataset dataset = new InMemoryDataset(100);
            Dataset<SimpleSample> skipped = dataset.skip(95);

            assertEquals(5, skipped.size());

            List<Integer> indices = new ArrayList<>();
            for (SimpleSample s : skipped) indices.add(s.index);

            assertEquals(List.of(95, 96, 97, 98, 99), indices);
        }

        @Test
        void testFilter() {
            InMemoryDataset dataset = new InMemoryDataset(100);
            Dataset<SimpleSample> filtered = dataset.filter(s -> s.index % 2 == 0);

            assertEquals(50, filtered.size());

            for (SimpleSample s : filtered) {
                assertTrue(s.index % 2 == 0);
            }
        }

        @Test
        void testMap() {
            InMemoryDataset dataset = new InMemoryDataset(10);
            Dataset<DoubleSample> mapped = dataset.map(s -> new DoubleSample(s.index * 2));

            assertEquals(10, mapped.size());

            int i = 0;
            for (DoubleSample s : mapped) {
                assertEquals(i * 2, s.value);
                i++;
            }
        }

        @Test
        void testBatches() {
            InMemoryDataset dataset = new InMemoryDataset(25);
            int batchSize = 8;

            List<Dataset.Batch<SimpleSample>> batches = new ArrayList<>();
            for (Dataset.Batch<SimpleSample> batch : dataset.batches(batchSize)) {
                batches.add(batch);
            }

            assertEquals(4, batches.size());  // 8 + 8 + 8 + 1
            assertEquals(8, batches.get(0).size());
            assertEquals(8, batches.get(1).size());
            assertEquals(8, batches.get(2).size());
            assertEquals(1, batches.get(3).size());
        }

        @Test
        void testBatchTensors() {
            InMemoryDataset dataset = new InMemoryDataset(4);

            for (Dataset.Batch<SimpleSample> batch : dataset.batches(4)) {
                assertEquals(4, batch.size());

                Map<String, TensorView> tensors = batch.tensors();
                assertTrue(tensors.containsKey("data"));

                TensorView data = batch.get("data");
                assertEquals(2, data.shape().length);
                assertEquals(4, data.shape()[0]);  // Batch dimension
            }
        }

        @Test
        void testChainedOperations() {
            InMemoryDataset dataset = new InMemoryDataset(100);

            Dataset<SimpleSample> processed = dataset
                    .shuffle(42)
                    .filter(s -> s.index % 3 == 0)
                    .take(10);

            assertEquals(10, processed.size());
        }
    }

    // Test implementations

    private static class SimpleSample implements Dataset.Sample {
        final int index;
        final Arena arena = Arena.ofConfined();

        SimpleSample(int index) {
            this.index = index;
        }

        @Override
        public Map<String, TensorView> toTensors() {
            long[] shape = {4};
            MemorySegment segment = arena.allocate(16);
            for (int i = 0; i < 4; i++) {
                segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, index + i * 0.1f);
            }
            TensorInfo info = new TensorInfo("data", DType.F32, shape, 0, 16);
            return Map.of("data", new TensorView(segment, info));
        }
    }

    private static class DoubleSample implements Dataset.Sample {
        final int value;
        final Arena arena = Arena.ofConfined();

        DoubleSample(int value) {
            this.value = value;
        }

        @Override
        public Map<String, TensorView> toTensors() {
            MemorySegment segment = arena.allocate(4);
            segment.set(ValueLayout.JAVA_INT, 0, value);
            TensorInfo info = new TensorInfo("value", DType.I32, new long[]{}, 0, 4);
            return Map.of("value", new TensorView(segment, info));
        }
    }

    private static class InMemoryDataset extends AbstractDataset<SimpleSample> {
        private final int size;

        InMemoryDataset(int size) {
            super("in-memory", Dataset.Split.TRAIN);
            this.size = size;
        }

        @Override
        protected long totalSize() {
            return size;
        }

        @Override
        protected SimpleSample getRaw(long index) {
            return new SimpleSample((int) index);
        }

        @Override
        public Dataset<SimpleSample> split(Dataset.Split split) {
            return this;
        }

        @Override
        public Dataset.DatasetInfo info() {
            return Dataset.DatasetInfo.builder("in-memory")
                    .totalSamples(size)
                    .build();
        }
    }
}
