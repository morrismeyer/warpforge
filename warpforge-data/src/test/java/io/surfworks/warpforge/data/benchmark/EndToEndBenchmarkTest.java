package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.dataset.Dataset;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndToEndBenchmarkTest {

    // Simple test sample
    private static class TestSample implements Dataset.Sample {
        final int id;
        final Arena arena = Arena.ofConfined();

        TestSample(int id) {
            this.id = id;
        }

        @Override
        public Map<String, TensorView> toTensors() {
            MemorySegment segment = arena.allocate(4);
            segment.set(ValueLayout.JAVA_INT, 0, id);
            TensorInfo info = new TensorInfo("id", DType.I32, new long[]{}, 0, 4);
            return Map.of("id", new TensorView(segment, info));
        }
    }

    // Simple test dataset
    private static class TestDataset implements Dataset<TestSample> {
        private final int size;

        TestDataset(int size) {
            this.size = size;
        }

        @Override
        public String name() { return "test-dataset"; }

        @Override
        public long size() { return size; }

        @Override
        public TestSample get(long index) { return new TestSample((int) index); }

        @Override
        public Dataset<TestSample> shuffle(long seed) { return this; }

        @Override
        public Dataset<TestSample> take(long n) {
            return new TestDataset((int) Math.min(n, size));
        }

        @Override
        public Dataset<TestSample> skip(long n) {
            return new TestDataset((int) Math.max(0, size - n));
        }

        @Override
        public <U extends Sample> Dataset<U> map(java.util.function.Function<TestSample, U> transform) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Dataset<TestSample> filter(java.util.function.Predicate<TestSample> predicate) {
            throw new UnsupportedOperationException();
        }

        @Override
        public Iterable<Batch<TestSample>> batches(int batchSize) {
            return () -> new java.util.Iterator<>() {
                int pos = 0;

                @Override
                public boolean hasNext() { return pos < size; }

                @Override
                public Batch<TestSample> next() {
                    int end = Math.min(pos + batchSize, size);
                    List<TestSample> samples = new java.util.ArrayList<>();
                    for (int i = pos; i < end; i++) {
                        samples.add(get(i));
                    }
                    pos = end;
                    return new SimpleBatch(samples);
                }
            };
        }

        @Override
        public Split split() { return Split.TRAIN; }

        @Override
        public Dataset<TestSample> split(Split split) { return this; }

        @Override
        public DatasetInfo info() {
            return DatasetInfo.builder("test-dataset").totalSamples(size).build();
        }

        @Override
        public java.util.Iterator<TestSample> iterator() {
            return new java.util.Iterator<>() {
                int pos = 0;

                @Override
                public boolean hasNext() { return pos < size; }

                @Override
                public TestSample next() { return get(pos++); }
            };
        }

        private class SimpleBatch implements Batch<TestSample> {
            private final List<TestSample> samples;

            SimpleBatch(List<TestSample> samples) {
                this.samples = samples;
            }

            @Override
            public int size() { return samples.size(); }

            @Override
            public List<TestSample> samples() { return samples; }

            @Override
            public Map<String, TensorView> tensors() {
                return Map.of();
            }

            @Override
            public TensorView get(String name) { return null; }
        }
    }

    // Concrete test benchmark
    private static class TestEndToEndBenchmark extends EndToEndBenchmark<TestSample> {

        TestEndToEndBenchmark() {
            super("test-endtoend", "test-model", Path.of("/tmp"));
        }

        @Override
        protected Dataset<TestSample> loadDataset(BenchmarkConfig config) {
            return new TestDataset(100);
        }

        @Override
        public Map<String, TensorView> runInference(Map<String, TensorView> inputs) {
            return Map.of();
        }

        @Override
        public List<String> outputsToValidate() {
            return List.of();
        }
    }

    @Nested
    class BasicTests {

        @Test
        void testName() {
            TestEndToEndBenchmark benchmark = new TestEndToEndBenchmark();
            assertEquals("test-endtoend", benchmark.name());
        }

        @Test
        void testModelId() {
            TestEndToEndBenchmark benchmark = new TestEndToEndBenchmark();
            assertEquals("test-model", benchmark.modelId());
        }
    }

    @Nested
    class SetupTests {

        @Test
        void testSetupLoadsDataset() throws IOException {
            TestEndToEndBenchmark benchmark = new TestEndToEndBenchmark();
            BenchmarkConfig config = BenchmarkConfig.builder("test-model")
                    .batchSize(8)
                    .build();

            benchmark.setup(config);

            assertNotNull(benchmark.getDataset());
            assertEquals(100, benchmark.getDataset().size());

            benchmark.teardown();
        }
    }

    @Nested
    class PrepareInputsTests {

        @Test
        void testPrepareInputsReturnsBatch() throws IOException {
            TestEndToEndBenchmark benchmark = new TestEndToEndBenchmark();
            BenchmarkConfig config = BenchmarkConfig.builder("test-model")
                    .batchSize(8)
                    .build();

            benchmark.setup(config);
            Map<String, TensorView> inputs = benchmark.prepareInputs(config);

            assertNotNull(inputs);

            benchmark.teardown();
        }

        @Test
        void testPrepareInputsCyclesIterator() throws IOException {
            TestEndToEndBenchmark benchmark = new TestEndToEndBenchmark();
            BenchmarkConfig config = BenchmarkConfig.builder("test-model")
                    .batchSize(50)
                    .build();

            benchmark.setup(config);

            // First call
            benchmark.prepareInputs(config);
            // Second call
            benchmark.prepareInputs(config);
            // Third call should cycle
            Map<String, TensorView> inputs = benchmark.prepareInputs(config);

            assertNotNull(inputs);

            benchmark.teardown();
        }
    }

    @Nested
    class DatasetStatsTests {

        @Test
        void testComputeDatasetStats() throws IOException {
            TestEndToEndBenchmark benchmark = new TestEndToEndBenchmark();
            BenchmarkConfig config = BenchmarkConfig.builder("test-model")
                    .batchSize(8)
                    .build();

            benchmark.setup(config);
            EndToEndBenchmark.DatasetStats stats = benchmark.computeDatasetStats();

            assertNotNull(stats);
            assertEquals(100, stats.totalSamples());
            assertTrue(stats.sampledCount() > 0);

            benchmark.teardown();
        }

        @Test
        void testDatasetStatsSummary() throws IOException {
            TestEndToEndBenchmark benchmark = new TestEndToEndBenchmark();
            BenchmarkConfig config = BenchmarkConfig.builder("test-model")
                    .batchSize(8)
                    .build();

            benchmark.setup(config);
            EndToEndBenchmark.DatasetStats stats = benchmark.computeDatasetStats();

            String summary = stats.summary();
            assertNotNull(summary);
            assertTrue(summary.contains("100 samples"));

            benchmark.teardown();
        }
    }

    @Nested
    class HelperMethodTests {

        @Test
        void testPadSequences() throws IOException {
            TestEndToEndBenchmark benchmark = new TestEndToEndBenchmark();
            BenchmarkConfig config = BenchmarkConfig.builder("test-model").build();
            benchmark.setup(config);

            List<int[]> sequences = List.of(
                    new int[]{1, 2, 3},
                    new int[]{4, 5}
            );

            TensorView padded = benchmark.padSequences(sequences, 5, 0);

            assertNotNull(padded);
            assertEquals(2, padded.info().shape()[0]);
            assertEquals(5, padded.info().shape()[1]);

            benchmark.teardown();
        }

        @Test
        void testCreateAttentionMask() throws IOException {
            TestEndToEndBenchmark benchmark = new TestEndToEndBenchmark();
            BenchmarkConfig config = BenchmarkConfig.builder("test-model").build();
            benchmark.setup(config);

            List<Integer> lengths = List.of(3, 5, 2);

            TensorView mask = benchmark.createAttentionMask(lengths, 8);

            assertNotNull(mask);
            assertEquals(3, mask.info().shape()[0]);
            assertEquals(8, mask.info().shape()[1]);

            benchmark.teardown();
        }
    }
}
