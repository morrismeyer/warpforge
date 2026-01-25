package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.dataset.Dataset;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Base class for end-to-end benchmarks that combine dataset loading with model inference.
 *
 * <p>End-to-end benchmarks measure the full data pipeline:
 * <ul>
 *   <li>Dataset loading and preprocessing</li>
 *   <li>Batching and collation</li>
 *   <li>Model inference</li>
 *   <li>Output validation</li>
 * </ul>
 *
 * @param <S> Sample type from the dataset
 */
public abstract class EndToEndBenchmark<S extends Dataset.Sample> implements ModelBenchmark {

    protected final String name;
    protected final String modelId;
    protected final Path datasetPath;

    protected Dataset<S> dataset;
    protected Iterator<Dataset.Batch<S>> batchIterator;
    protected Arena arena;
    protected int currentBatchIndex;

    /**
     * Create an end-to-end benchmark.
     *
     * @param name Benchmark name
     * @param modelId Model identifier
     * @param datasetPath Path to the dataset
     */
    protected EndToEndBenchmark(String name, String modelId, Path datasetPath) {
        this.name = name;
        this.modelId = modelId;
        this.datasetPath = datasetPath;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public String modelId() {
        return modelId;
    }

    @Override
    public void setup(BenchmarkConfig config) throws IOException {
        this.arena = Arena.ofShared();
        this.dataset = loadDataset(config);
        this.currentBatchIndex = 0;

        // Create batch iterator
        Dataset<S> processedDataset = dataset;
        if (shouldShuffle()) {
            processedDataset = processedDataset.shuffle(42L); // Fixed seed for reproducibility
        }
        if (maxSamples() > 0) {
            processedDataset = processedDataset.take(maxSamples());
        }

        this.batchIterator = processedDataset.batches(config.batchSize()).iterator();
    }

    @Override
    public Map<String, TensorView> prepareInputs(BenchmarkConfig config) {
        // Get next batch, cycling if needed
        if (!batchIterator.hasNext()) {
            // Reset iterator with a different seed based on batch index
            Dataset<S> processedDataset = dataset;
            if (shouldShuffle()) {
                processedDataset = processedDataset.shuffle(42L + currentBatchIndex);
            }
            if (maxSamples() > 0) {
                processedDataset = processedDataset.take(maxSamples());
            }
            batchIterator = processedDataset.batches(config.batchSize()).iterator();
        }

        Dataset.Batch<S> batch = batchIterator.next();
        currentBatchIndex++;

        // Transform batch tensors for model input
        return transformBatchForModel(batch, config);
    }

    @Override
    public void teardown() {
        if (arena != null) {
            arena.close();
        }
    }

    /**
     * Load the dataset for this benchmark.
     */
    protected abstract Dataset<S> loadDataset(BenchmarkConfig config) throws IOException;

    /**
     * Transform batch tensors for model input.
     * Override to customize preprocessing.
     */
    protected Map<String, TensorView> transformBatchForModel(Dataset.Batch<S> batch, BenchmarkConfig config) {
        return batch.tensors();
    }

    /**
     * Whether to shuffle the dataset.
     */
    protected boolean shouldShuffle() {
        return true;
    }

    /**
     * Maximum number of samples to use (0 for all).
     */
    protected long maxSamples() {
        return 0;
    }

    /**
     * Get the underlying dataset (for testing).
     */
    public Dataset<S> getDataset() {
        return dataset;
    }

    /**
     * Create a padded tensor from variable-length inputs.
     */
    protected TensorView padSequences(List<int[]> sequences, int maxLen, int padValue) {
        int batchSize = sequences.size();
        long[] shape = {batchSize, maxLen};
        long byteSize = batchSize * maxLen * 4L;

        MemorySegment segment = arena.allocate(byteSize);

        for (int i = 0; i < batchSize; i++) {
            int[] seq = sequences.get(i);
            for (int j = 0; j < maxLen; j++) {
                int value = j < seq.length ? seq[j] : padValue;
                segment.setAtIndex(ValueLayout.JAVA_INT, i * maxLen + j, value);
            }
        }

        TensorInfo info = new TensorInfo("padded", DType.I32, shape, 0, byteSize);
        return new TensorView(segment, info);
    }

    /**
     * Create an attention mask from sequence lengths.
     */
    protected TensorView createAttentionMask(List<Integer> lengths, int maxLen) {
        int batchSize = lengths.size();
        long[] shape = {batchSize, maxLen};
        long byteSize = batchSize * maxLen * 4L;

        MemorySegment segment = arena.allocate(byteSize);

        for (int i = 0; i < batchSize; i++) {
            int len = lengths.get(i);
            for (int j = 0; j < maxLen; j++) {
                segment.setAtIndex(ValueLayout.JAVA_INT, i * maxLen + j, j < len ? 1 : 0);
            }
        }

        TensorInfo info = new TensorInfo("attention_mask", DType.I32, shape, 0, byteSize);
        return new TensorView(segment, info);
    }

    /**
     * Normalize image tensor to standard range.
     */
    protected TensorView normalizeImage(TensorView image, float[] mean, float[] std) {
        long[] shape = image.info().shape();
        long byteSize = image.info().size();

        MemorySegment segment = arena.allocate(byteSize);
        MemorySegment source = image.data();

        int channels = (int) shape[shape.length - 3];
        int height = (int) shape[shape.length - 2];
        int width = (int) shape[shape.length - 1];
        int pixelsPerChannel = height * width;

        int batchSize = shape.length == 4 ? (int) shape[0] : 1;

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                float m = mean[c % mean.length];
                float s = std[c % std.length];
                for (int p = 0; p < pixelsPerChannel; p++) {
                    long idx = b * channels * pixelsPerChannel + c * pixelsPerChannel + p;
                    float value = source.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                    float normalized = (value - m) / s;
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, idx, normalized);
                }
            }
        }

        return new TensorView(segment, image.info());
    }

    /**
     * Compute dataset statistics.
     */
    public DatasetStats computeDatasetStats() {
        if (dataset == null) {
            return new DatasetStats(0, 0, 0, 0);
        }

        long totalSamples = dataset.size();
        long totalElements = 0;
        long totalBytes = 0;
        int sampleCount = 0;

        // Sample up to 100 items for stats
        for (S sample : dataset.take(100)) {
            Map<String, TensorView> tensors = sample.toTensors();
            for (TensorView tensor : tensors.values()) {
                totalElements += tensor.info().elementCount();
                totalBytes += tensor.info().size();
            }
            sampleCount++;
        }

        double avgElementsPerSample = sampleCount > 0 ? (double) totalElements / sampleCount : 0;
        double avgBytesPerSample = sampleCount > 0 ? (double) totalBytes / sampleCount : 0;

        return new DatasetStats(totalSamples, avgElementsPerSample, avgBytesPerSample, sampleCount);
    }

    /**
     * Dataset statistics.
     */
    public record DatasetStats(
            long totalSamples,
            double avgElementsPerSample,
            double avgBytesPerSample,
            int sampledCount
    ) {
        public String summary() {
            return String.format(
                    "Dataset: %d samples, ~%.0f elements/sample, ~%.0f bytes/sample (sampled %d)",
                    totalSamples, avgElementsPerSample, avgBytesPerSample, sampledCount
            );
        }
    }
}
