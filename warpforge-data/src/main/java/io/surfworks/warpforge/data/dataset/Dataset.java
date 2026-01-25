package io.surfworks.warpforge.data.dataset;

import io.surfworks.warpforge.data.TensorView;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Interface for ML datasets.
 *
 * <p>Provides a common abstraction for accessing training/validation data
 * with support for batching, shuffling, and transformations.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Load a dataset
 * Dataset<ImageClassificationSample> dataset = ImageNet.load(Path.of("/data/imagenet"))
 *     .split(Split.TRAIN)
 *     .shuffle(42)
 *     .batch(32);
 *
 * // Iterate over batches
 * for (Batch<ImageClassificationSample> batch : dataset.batches()) {
 *     TensorView images = batch.get("image");
 *     TensorView labels = batch.get("label");
 *     // Process batch...
 * }
 * }</pre>
 *
 * @param <T> The sample type for this dataset
 */
public interface Dataset<T extends Dataset.Sample> extends Iterable<T> {

    /**
     * Base interface for dataset samples.
     */
    interface Sample {
        /**
         * Convert sample to tensor map for model input.
         */
        Map<String, TensorView> toTensors();
    }

    /**
     * Dataset split types.
     */
    enum Split {
        TRAIN("train"),
        VALIDATION("validation", "val", "valid"),
        TEST("test");

        private final String[] aliases;

        Split(String... aliases) {
            this.aliases = aliases;
        }

        public String[] aliases() {
            return aliases;
        }

        public static Split fromString(String s) {
            String lower = s.toLowerCase();
            for (Split split : values()) {
                for (String alias : split.aliases) {
                    if (alias.equals(lower)) return split;
                }
            }
            throw new IllegalArgumentException("Unknown split: " + s);
        }
    }

    /**
     * Dataset name/identifier.
     */
    String name();

    /**
     * Number of samples in the dataset.
     */
    long size();

    /**
     * Get a sample by index.
     */
    T get(long index);

    /**
     * Get the current split.
     */
    Split split();

    /**
     * Select a specific split.
     */
    Dataset<T> split(Split split);

    /**
     * Shuffle the dataset with the given seed.
     */
    Dataset<T> shuffle(long seed);

    /**
     * Take the first n samples.
     */
    Dataset<T> take(long n);

    /**
     * Skip the first n samples.
     */
    Dataset<T> skip(long n);

    /**
     * Apply a transformation to each sample.
     */
    <U extends Sample> Dataset<U> map(Function<T, U> transform);

    /**
     * Filter samples by predicate.
     */
    Dataset<T> filter(java.util.function.Predicate<T> predicate);

    /**
     * Create batched iterator.
     */
    Iterable<Batch<T>> batches(int batchSize);

    /**
     * Get dataset metadata.
     */
    DatasetInfo info();

    /**
     * Iterator over samples.
     */
    @Override
    Iterator<T> iterator();

    /**
     * A batch of samples.
     */
    interface Batch<T extends Sample> {
        /**
         * Samples in this batch.
         */
        List<T> samples();

        /**
         * Batch size.
         */
        int size();

        /**
         * Get collated tensor by name.
         */
        TensorView get(String name);

        /**
         * Get all collated tensors.
         */
        Map<String, TensorView> tensors();
    }

    /**
     * Dataset metadata.
     */
    record DatasetInfo(
            String name,
            String description,
            long totalSamples,
            Map<Split, Long> splitSizes,
            List<String> featureNames,
            Map<String, Object> extras
    ) {
        public static Builder builder(String name) {
            return new Builder(name);
        }

        public static class Builder {
            private final String name;
            private String description = "";
            private long totalSamples = 0;
            private final java.util.EnumMap<Split, Long> splitSizes = new java.util.EnumMap<>(Split.class);
            private final List<String> featureNames = new java.util.ArrayList<>();
            private final Map<String, Object> extras = new java.util.HashMap<>();

            private Builder(String name) {
                this.name = name;
            }

            public Builder description(String description) {
                this.description = description;
                return this;
            }

            public Builder totalSamples(long totalSamples) {
                this.totalSamples = totalSamples;
                return this;
            }

            public Builder splitSize(Split split, long size) {
                this.splitSizes.put(split, size);
                return this;
            }

            public Builder feature(String name) {
                this.featureNames.add(name);
                return this;
            }

            public Builder extra(String key, Object value) {
                this.extras.put(key, value);
                return this;
            }

            public DatasetInfo build() {
                return new DatasetInfo(name, description, totalSamples,
                        Map.copyOf(splitSizes), List.copyOf(featureNames), Map.copyOf(extras));
            }
        }
    }
}
