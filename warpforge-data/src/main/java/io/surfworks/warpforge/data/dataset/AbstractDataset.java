package io.surfworks.warpforge.data.dataset;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Abstract base implementation of Dataset.
 *
 * <p>Provides common functionality for shuffling, batching, mapping, and filtering.
 * Subclasses need to implement loading and sample access.
 */
public abstract class AbstractDataset<T extends Dataset.Sample> implements Dataset<T> {

    protected final String name;
    protected Split currentSplit;
    protected long[] indices;  // For shuffling
    protected long offset = 0;
    protected long limit = -1;

    protected AbstractDataset(String name, Split split) {
        this.name = name;
        this.currentSplit = split;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public Split split() {
        return currentSplit;
    }

    @Override
    public long size() {
        long total = totalSize();
        if (limit >= 0) {
            total = Math.min(limit, total - offset);
        } else {
            total = total - offset;
        }
        return Math.max(0, total);
    }

    /**
     * Get the total number of samples before offset/limit.
     */
    protected abstract long totalSize();

    /**
     * Get a sample by raw index (before shuffle/offset).
     */
    protected abstract T getRaw(long index);

    @Override
    public T get(long index) {
        if (index < 0 || index >= size()) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds for size " + size());
        }
        long rawIndex = offset + index;
        if (indices != null) {
            rawIndex = indices[(int) (offset + index)];
        }
        return getRaw(rawIndex);
    }

    @Override
    public Dataset<T> shuffle(long seed) {
        ShuffledDataset<T> shuffled = new ShuffledDataset<>(this, seed);
        return shuffled;
    }

    @Override
    public Dataset<T> take(long n) {
        LimitedDataset<T> limited = new LimitedDataset<>(this, 0, n);
        return limited;
    }

    @Override
    public Dataset<T> skip(long n) {
        LimitedDataset<T> skipped = new LimitedDataset<>(this, n, -1);
        return skipped;
    }

    @Override
    public <U extends Sample> Dataset<U> map(Function<T, U> transform) {
        return new MappedDataset<>(this, transform);
    }

    @Override
    public Dataset<T> filter(Predicate<T> predicate) {
        return new FilteredDataset<>(this, predicate);
    }

    @Override
    public Iterable<Batch<T>> batches(int batchSize) {
        return () -> new BatchIterator<>(this, batchSize);
    }

    @Override
    public Iterator<T> iterator() {
        return new SampleIterator<>(this);
    }

    /**
     * Simple iterator over samples.
     */
    private static class SampleIterator<T extends Sample> implements Iterator<T> {
        private final Dataset<T> dataset;
        private long index = 0;

        SampleIterator(Dataset<T> dataset) {
            this.dataset = dataset;
        }

        @Override
        public boolean hasNext() {
            return index < dataset.size();
        }

        @Override
        public T next() {
            if (!hasNext()) throw new NoSuchElementException();
            return dataset.get(index++);
        }
    }

    /**
     * Iterator over batches.
     */
    private static class BatchIterator<T extends Sample> implements Iterator<Batch<T>> {
        private final Dataset<T> dataset;
        private final int batchSize;
        private long index = 0;
        private final Arena arena = Arena.ofConfined();

        BatchIterator(Dataset<T> dataset, int batchSize) {
            this.dataset = dataset;
            this.batchSize = batchSize;
        }

        @Override
        public boolean hasNext() {
            return index < dataset.size();
        }

        @Override
        public Batch<T> next() {
            if (!hasNext()) throw new NoSuchElementException();

            List<T> samples = new ArrayList<>();
            long end = Math.min(index + batchSize, dataset.size());
            for (long i = index; i < end; i++) {
                samples.add(dataset.get(i));
            }
            index = end;

            return new SimpleBatch<>(samples, arena);
        }
    }

    /**
     * Simple batch implementation with tensor collation.
     */
    private static class SimpleBatch<T extends Sample> implements Batch<T> {
        private final List<T> samples;
        private final Map<String, TensorView> tensors;

        SimpleBatch(List<T> samples, Arena arena) {
            this.samples = samples;
            this.tensors = collate(samples, arena);
        }

        private Map<String, TensorView> collate(List<T> samples, Arena arena) {
            if (samples.isEmpty()) return Map.of();

            // Get tensor names from first sample
            Map<String, TensorView> first = samples.get(0).toTensors();
            Map<String, TensorView> result = new HashMap<>();

            for (String name : first.keySet()) {
                TensorView template = first.get(name);

                // Compute batched shape
                long[] templateShape = template.shape();
                long[] batchedShape = new long[templateShape.length + 1];
                batchedShape[0] = samples.size();
                System.arraycopy(templateShape, 0, batchedShape, 1, templateShape.length);

                // Compute total size
                long elementCount = samples.size();
                for (long dim : templateShape) elementCount *= dim;
                long byteSize = elementCount * template.dtype().byteSize();

                // Allocate and copy
                MemorySegment segment = arena.allocate(byteSize);
                long offset = 0;
                long sampleBytes = template.byteSize();
                for (T sample : samples) {
                    TensorView sampleTensor = sample.toTensors().get(name);
                    MemorySegment.copy(sampleTensor.data(), 0, segment, offset, sampleBytes);
                    offset += sampleBytes;
                }

                TensorInfo info = new TensorInfo(name, template.dtype(), batchedShape, 0, byteSize);
                result.put(name, new TensorView(segment, info));
            }

            return result;
        }

        @Override
        public List<T> samples() {
            return samples;
        }

        @Override
        public int size() {
            return samples.size();
        }

        @Override
        public TensorView get(String name) {
            return tensors.get(name);
        }

        @Override
        public Map<String, TensorView> tensors() {
            return tensors;
        }
    }

    /**
     * Shuffled view of a dataset.
     */
    private static class ShuffledDataset<T extends Sample> extends AbstractDataset<T> {
        private final AbstractDataset<T> source;
        private final long[] shuffledIndices;

        ShuffledDataset(AbstractDataset<T> source, long seed) {
            super(source.name, source.currentSplit);
            this.source = source;

            // Create shuffled indices
            int size = (int) source.size();
            this.shuffledIndices = new long[size];
            for (int i = 0; i < size; i++) {
                shuffledIndices[i] = i;
            }

            // Fisher-Yates shuffle
            Random random = new Random(seed);
            for (int i = size - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                long temp = shuffledIndices[i];
                shuffledIndices[i] = shuffledIndices[j];
                shuffledIndices[j] = temp;
            }
        }

        @Override
        protected long totalSize() {
            return shuffledIndices.length;
        }

        @Override
        protected T getRaw(long index) {
            return source.get(shuffledIndices[(int) index]);
        }

        @Override
        public Dataset<T> split(Split split) {
            return source.split(split).shuffle(42);
        }

        @Override
        public DatasetInfo info() {
            return source.info();
        }
    }

    /**
     * Limited/offset view of a dataset.
     */
    private static class LimitedDataset<T extends Sample> extends AbstractDataset<T> {
        private final AbstractDataset<T> source;
        private final long skipCount;
        private final long takeCount;

        LimitedDataset(AbstractDataset<T> source, long skip, long take) {
            super(source.name, source.currentSplit);
            this.source = source;
            this.skipCount = skip;
            this.takeCount = take;
        }

        @Override
        protected long totalSize() {
            long sourceSize = source.size();
            long afterSkip = Math.max(0, sourceSize - skipCount);
            return takeCount >= 0 ? Math.min(takeCount, afterSkip) : afterSkip;
        }

        @Override
        protected T getRaw(long index) {
            return source.get(skipCount + index);
        }

        @Override
        public Dataset<T> split(Split split) {
            return source.split(split);
        }

        @Override
        public DatasetInfo info() {
            return source.info();
        }
    }

    /**
     * Mapped view of a dataset.
     */
    private static class MappedDataset<T extends Sample, U extends Sample> extends AbstractDataset<U> {
        private final AbstractDataset<T> source;
        private final Function<T, U> transform;

        MappedDataset(AbstractDataset<T> source, Function<T, U> transform) {
            super(source.name, source.currentSplit);
            this.source = source;
            this.transform = transform;
        }

        @Override
        protected long totalSize() {
            return source.size();
        }

        @Override
        protected U getRaw(long index) {
            return transform.apply(source.get(index));
        }

        @Override
        public Dataset<U> split(Split split) {
            return new MappedDataset<>((AbstractDataset<T>) source.split(split), transform);
        }

        @Override
        public DatasetInfo info() {
            return source.info();
        }
    }

    /**
     * Filtered view of a dataset.
     */
    private static class FilteredDataset<T extends Sample> extends AbstractDataset<T> {
        private final AbstractDataset<T> source;
        private final long[] filteredIndices;

        FilteredDataset(AbstractDataset<T> source, Predicate<T> predicate) {
            super(source.name, source.currentSplit);
            this.source = source;

            // Collect matching indices
            List<Long> matching = new ArrayList<>();
            for (long i = 0; i < source.size(); i++) {
                if (predicate.test(source.get(i))) {
                    matching.add(i);
                }
            }

            this.filteredIndices = matching.stream().mapToLong(Long::longValue).toArray();
        }

        @Override
        protected long totalSize() {
            return filteredIndices.length;
        }

        @Override
        protected T getRaw(long index) {
            return source.get(filteredIndices[(int) index]);
        }

        @Override
        public Dataset<T> split(Split split) {
            throw new UnsupportedOperationException("Cannot change split on filtered dataset");
        }

        @Override
        public DatasetInfo info() {
            return source.info();
        }
    }
}
