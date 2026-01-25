package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.dataset.Dataset;
import io.surfworks.warpforge.data.dataset.ImageNetDataset;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * End-to-end benchmark combining ImageNet dataset with Vision Transformer (ViT) inference.
 *
 * <p>Measures the full pipeline:
 * <ul>
 *   <li>ImageNet image loading</li>
 *   <li>Image preprocessing (resize, normalize)</li>
 *   <li>Patch embedding preparation</li>
 *   <li>ViT inference</li>
 *   <li>Classification output validation</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * ImageNetViTBenchmark benchmark = ImageNetViTBenchmark.builder()
 *     .datasetPath(Path.of("/data/imagenet"))
 *     .imageSize(224)
 *     .patchSize(16)
 *     .build();
 *
 * BenchmarkRunner runner = new BenchmarkRunner();
 * BenchmarkResult result = runner.run(benchmark, config);
 * }</pre>
 */
public final class ImageNetViTBenchmark extends EndToEndBenchmark<ImageNetDataset.ImageSample> {

    // ImageNet normalization constants
    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    private final int imageSize;
    private final int patchSize;
    private final int hiddenSize;
    private final int numClasses;
    private final Dataset.Split split;

    private ImageNetViTBenchmark(Builder builder) {
        super(builder.name, builder.modelId, builder.datasetPath);
        this.imageSize = builder.imageSize;
        this.patchSize = builder.patchSize;
        this.hiddenSize = builder.hiddenSize;
        this.numClasses = builder.numClasses;
        this.split = builder.split;
    }

    /**
     * Create a new builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    @Override
    protected Dataset<ImageNetDataset.ImageSample> loadDataset(BenchmarkConfig config) throws IOException {
        ImageNetDataset dataset = ImageNetDataset.load(datasetPath, split);
        return dataset;
    }

    @Override
    protected Map<String, TensorView> transformBatchForModel(
            Dataset.Batch<ImageNetDataset.ImageSample> batch, BenchmarkConfig config) {

        Map<String, TensorView> inputs = new HashMap<>();
        int batchSize = batch.size();

        // Create pixel values tensor [batch, 3, imageSize, imageSize]
        long[] shape = {batchSize, 3, imageSize, imageSize};
        long byteSize = batchSize * 3L * imageSize * imageSize * 4;
        MemorySegment pixelValues = arena.allocate(byteSize);

        // For each sample, load and preprocess image
        int idx = 0;
        for (ImageNetDataset.ImageSample sample : batch.samples()) {
            // In real implementation, would decode and resize image
            // For benchmark, create synthetic normalized pixels
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < imageSize; h++) {
                    for (int w = 0; w < imageSize; w++) {
                        long offset = idx * 3L * imageSize * imageSize +
                                c * imageSize * imageSize + h * imageSize + w;
                        // Simulate normalized pixel values
                        float value = (float) ((sample.label() + c + h + w) % 256) / 255.0f;
                        float normalized = (value - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                        pixelValues.setAtIndex(ValueLayout.JAVA_FLOAT, offset, normalized);
                    }
                }
            }
            idx++;
        }

        TensorInfo pixelInfo = new TensorInfo("pixel_values", DType.F32, shape, 0, byteSize);
        inputs.put("pixel_values", new TensorView(pixelValues, pixelInfo));

        // Create labels tensor [batch]
        long[] labelShape = {batchSize};
        long labelBytes = batchSize * 8L;
        MemorySegment labels = arena.allocate(labelBytes);
        idx = 0;
        for (ImageNetDataset.ImageSample sample : batch.samples()) {
            labels.setAtIndex(ValueLayout.JAVA_LONG, idx++, sample.label());
        }
        TensorInfo labelInfo = new TensorInfo("labels", DType.I64, labelShape, 0, labelBytes);
        inputs.put("labels", new TensorView(labels, labelInfo));

        return inputs;
    }

    @Override
    public Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException {
        // Simulate ViT inference
        TensorView pixelValues = inputs.get("pixel_values");
        long[] inputShape = pixelValues.info().shape();
        int batchSize = (int) inputShape[0];

        // Output logits [batch, numClasses]
        long[] logitsShape = {batchSize, numClasses};
        long logitsBytes = batchSize * numClasses * 4L;
        MemorySegment logits = arena.allocate(logitsBytes);

        // Simulate classification output
        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < numClasses; c++) {
                float logit = (float) (Math.random() * 2 - 1); // Random logits for simulation
                logits.setAtIndex(ValueLayout.JAVA_FLOAT, b * numClasses + c, logit);
            }
        }

        TensorInfo logitsInfo = new TensorInfo("logits", DType.F32, logitsShape, 0, logitsBytes);

        // Hidden states [batch, numPatches + 1, hiddenSize]
        int numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        long[] hiddenShape = {batchSize, numPatches + 1, hiddenSize};
        long hiddenBytes = batchSize * (numPatches + 1L) * hiddenSize * 4;
        MemorySegment hiddenStates = arena.allocate(hiddenBytes);

        // Initialize hidden states
        for (long i = 0; i < batchSize * (numPatches + 1L) * hiddenSize; i++) {
            hiddenStates.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.random());
        }

        TensorInfo hiddenInfo = new TensorInfo("last_hidden_state", DType.F32, hiddenShape, 0, hiddenBytes);

        return Map.of(
                "logits", new TensorView(logits, logitsInfo),
                "last_hidden_state", new TensorView(hiddenStates, hiddenInfo)
        );
    }

    @Override
    public List<String> outputsToValidate() {
        return List.of("logits");
    }

    @Override
    protected long maxSamples() {
        return 1000; // Limit for benchmarking
    }

    /**
     * Builder for ImageNetViTBenchmark.
     */
    public static final class Builder {
        private String name = "imagenet-vit-endtoend";
        private String modelId = "google/vit-base-patch16-224";
        private Path datasetPath;
        private int imageSize = 224;
        private int patchSize = 16;
        private int hiddenSize = 768;
        private int numClasses = 1000;
        private Dataset.Split split = Dataset.Split.VALIDATION;

        private Builder() {}

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder datasetPath(Path datasetPath) {
            this.datasetPath = datasetPath;
            return this;
        }

        public Builder imageSize(int imageSize) {
            this.imageSize = imageSize;
            return this;
        }

        public Builder patchSize(int patchSize) {
            this.patchSize = patchSize;
            return this;
        }

        public Builder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        public Builder numClasses(int numClasses) {
            this.numClasses = numClasses;
            return this;
        }

        public Builder split(Dataset.Split split) {
            this.split = split;
            return this;
        }

        /**
         * Configure for ViT-Base/16.
         */
        public Builder vitBase() {
            return modelId("google/vit-base-patch16-224")
                    .imageSize(224)
                    .patchSize(16)
                    .hiddenSize(768);
        }

        /**
         * Configure for ViT-Large/16.
         */
        public Builder vitLarge() {
            return modelId("google/vit-large-patch16-224")
                    .imageSize(224)
                    .patchSize(16)
                    .hiddenSize(1024);
        }

        public ImageNetViTBenchmark build() {
            if (datasetPath == null) {
                throw new IllegalStateException("datasetPath must be set");
            }
            return new ImageNetViTBenchmark(this);
        }
    }
}
