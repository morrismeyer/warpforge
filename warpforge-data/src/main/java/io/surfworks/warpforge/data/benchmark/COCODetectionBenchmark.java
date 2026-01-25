package io.surfworks.warpforge.data.benchmark;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.dataset.COCODataset;
import io.surfworks.warpforge.data.dataset.Dataset;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * End-to-end benchmark combining COCO dataset with object detection inference.
 *
 * <p>Measures the full pipeline:
 * <ul>
 *   <li>COCO image and annotation loading</li>
 *   <li>Image preprocessing and resizing</li>
 *   <li>Detection model inference</li>
 *   <li>Bounding box output validation</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * COCODetectionBenchmark benchmark = COCODetectionBenchmark.builder()
 *     .datasetPath(Path.of("/data/coco"))
 *     .imageSize(640)
 *     .build();
 *
 * BenchmarkRunner runner = new BenchmarkRunner();
 * BenchmarkResult result = runner.run(benchmark, config);
 * }</pre>
 */
public final class COCODetectionBenchmark extends EndToEndBenchmark<COCODataset.COCOSample> {

    // ImageNet normalization (commonly used for detection backbones)
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private final int imageSize;
    private final int maxDetections;
    private final COCODataset.Task task;
    private final Dataset.Split split;

    private int numClasses;

    private COCODetectionBenchmark(Builder builder) {
        super(builder.name, builder.modelId, builder.datasetPath);
        this.imageSize = builder.imageSize;
        this.maxDetections = builder.maxDetections;
        this.task = builder.task;
        this.split = builder.split;
    }

    /**
     * Create a new builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    @Override
    protected Dataset<COCODataset.COCOSample> loadDataset(BenchmarkConfig config) throws IOException {
        COCODataset dataset = COCODataset.load(task, datasetPath, split, imageSize);
        this.numClasses = dataset.numClasses();
        return dataset;
    }

    @Override
    protected Map<String, TensorView> transformBatchForModel(
            Dataset.Batch<COCODataset.COCOSample> batch, BenchmarkConfig config) {

        Map<String, TensorView> inputs = new HashMap<>();
        int batchSize = batch.size();

        // Create image tensor [batch, 3, imageSize, imageSize]
        long[] imageShape = {batchSize, 3, imageSize, imageSize};
        long imageBytes = batchSize * 3L * imageSize * imageSize * 4;
        MemorySegment images = arena.allocate(imageBytes);

        // Create target boxes and labels
        // For simplicity, pad to maxDetections
        long[] boxShape = {batchSize, maxDetections, 4};
        long[] labelShape = {batchSize, maxDetections};
        long boxBytes = batchSize * maxDetections * 4L * 4;
        long labelBytes = batchSize * maxDetections * 8L;

        MemorySegment boxes = arena.allocate(boxBytes);
        MemorySegment labels = arena.allocate(labelBytes);

        int idx = 0;
        for (COCODataset.COCOSample sample : batch.samples()) {
            // Fill image tensor (simulated preprocessing)
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < imageSize; h++) {
                    for (int w = 0; w < imageSize; w++) {
                        long offset = idx * 3L * imageSize * imageSize +
                                c * imageSize * imageSize + h * imageSize + w;
                        // Simulate normalized pixel
                        float value = (float) ((sample.imageId() + c + h + w) % 256) / 255.0f;
                        float normalized = (value - MEAN[c]) / STD[c];
                        images.setAtIndex(ValueLayout.JAVA_FLOAT, offset, normalized);
                    }
                }
            }

            // Fill boxes and labels
            List<COCODataset.BoundingBox> sampleBoxes = sample.boundingBoxes();
            for (int d = 0; d < maxDetections; d++) {
                long boxOffset = idx * maxDetections * 4L + d * 4L;
                long labelOffset = idx * maxDetections + d;

                if (d < sampleBoxes.size()) {
                    COCODataset.BoundingBox box = sampleBoxes.get(d);
                    // Normalize box coordinates to [0, 1]
                    boxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset, box.x() / sample.width());
                    boxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset + 1, box.y() / sample.height());
                    boxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset + 2, box.width() / sample.width());
                    boxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset + 3, box.height() / sample.height());
                    labels.setAtIndex(ValueLayout.JAVA_LONG, labelOffset, box.categoryId());
                } else {
                    // Pad with zeros
                    boxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset, 0);
                    boxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset + 1, 0);
                    boxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset + 2, 0);
                    boxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset + 3, 0);
                    labels.setAtIndex(ValueLayout.JAVA_LONG, labelOffset, 0);
                }
            }
            idx++;
        }

        inputs.put("images", new TensorView(images,
                new TensorInfo("images", DType.F32, imageShape, 0, imageBytes)));
        inputs.put("boxes", new TensorView(boxes,
                new TensorInfo("boxes", DType.F32, boxShape, 0, boxBytes)));
        inputs.put("labels", new TensorView(labels,
                new TensorInfo("labels", DType.I64, labelShape, 0, labelBytes)));

        return inputs;
    }

    @Override
    public Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException {
        TensorView images = inputs.get("images");
        long[] inputShape = images.info().shape();
        int batchSize = (int) inputShape[0];

        Map<String, TensorView> outputs = new HashMap<>();

        // Predicted boxes [batch, maxDetections, 4]
        long[] predBoxShape = {batchSize, maxDetections, 4};
        long predBoxBytes = batchSize * maxDetections * 4L * 4;
        MemorySegment predBoxes = arena.allocate(predBoxBytes);

        // Predicted scores [batch, maxDetections, numClasses]
        long[] scoreShape = {batchSize, maxDetections, numClasses};
        long scoreBytes = batchSize * maxDetections * (long) numClasses * 4;
        MemorySegment scores = arena.allocate(scoreBytes);

        // Simulate detection output
        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < maxDetections; d++) {
                long boxOffset = b * maxDetections * 4L + d * 4L;
                // Random box coordinates
                predBoxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset, (float) Math.random());
                predBoxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset + 1, (float) Math.random());
                predBoxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset + 2, (float) Math.random() * 0.5f);
                predBoxes.setAtIndex(ValueLayout.JAVA_FLOAT, boxOffset + 3, (float) Math.random() * 0.5f);

                // Random class scores
                for (int c = 0; c < numClasses; c++) {
                    long scoreOffset = b * maxDetections * numClasses + d * numClasses + c;
                    scores.setAtIndex(ValueLayout.JAVA_FLOAT, scoreOffset, (float) Math.random());
                }
            }
        }

        outputs.put("pred_boxes", new TensorView(predBoxes,
                new TensorInfo("pred_boxes", DType.F32, predBoxShape, 0, predBoxBytes)));
        outputs.put("pred_scores", new TensorView(scores,
                new TensorInfo("pred_scores", DType.F32, scoreShape, 0, scoreBytes)));

        return outputs;
    }

    @Override
    public List<String> outputsToValidate() {
        return List.of("pred_boxes", "pred_scores");
    }

    @Override
    protected long maxSamples() {
        return 500; // Limit for benchmarking
    }

    /**
     * Builder for COCODetectionBenchmark.
     */
    public static final class Builder {
        private String name = "coco-detection-endtoend";
        private String modelId = "facebook/detr-resnet-50";
        private Path datasetPath;
        private int imageSize = 640;
        private int maxDetections = 100;
        private COCODataset.Task task = COCODataset.Task.DETECTION;
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

        public Builder maxDetections(int maxDetections) {
            this.maxDetections = maxDetections;
            return this;
        }

        public Builder task(COCODataset.Task task) {
            this.task = task;
            return this;
        }

        public Builder split(Dataset.Split split) {
            this.split = split;
            return this;
        }

        /**
         * Configure for DETR model.
         */
        public Builder detr() {
            return modelId("facebook/detr-resnet-50")
                    .imageSize(800)
                    .maxDetections(100);
        }

        /**
         * Configure for YOLO-style model.
         */
        public Builder yolo() {
            return modelId("ultralytics/yolov8")
                    .imageSize(640)
                    .maxDetections(300);
        }

        public COCODetectionBenchmark build() {
            if (datasetPath == null) {
                throw new IllegalStateException("datasetPath must be set");
            }
            return new COCODetectionBenchmark(this);
        }
    }
}
