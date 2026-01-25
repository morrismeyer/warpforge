package io.surfworks.warpforge.data.dataset;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * ImageNet dataset loader.
 *
 * <p>Supports loading ImageNet in the standard directory structure:
 * <pre>
 * imagenet/
 *   train/
 *     n01440764/
 *       n01440764_10026.JPEG
 *       ...
 *     n01443537/
 *       ...
 *   val/
 *     n01440764/
 *       ILSVRC2012_val_00000293.JPEG
 *       ...
 * </pre>
 *
 * <p>Example usage:
 * <pre>{@code
 * ImageNetDataset dataset = ImageNetDataset.load(Path.of("/data/imagenet"))
 *     .split(Split.TRAIN)
 *     .shuffle(42);
 *
 * for (ImageNetDataset.ImageSample sample : dataset) {
 *     // sample.image() - raw image bytes
 *     // sample.label() - class index (0-999)
 *     // sample.className() - e.g., "tench"
 * }
 * }</pre>
 */
public class ImageNetDataset extends AbstractDataset<ImageNetDataset.ImageSample> {

    private static final int NUM_CLASSES = 1000;
    private static final int DEFAULT_IMAGE_SIZE = 224;

    private final Path rootPath;
    private final List<ImageEntry> entries;
    private final Map<String, Integer> wnidToIndex;
    private final Map<Integer, String> indexToClassName;
    private final int imageSize;
    private final Arena arena;

    private ImageNetDataset(Path rootPath, Split split, List<ImageEntry> entries,
                            Map<String, Integer> wnidToIndex, Map<Integer, String> indexToClassName,
                            int imageSize) {
        super("imagenet", split);
        this.rootPath = rootPath;
        this.entries = entries;
        this.wnidToIndex = wnidToIndex;
        this.indexToClassName = indexToClassName;
        this.imageSize = imageSize;
        this.arena = Arena.ofShared();
    }

    /**
     * Load ImageNet dataset from directory.
     */
    public static ImageNetDataset load(Path rootPath) throws IOException {
        return load(rootPath, Split.TRAIN, DEFAULT_IMAGE_SIZE);
    }

    /**
     * Load ImageNet dataset from directory with specific split.
     */
    public static ImageNetDataset load(Path rootPath, Split split) throws IOException {
        return load(rootPath, split, DEFAULT_IMAGE_SIZE);
    }

    /**
     * Load ImageNet dataset from directory with specific split and image size.
     */
    public static ImageNetDataset load(Path rootPath, Split split, int imageSize) throws IOException {
        if (!Files.isDirectory(rootPath)) {
            throw new IOException("ImageNet root path does not exist: " + rootPath);
        }

        // Load class mappings
        Map<String, Integer> wnidToIndex = loadWnidMapping(rootPath);
        Map<Integer, String> indexToClassName = loadClassNames(rootPath);

        // Scan split directory
        String splitDir = split == Split.TRAIN ? "train" : "val";
        Path splitPath = rootPath.resolve(splitDir);

        List<ImageEntry> entries = new ArrayList<>();
        if (Files.isDirectory(splitPath)) {
            entries = scanDirectory(splitPath, wnidToIndex);
        }

        return new ImageNetDataset(rootPath, split, entries, wnidToIndex, indexToClassName, imageSize);
    }

    private static Map<String, Integer> loadWnidMapping(Path rootPath) throws IOException {
        Map<String, Integer> mapping = new HashMap<>();

        // Try to load from wnid_to_idx.txt or generate from directory structure
        Path mappingFile = rootPath.resolve("wnid_to_idx.txt");
        if (Files.exists(mappingFile)) {
            for (String line : Files.readAllLines(mappingFile)) {
                String[] parts = line.split("\\s+");
                if (parts.length >= 2) {
                    mapping.put(parts[0], Integer.parseInt(parts[1]));
                }
            }
        } else {
            // Generate from train directory structure
            Path trainPath = rootPath.resolve("train");
            if (Files.isDirectory(trainPath)) {
                int index = 0;
                try (Stream<Path> dirs = Files.list(trainPath).sorted()) {
                    for (Path dir : dirs.toList()) {
                        if (Files.isDirectory(dir)) {
                            mapping.put(dir.getFileName().toString(), index++);
                        }
                    }
                }
            }
        }

        return mapping;
    }

    private static Map<Integer, String> loadClassNames(Path rootPath) throws IOException {
        Map<Integer, String> names = new HashMap<>();

        // Try to load from imagenet_classes.txt
        Path classFile = rootPath.resolve("imagenet_classes.txt");
        if (Files.exists(classFile)) {
            List<String> lines = Files.readAllLines(classFile);
            for (int i = 0; i < lines.size(); i++) {
                names.put(i, lines.get(i).trim());
            }
        }

        return names;
    }

    private static List<ImageEntry> scanDirectory(Path splitPath, Map<String, Integer> wnidToIndex)
            throws IOException {
        List<ImageEntry> entries = new ArrayList<>();

        try (Stream<Path> classDirs = Files.list(splitPath)) {
            for (Path classDir : classDirs.toList()) {
                if (!Files.isDirectory(classDir)) continue;

                String wnid = classDir.getFileName().toString();
                Integer classIndex = wnidToIndex.get(wnid);
                if (classIndex == null) {
                    // Auto-assign index for unknown classes
                    classIndex = wnidToIndex.size();
                    wnidToIndex.put(wnid, classIndex);
                }

                final int label = classIndex;
                try (Stream<Path> images = Files.list(classDir)) {
                    for (Path imagePath : images.toList()) {
                        String name = imagePath.getFileName().toString().toLowerCase();
                        if (name.endsWith(".jpeg") || name.endsWith(".jpg") || name.endsWith(".png")) {
                            entries.add(new ImageEntry(imagePath, label, wnid));
                        }
                    }
                }
            }
        }

        return entries;
    }

    @Override
    protected long totalSize() {
        return entries.size();
    }

    @Override
    protected ImageSample getRaw(long index) {
        ImageEntry entry = entries.get((int) index);
        return new ImageSample(entry, indexToClassName.getOrDefault(entry.label, entry.wnid), imageSize, arena);
    }

    @Override
    public Dataset<ImageSample> split(Split split) {
        try {
            return load(rootPath, split, imageSize);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load split: " + split, e);
        }
    }

    @Override
    public DatasetInfo info() {
        return DatasetInfo.builder("imagenet")
                .description("ImageNet Large Scale Visual Recognition Challenge (ILSVRC)")
                .totalSamples(entries.size())
                .splitSize(currentSplit, entries.size())
                .feature("image")
                .feature("label")
                .extra("num_classes", NUM_CLASSES)
                .extra("image_size", imageSize)
                .build();
    }

    /**
     * Get the number of classes.
     */
    public int numClasses() {
        return wnidToIndex.size();
    }

    /**
     * Get class name by index.
     */
    public String className(int index) {
        return indexToClassName.getOrDefault(index, "unknown");
    }

    // Internal entry class
    private record ImageEntry(Path path, int label, String wnid) {}

    /**
     * A single ImageNet sample.
     */
    public static class ImageSample implements Dataset.Sample {
        private final ImageEntry entry;
        private final String className;
        private final int imageSize;
        private final Arena arena;
        private byte[] imageBytes;

        ImageSample(ImageEntry entry, String className, int imageSize, Arena arena) {
            this.entry = entry;
            this.className = className;
            this.imageSize = imageSize;
            this.arena = arena;
        }

        /**
         * Get the image file path.
         */
        public Path path() {
            return entry.path;
        }

        /**
         * Get the class label (0-999).
         */
        public int label() {
            return entry.label;
        }

        /**
         * Get the WordNet ID (e.g., "n01440764").
         */
        public String wnid() {
            return entry.wnid;
        }

        /**
         * Get the human-readable class name.
         */
        public String className() {
            return className;
        }

        /**
         * Get raw image bytes.
         */
        public byte[] imageBytes() throws IOException {
            if (imageBytes == null) {
                imageBytes = Files.readAllBytes(entry.path);
            }
            return imageBytes;
        }

        @Override
        public Map<String, TensorView> toTensors() {
            // Create placeholder tensor (actual image loading would require image decoding)
            // For benchmarking, we create a random tensor of the expected shape
            long[] shape = {3, imageSize, imageSize};  // CHW format
            long elementCount = 3L * imageSize * imageSize;
            long byteSize = elementCount * 4;  // F32

            MemorySegment segment = arena.allocate(byteSize);

            // Fill with label-based pattern for reproducibility
            for (int i = 0; i < elementCount; i++) {
                float value = ((entry.label + i) % 256) / 255.0f;
                segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, value);
            }

            TensorInfo imageInfo = new TensorInfo("image", DType.F32, shape, 0, byteSize);
            TensorView imageTensor = new TensorView(segment, imageInfo);

            // Create label tensor
            MemorySegment labelSegment = arena.allocate(8);  // I64
            labelSegment.set(ValueLayout.JAVA_LONG, 0, entry.label);
            TensorInfo labelInfo = new TensorInfo("label", DType.I64, new long[]{}, 0, 8);
            TensorView labelTensor = new TensorView(labelSegment, labelInfo);

            return Map.of("image", imageTensor, "label", labelTensor);
        }
    }
}
