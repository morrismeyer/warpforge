package io.surfworks.warpforge.data.golden;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * File-based implementation of GoldenStore.
 *
 * <p>Stores golden outputs as pairs of files:
 * <ul>
 *   <li>{id}.json - Metadata (shape, dtype, pytorch version, etc.)</li>
 *   <li>{id}.bin - Raw tensor data (little-endian)</li>
 * </ul>
 *
 * <p>IDs with slashes are converted to directory structure:
 * "bert-base/pooler_output" → "bert-base/pooler_output.json" + "bert-base/pooler_output.bin"
 */
public final class FileGoldenStore implements GoldenStore {

    private static final Gson GSON = new GsonBuilder()
            .setPrettyPrinting()
            .create();

    private final Path directory;
    private final Arena arena;

    /**
     * Create a new file-based golden store.
     *
     * @param directory Directory to store golden outputs
     */
    public FileGoldenStore(Path directory) {
        this.directory = Objects.requireNonNull(directory);
        this.arena = Arena.ofShared();
    }

    @Override
    public void save(GoldenOutput output) throws IOException {
        Path basePath = resolvePath(output.id());
        Path metaPath = Path.of(basePath + ".json");
        Path dataPath = Path.of(basePath + ".bin");

        // Create parent directories
        Files.createDirectories(metaPath.getParent());

        // Write metadata
        JsonObject meta = new JsonObject();
        meta.addProperty("id", output.id());
        meta.addProperty("dtype", output.tensorInfo().dtype().name());
        meta.add("shape", GSON.toJsonTree(output.tensorInfo().shape()));
        meta.addProperty("created_at", output.createdAt().toString());

        JsonObject metadata = new JsonObject();
        for (Map.Entry<String, String> entry : output.metadata().entrySet()) {
            metadata.addProperty(entry.getKey(), entry.getValue());
        }
        meta.add("metadata", metadata);

        Files.writeString(metaPath, GSON.toJson(meta), StandardCharsets.UTF_8);

        // Write tensor data
        try (FileChannel channel = FileChannel.open(dataPath,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            channel.write(output.data().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN));
        }
    }

    @Override
    public Optional<GoldenOutput> load(String id) throws IOException {
        Path basePath = resolvePath(id);
        Path metaPath = Path.of(basePath + ".json");
        Path dataPath = Path.of(basePath + ".bin");

        if (!Files.exists(metaPath) || !Files.exists(dataPath)) {
            return Optional.empty();
        }

        // Read metadata
        String metaJson = Files.readString(metaPath, StandardCharsets.UTF_8);
        JsonObject meta = GSON.fromJson(metaJson, JsonObject.class);

        String storedId = meta.get("id").getAsString();
        DType dtype = DType.valueOf(meta.get("dtype").getAsString());
        long[] shape = GSON.fromJson(meta.get("shape"), long[].class);
        Instant createdAt = Instant.parse(meta.get("created_at").getAsString());

        Map<String, String> metadata = new HashMap<>();
        JsonObject metaObj = meta.getAsJsonObject("metadata");
        if (metaObj != null) {
            for (String key : metaObj.keySet()) {
                metadata.put(key, metaObj.get(key).getAsString());
            }
        }

        // Memory-map tensor data
        long dataSize = Files.size(dataPath);
        MemorySegment data;
        try (FileChannel channel = FileChannel.open(dataPath, StandardOpenOption.READ)) {
            data = channel.map(FileChannel.MapMode.READ_ONLY, 0, dataSize, arena);
        }

        TensorInfo tensorInfo = new TensorInfo(storedId, dtype, shape, 0, dataSize);

        return Optional.of(new GoldenOutput(storedId, tensorInfo, data, metadata, createdAt));
    }

    @Override
    public boolean exists(String id) {
        Path basePath = resolvePath(id);
        return Files.exists(Path.of(basePath + ".json")) &&
                Files.exists(Path.of(basePath + ".bin"));
    }

    @Override
    public boolean delete(String id) throws IOException {
        Path basePath = resolvePath(id);
        Path metaPath = Path.of(basePath + ".json");
        Path dataPath = Path.of(basePath + ".bin");

        boolean deleted = false;
        if (Files.exists(metaPath)) {
            Files.delete(metaPath);
            deleted = true;
        }
        if (Files.exists(dataPath)) {
            Files.delete(dataPath);
            deleted = true;
        }

        return deleted;
    }

    @Override
    public List<String> list() throws IOException {
        return listByPrefix("");
    }

    @Override
    public List<String> listByPrefix(String prefix) throws IOException {
        List<String> ids = new ArrayList<>();

        if (!Files.exists(directory)) {
            return ids;
        }

        Path prefixPath = prefix.isEmpty() ? directory : directory.resolve(prefix);
        Path searchDir = Files.isDirectory(prefixPath) ? prefixPath : prefixPath.getParent();

        if (searchDir == null || !Files.exists(searchDir)) {
            return ids;
        }

        try (Stream<Path> stream = Files.walk(searchDir)) {
            stream.filter(p -> p.toString().endsWith(".json"))
                    .forEach(p -> {
                        String relativePath = directory.relativize(p).toString();
                        // Remove .json extension
                        String id = relativePath.substring(0, relativePath.length() - 5);
                        // Convert path separators to forward slashes
                        id = id.replace(java.io.File.separatorChar, '/');
                        if (prefix.isEmpty() || id.startsWith(prefix)) {
                            ids.add(id);
                        }
                    });
        }

        return ids;
    }

    /**
     * Get the storage directory.
     */
    public Path directory() {
        return directory;
    }

    /**
     * Resolve an ID to a file path (without extension).
     */
    private Path resolvePath(String id) {
        // Replace forward slashes with system separator for file path
        String pathStr = id.replace('/', java.io.File.separatorChar);
        return directory.resolve(pathStr);
    }
}
