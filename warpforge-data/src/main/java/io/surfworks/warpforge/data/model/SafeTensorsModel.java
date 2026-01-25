package io.surfworks.warpforge.data.model;

import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.format.SafeTensors;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * ModelSource implementation for SafeTensors format.
 *
 * <p>Supports both single .safetensors files and sharded models
 * (multiple files like model-00001-of-00004.safetensors).
 */
public final class SafeTensorsModel implements ModelSource {

    private static final Gson GSON = new Gson();

    private final String id;
    private final List<SafeTensors> shards;
    private final Map<String, TensorInfo> tensorInfos;
    private final Map<String, SafeTensors> tensorToShard;
    private final Map<String, Object> metadata;

    private SafeTensorsModel(String id, List<SafeTensors> shards,
                             Map<String, TensorInfo> tensorInfos,
                             Map<String, SafeTensors> tensorToShard,
                             Map<String, Object> metadata) {
        this.id = id;
        this.shards = shards;
        this.tensorInfos = tensorInfos;
        this.tensorToShard = tensorToShard;
        this.metadata = metadata;
    }

    /**
     * Load a model from a single SafeTensors file.
     */
    public static SafeTensorsModel fromFile(String id, Path path) throws IOException {
        SafeTensors st = SafeTensors.open(path);
        Map<String, TensorInfo> infos = new LinkedHashMap<>(st.allTensorInfos());
        Map<String, SafeTensors> tensorToShard = new LinkedHashMap<>();
        for (String name : st.tensorNames()) {
            tensorToShard.put(name, st);
        }

        Map<String, Object> metadata = new LinkedHashMap<>(st.metadata());
        return new SafeTensorsModel(id, List.of(st), infos, tensorToShard, metadata);
    }

    /**
     * Load a model from a directory containing SafeTensors files.
     *
     * <p>Handles sharded models (model-00001-of-00004.safetensors, etc.)
     * and loads config.json if present.
     */
    public static SafeTensorsModel fromDirectory(String id, Path dir) throws IOException {
        if (!Files.isDirectory(dir)) {
            throw new IOException("Not a directory: " + dir);
        }

        // Find all .safetensors files
        List<Path> safetensorFiles;
        try (Stream<Path> files = Files.list(dir)) {
            safetensorFiles = files
                    .filter(p -> p.toString().endsWith(".safetensors"))
                    .sorted()
                    .collect(Collectors.toList());
        }

        if (safetensorFiles.isEmpty()) {
            throw new IOException("No .safetensors files found in: " + dir);
        }

        // Load all shards
        List<SafeTensors> shards = new ArrayList<>();
        Map<String, TensorInfo> tensorInfos = new LinkedHashMap<>();
        Map<String, SafeTensors> tensorToShard = new LinkedHashMap<>();

        for (Path file : safetensorFiles) {
            SafeTensors st = SafeTensors.open(file);
            shards.add(st);

            for (String name : st.tensorNames()) {
                tensorInfos.put(name, st.tensorInfo(name));
                tensorToShard.put(name, st);
            }
        }

        // Load config.json if present
        Map<String, Object> metadata = new LinkedHashMap<>();
        Path configPath = dir.resolve("config.json");
        if (Files.exists(configPath)) {
            String configJson = Files.readString(configPath);
            JsonObject config = GSON.fromJson(configJson, JsonObject.class);
            for (var entry : config.entrySet()) {
                metadata.put(entry.getKey(), entry.getValue());
            }
        }

        // Merge metadata from first shard
        if (!shards.isEmpty()) {
            metadata.putAll(shards.get(0).metadata());
        }

        return new SafeTensorsModel(id, shards, tensorInfos, tensorToShard, metadata);
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public Set<String> tensorNames() {
        return Collections.unmodifiableSet(tensorInfos.keySet());
    }

    @Override
    public boolean hasTensor(String name) {
        return tensorInfos.containsKey(name);
    }

    @Override
    public TensorInfo tensorInfo(String name) {
        TensorInfo info = tensorInfos.get(name);
        if (info == null) {
            throw new IllegalArgumentException("No tensor named: " + name);
        }
        return info;
    }

    @Override
    public TensorView tensor(String name) {
        SafeTensors shard = tensorToShard.get(name);
        if (shard == null) {
            throw new IllegalArgumentException("No tensor named: " + name);
        }
        return shard.tensor(name);
    }

    @Override
    public Map<String, TensorInfo> allTensorInfos() {
        return Collections.unmodifiableMap(tensorInfos);
    }

    @Override
    public Map<String, Object> metadata() {
        return Collections.unmodifiableMap(metadata);
    }

    @Override
    public void close() {
        for (SafeTensors shard : shards) {
            shard.close();
        }
    }

    @Override
    public String toString() {
        return String.format("SafeTensorsModel[%s: %d tensors, %d shards, %d params]",
                id, tensorInfos.size(), shards.size(), parameterCount());
    }
}
