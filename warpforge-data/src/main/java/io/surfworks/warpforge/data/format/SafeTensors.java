package io.surfworks.warpforge.data.format;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Reader for the SafeTensors format.
 *
 * <p>SafeTensors is a simple, safe format for storing tensors. The format is:
 * <pre>
 * [8 bytes: header length (little-endian u64)]
 * [header_length bytes: JSON header]
 * [remaining bytes: tensor data]
 * </pre>
 *
 * <p>The JSON header contains metadata for each tensor including dtype, shape,
 * and byte offsets into the data section.
 *
 * <p>This class uses memory-mapped I/O for zero-copy access to tensor data.
 *
 * @see <a href="https://huggingface.co/docs/safetensors">SafeTensors documentation</a>
 */
public final class SafeTensors implements AutoCloseable {

    private static final Gson GSON = new Gson();

    private final Path path;
    private final Arena arena;
    private final MemorySegment mappedFile;
    private final Map<String, TensorInfo> tensors;
    private final long dataOffset;
    private final Map<String, String> metadata;

    private SafeTensors(Path path, Arena arena, MemorySegment mappedFile,
                        Map<String, TensorInfo> tensors, long dataOffset,
                        Map<String, String> metadata) {
        this.path = path;
        this.arena = arena;
        this.mappedFile = mappedFile;
        this.tensors = tensors;
        this.dataOffset = dataOffset;
        this.metadata = metadata;
    }

    /**
     * Open a SafeTensors file for reading.
     *
     * @param path Path to the .safetensors file
     * @return A SafeTensors instance for accessing the tensors
     * @throws IOException if the file cannot be read or is invalid
     */
    public static SafeTensors open(Path path) throws IOException {
        Arena arena = Arena.ofShared();
        try {
            return open(path, arena);
        } catch (Exception e) {
            arena.close();
            throw e;
        }
    }

    /**
     * Open a SafeTensors file with a provided Arena.
     *
     * @param path  Path to the .safetensors file
     * @param arena Arena for memory management
     * @return A SafeTensors instance for accessing the tensors
     * @throws IOException if the file cannot be read or is invalid
     */
    public static SafeTensors open(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize < 8) {
                throw new IOException("File too small to be a SafeTensors file: " + fileSize + " bytes");
            }

            MemorySegment mapped = channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    0,
                    fileSize,
                    arena
            );

            // Read header length (little-endian u64)
            long headerLength = mapped.get(ValueLayout.JAVA_LONG_UNALIGNED, 0);
            if (headerLength < 0 || headerLength > fileSize - 8) {
                throw new IOException("Invalid header length: " + headerLength);
            }

            // Read JSON header
            long headerStart = 8;
            long dataOffset = 8 + headerLength;

            byte[] headerBytes = new byte[(int) headerLength];
            MemorySegment.copy(mapped, ValueLayout.JAVA_BYTE, headerStart, headerBytes, 0, (int) headerLength);
            String headerJson = new String(headerBytes, StandardCharsets.UTF_8);

            // Parse header
            JsonObject header = GSON.fromJson(headerJson, JsonObject.class);
            Map<String, TensorInfo> tensors = new LinkedHashMap<>();
            Map<String, String> metadata = new LinkedHashMap<>();

            for (Map.Entry<String, JsonElement> entry : header.entrySet()) {
                String key = entry.getKey();
                JsonElement value = entry.getValue();

                if (key.equals("__metadata__")) {
                    // Parse metadata section
                    if (value.isJsonObject()) {
                        for (Map.Entry<String, JsonElement> metaEntry : value.getAsJsonObject().entrySet()) {
                            if (metaEntry.getValue().isJsonPrimitive()) {
                                metadata.put(metaEntry.getKey(), metaEntry.getValue().getAsString());
                            }
                        }
                    }
                } else {
                    // Parse tensor info
                    JsonObject tensorObj = value.getAsJsonObject();
                    TensorInfo info = parseTensorInfo(key, tensorObj);
                    tensors.put(key, info);
                }
            }

            return new SafeTensors(path, arena, mapped, tensors, dataOffset, metadata);
        }
    }

    private static TensorInfo parseTensorInfo(String name, JsonObject obj) {
        String dtypeStr = obj.get("dtype").getAsString();
        DType dtype = DType.fromSafeTensors(dtypeStr);

        // Shape is an array of ints
        var shapeArray = obj.getAsJsonArray("shape");
        long[] shape = new long[shapeArray.size()];
        for (int i = 0; i < shapeArray.size(); i++) {
            shape[i] = shapeArray.get(i).getAsLong();
        }

        // Data offsets
        var offsetsArray = obj.getAsJsonArray("data_offsets");
        long startOffset = offsetsArray.get(0).getAsLong();
        long endOffset = offsetsArray.get(1).getAsLong();
        long size = endOffset - startOffset;

        return new TensorInfo(name, dtype, shape, startOffset, size);
    }

    /**
     * Get the path to the underlying file.
     */
    public Path path() {
        return path;
    }

    /**
     * Get names of all tensors in the file.
     */
    public Set<String> tensorNames() {
        return Collections.unmodifiableSet(tensors.keySet());
    }

    /**
     * Check if a tensor with the given name exists.
     */
    public boolean hasTensor(String name) {
        return tensors.containsKey(name);
    }

    /**
     * Get metadata about a tensor without loading its data.
     */
    public TensorInfo tensorInfo(String name) {
        TensorInfo info = tensors.get(name);
        if (info == null) {
            throw new IllegalArgumentException("No tensor named: " + name);
        }
        return info;
    }

    /**
     * Get a view of a tensor's data.
     * This is zero-copy - the view references the memory-mapped file directly.
     */
    public TensorView tensor(String name) {
        TensorInfo info = tensorInfo(name);
        MemorySegment tensorData = mappedFile.asSlice(dataOffset + info.offset(), info.size());
        return new TensorView(tensorData, info);
    }

    /**
     * Get all tensor infos.
     */
    public Map<String, TensorInfo> allTensorInfos() {
        return Collections.unmodifiableMap(tensors);
    }

    /**
     * Get file metadata (from __metadata__ section).
     */
    public Map<String, String> metadata() {
        return Collections.unmodifiableMap(metadata);
    }

    /**
     * Total number of tensors in the file.
     */
    public int tensorCount() {
        return tensors.size();
    }

    /**
     * Total size of the file in bytes.
     */
    public long fileSize() {
        return mappedFile.byteSize();
    }

    @Override
    public void close() {
        arena.close();
    }

    @Override
    public String toString() {
        return String.format("SafeTensors[%s: %d tensors, %d bytes]",
                path.getFileName(), tensors.size(), mappedFile.byteSize());
    }
}
