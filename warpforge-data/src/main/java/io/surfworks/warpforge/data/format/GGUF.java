package io.surfworks.warpforge.data.format;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

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
 * Reader for the GGUF (GGML Universal Format) file format.
 *
 * <p>GGUF is the format used by llama.cpp and related projects for storing
 * quantized language models. It supports various quantization types including
 * Q4_0, Q4_K_M, Q8_0, and standard floating point formats.
 *
 * <p>File structure:
 * <pre>
 * [4 bytes: magic "GGUF"]
 * [4 bytes: version (uint32)]
 * [8 bytes: tensor count (uint64)]
 * [8 bytes: metadata kv count (uint64)]
 * [metadata key-value pairs...]
 * [tensor infos...]
 * [padding to alignment]
 * [tensor data...]
 * </pre>
 *
 * <p>This class uses memory-mapped I/O for zero-copy access to tensor data.
 *
 * @see <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF specification</a>
 */
public final class GGUF implements AutoCloseable {

    /** GGUF magic number: "GGUF" in little-endian */
    private static final int MAGIC = 0x46554747; // "GGUF" as little-endian int

    /** Default alignment for tensor data */
    private static final int DEFAULT_ALIGNMENT = 32;

    private final Path path;
    private final Arena arena;
    private final MemorySegment mappedFile;
    private final int version;
    private final Map<String, TensorInfo> tensors;
    private final Map<String, Object> metadata;
    private final long dataOffset;

    private GGUF(Path path, Arena arena, MemorySegment mappedFile, int version,
                 Map<String, TensorInfo> tensors, Map<String, Object> metadata,
                 long dataOffset) {
        this.path = path;
        this.arena = arena;
        this.mappedFile = mappedFile;
        this.version = version;
        this.tensors = tensors;
        this.metadata = metadata;
        this.dataOffset = dataOffset;
    }

    /**
     * Open a GGUF file for reading.
     *
     * @param path Path to the .gguf file
     * @return A GGUF instance for accessing the tensors
     * @throws IOException if the file cannot be read or is invalid
     */
    public static GGUF open(Path path) throws IOException {
        Arena arena = Arena.ofShared();
        try {
            return open(path, arena);
        } catch (Exception e) {
            arena.close();
            throw e;
        }
    }

    /**
     * Open a GGUF file with a provided Arena.
     *
     * @param path  Path to the .gguf file
     * @param arena Arena for memory management
     * @return A GGUF instance for accessing the tensors
     * @throws IOException if the file cannot be read or is invalid
     */
    public static GGUF open(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();
            if (fileSize < 24) {
                throw new IOException("File too small to be a GGUF file: " + fileSize + " bytes");
            }

            MemorySegment mapped = channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    0,
                    fileSize,
                    arena
            );

            GGUFReader reader = new GGUFReader(mapped);

            // Read and validate magic
            int magic = reader.readInt();
            if (magic != MAGIC) {
                throw new IOException(String.format(
                        "Invalid GGUF magic: expected 0x%08X, got 0x%08X", MAGIC, magic));
            }

            // Read version
            int version = reader.readInt();
            if (version < 1 || version > 3) {
                throw new IOException("Unsupported GGUF version: " + version);
            }

            // Read counts
            long tensorCount = reader.readLong();
            long metadataKvCount = reader.readLong();

            // Parse metadata
            Map<String, Object> metadata = new LinkedHashMap<>();
            int alignment = DEFAULT_ALIGNMENT;

            for (long i = 0; i < metadataKvCount; i++) {
                String key = reader.readString(version);
                Object value = reader.readValue(version);
                metadata.put(key, value);

                // Check for custom alignment
                if (key.equals("general.alignment") && value instanceof Number) {
                    alignment = ((Number) value).intValue();
                }
            }

            // Parse tensor infos
            Map<String, TensorInfo> tensors = new LinkedHashMap<>();
            long[] tensorOffsets = new long[(int) tensorCount];

            for (int i = 0; i < tensorCount; i++) {
                String name = reader.readString(version);
                int nDims = reader.readInt();

                long[] shape = new long[nDims];
                for (int d = 0; d < nDims; d++) {
                    shape[d] = reader.readLong();
                }

                int ggmlType = reader.readInt();
                DType dtype = ggmlTypeToDType(ggmlType);

                long offset = reader.readLong();
                tensorOffsets[i] = offset;

                // Calculate size from shape and dtype
                long size = calculateTensorSize(shape, dtype, ggmlType);

                TensorInfo info = new TensorInfo(name, dtype, shape, offset, size);
                tensors.put(name, info);
            }

            // Calculate data offset (aligned)
            long headerEnd = reader.position();
            long dataOffset = alignUp(headerEnd, alignment);

            return new GGUF(path, arena, mapped, version, tensors, metadata, dataOffset);
        }
    }

    /**
     * Get the GGUF format version.
     */
    public int version() {
        return version;
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
     * Get file metadata.
     */
    public Map<String, Object> metadata() {
        return Collections.unmodifiableMap(metadata);
    }

    /**
     * Get a string metadata value.
     */
    public String metadataString(String key) {
        Object value = metadata.get(key);
        return value != null ? value.toString() : null;
    }

    /**
     * Get the model architecture (from general.architecture).
     */
    public String architecture() {
        return metadataString("general.architecture");
    }

    /**
     * Get the model name (from general.name).
     */
    public String modelName() {
        return metadataString("general.name");
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
        String arch = architecture();
        String archStr = arch != null ? arch + ", " : "";
        return String.format("GGUF[%s: %s%d tensors, v%d, %d bytes]",
                path.getFileName(), archStr, tensors.size(), version, mappedFile.byteSize());
    }

    // =========================================================================
    // GGML Type Mapping
    // =========================================================================

    private static DType ggmlTypeToDType(int ggmlType) {
        return switch (ggmlType) {
            case 0 -> DType.F32;           // GGML_TYPE_F32
            case 1 -> DType.F16;           // GGML_TYPE_F16
            case 2 -> DType.Q4_0;          // GGML_TYPE_Q4_0
            case 3 -> DType.Q4_1;          // GGML_TYPE_Q4_1
            case 6 -> DType.Q5_0;          // GGML_TYPE_Q5_0
            case 7 -> DType.Q5_1;          // GGML_TYPE_Q5_1
            case 8 -> DType.Q8_0;          // GGML_TYPE_Q8_0
            case 9 -> DType.Q8_K;          // GGML_TYPE_Q8_1 (mapped to Q8_K for our purposes)
            case 14 -> DType.Q4_K_M;       // GGML_TYPE_Q4_K (mapped to Q4_K_M)
            case 16 -> DType.Q5_K_M;       // GGML_TYPE_Q5_K (mapped to Q5_K_M)
            case 18 -> DType.BF16;         // GGML_TYPE_BF16
            case 26 -> DType.F8_E5M2;      // GGML_TYPE_F8_E5M2 (if supported)
            case 27 -> DType.F8_E4M3;      // GGML_TYPE_F8_E4M3 (if supported)
            case 28 -> DType.I8;           // GGML_TYPE_I8
            case 29 -> DType.I16;          // GGML_TYPE_I16
            case 30 -> DType.I32;          // GGML_TYPE_I32
            case 31 -> DType.I64;          // GGML_TYPE_I64
            case 32 -> DType.F64;          // GGML_TYPE_F64
            default -> throw new IllegalArgumentException("Unknown GGML type: " + ggmlType);
        };
    }

    private static long calculateTensorSize(long[] shape, DType dtype, int ggmlType) {
        long elements = 1;
        for (long dim : shape) {
            elements *= dim;
        }

        // For block-quantized types, use block size calculations
        if (dtype.isBlockQuantized()) {
            return calculateQuantizedSize(elements, ggmlType);
        }

        return dtype.packedByteSize(elements);
    }

    private static long calculateQuantizedSize(long elements, int ggmlType) {
        // GGML block sizes and per-block byte sizes
        return switch (ggmlType) {
            case 2 -> // Q4_0: 32 elements per block, 18 bytes per block (16 + 2 for scale)
                    ((elements + 31) / 32) * 18;
            case 3 -> // Q4_1: 32 elements per block, 20 bytes per block
                    ((elements + 31) / 32) * 20;
            case 6 -> // Q5_0: 32 elements per block, 22 bytes per block
                    ((elements + 31) / 32) * 22;
            case 7 -> // Q5_1: 32 elements per block, 24 bytes per block
                    ((elements + 31) / 32) * 24;
            case 8 -> // Q8_0: 32 elements per block, 34 bytes per block
                    ((elements + 31) / 32) * 34;
            case 14 -> // Q4_K: 256 elements per block (super-block)
                    ((elements + 255) / 256) * 144;
            case 16 -> // Q5_K: 256 elements per block
                    ((elements + 255) / 256) * 176;
            default -> elements; // Fallback
        };
    }

    private static long alignUp(long value, int alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    // =========================================================================
    // GGUF Reader Helper
    // =========================================================================

    private static class GGUFReader {
        private final MemorySegment segment;
        private long pos;

        GGUFReader(MemorySegment segment) {
            this.segment = segment;
            this.pos = 0;
        }

        long position() {
            return pos;
        }

        int readInt() {
            int value = segment.get(ValueLayout.JAVA_INT_UNALIGNED, pos);
            pos += 4;
            return value;
        }

        long readLong() {
            long value = segment.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);
            pos += 8;
            return value;
        }

        float readFloat() {
            float value = segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, pos);
            pos += 4;
            return value;
        }

        double readDouble() {
            double value = segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, pos);
            pos += 8;
            return value;
        }

        byte readByte() {
            byte value = segment.get(ValueLayout.JAVA_BYTE, pos);
            pos += 1;
            return value;
        }

        short readShort() {
            short value = segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, pos);
            pos += 2;
            return value;
        }

        String readString(int version) {
            long length = version >= 2 ? readLong() : readInt();
            if (length > Integer.MAX_VALUE || length < 0) {
                throw new IllegalStateException("String too long: " + length);
            }
            byte[] bytes = new byte[(int) length];
            MemorySegment.copy(segment, ValueLayout.JAVA_BYTE, pos, bytes, 0, (int) length);
            pos += length;
            return new String(bytes, StandardCharsets.UTF_8);
        }

        Object readValue(int version) {
            int type = readInt();
            return switch (type) {
                case 0 -> (int) readByte();              // GGUF_TYPE_UINT8
                case 1 -> readByte();                     // GGUF_TYPE_INT8
                case 2 -> readShort() & 0xFFFF;           // GGUF_TYPE_UINT16
                case 3 -> readShort();                    // GGUF_TYPE_INT16
                case 4 -> readInt() & 0xFFFFFFFFL;        // GGUF_TYPE_UINT32
                case 5 -> readInt();                      // GGUF_TYPE_INT32
                case 6 -> readFloat();                    // GGUF_TYPE_FLOAT32
                case 7 -> readByte() != 0;                // GGUF_TYPE_BOOL
                case 8 -> readString(version);            // GGUF_TYPE_STRING
                case 9 -> readArray(version);             // GGUF_TYPE_ARRAY
                case 10 -> readLong();                    // GGUF_TYPE_UINT64
                case 11 -> readLong();                    // GGUF_TYPE_INT64
                case 12 -> readDouble();                  // GGUF_TYPE_FLOAT64
                default -> throw new IllegalArgumentException("Unknown GGUF value type: " + type);
            };
        }

        Object[] readArray(int version) {
            int elementType = readInt();
            long length = version >= 2 ? readLong() : readInt();
            if (length > Integer.MAX_VALUE || length < 0) {
                throw new IllegalStateException("Array too long: " + length);
            }

            Object[] array = new Object[(int) length];
            for (int i = 0; i < length; i++) {
                array[i] = readValueOfType(elementType, version);
            }
            return array;
        }

        Object readValueOfType(int type, int version) {
            return switch (type) {
                case 0 -> (int) readByte();
                case 1 -> readByte();
                case 2 -> readShort() & 0xFFFF;
                case 3 -> readShort();
                case 4 -> readInt() & 0xFFFFFFFFL;
                case 5 -> readInt();
                case 6 -> readFloat();
                case 7 -> readByte() != 0;
                case 8 -> readString(version);
                case 10 -> readLong();
                case 11 -> readLong();
                case 12 -> readDouble();
                default -> throw new IllegalArgumentException("Unknown GGUF value type: " + type);
            };
        }
    }
}
