package io.surfworks.warpforge.data.format;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ONNX (Open Neural Network Exchange) format reader.
 *
 * <p>ONNX is a widely-used format for ML model interchange. This implementation
 * provides read access to ONNX model weights and graph structure.
 *
 * <p>Supports:
 * <ul>
 *   <li>ONNX opset versions 7-21</li>
 *   <li>External data (weights stored separately)</li>
 *   <li>All standard tensor data types</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * try (ONNX.Model model = ONNX.load(Path.of("model.onnx"))) {
 *     // Get model metadata
 *     System.out.println("IR version: " + model.irVersion());
 *     System.out.println("Producer: " + model.producer());
 *
 *     // Access initializers (weights)
 *     for (String name : model.initializerNames()) {
 *         TensorView tensor = model.initializer(name);
 *         System.out.println(name + ": " + Arrays.toString(tensor.shape()));
 *     }
 *
 *     // Get graph inputs/outputs
 *     model.inputs().forEach(System.out::println);
 *     model.outputs().forEach(System.out::println);
 * }
 * }</pre>
 */
public final class ONNX {

    private ONNX() {}

    // ONNX data type constants (from onnx.proto)
    public static final int UNDEFINED = 0;
    public static final int FLOAT = 1;
    public static final int UINT8 = 2;
    public static final int INT8 = 3;
    public static final int UINT16 = 4;
    public static final int INT16 = 5;
    public static final int INT32 = 6;
    public static final int INT64 = 7;
    public static final int STRING = 8;
    public static final int BOOL = 9;
    public static final int FLOAT16 = 10;
    public static final int DOUBLE = 11;
    public static final int UINT32 = 12;
    public static final int UINT64 = 13;
    public static final int COMPLEX64 = 14;
    public static final int COMPLEX128 = 15;
    public static final int BFLOAT16 = 16;

    /**
     * Load an ONNX model from file.
     */
    public static Model load(Path path) throws IOException {
        return new Model(path);
    }

    /**
     * Convert ONNX data type to DType.
     */
    public static DType toDType(int onnxType) {
        return switch (onnxType) {
            case FLOAT -> DType.F32;
            case DOUBLE -> DType.F64;
            case FLOAT16 -> DType.F16;
            case BFLOAT16 -> DType.BF16;
            case INT8 -> DType.I8;
            case INT16 -> DType.I16;
            case INT32 -> DType.I32;
            case INT64 -> DType.I64;
            case UINT8 -> DType.U8;
            case UINT16 -> DType.U16;
            case UINT32 -> DType.U32;
            case BOOL -> DType.I8;
            default -> throw new IllegalArgumentException("Unsupported ONNX type: " + onnxType);
        };
    }

    /**
     * Represents an ONNX model.
     */
    public static final class Model implements AutoCloseable {
        private final Path path;
        private final Arena arena;
        private final MemorySegment data;
        private final long irVersion;
        private final String producerName;
        private final String producerVersion;
        private final String domain;
        private final long modelVersion;
        private final String docString;
        private final Graph graph;
        private final List<OperatorSetId> opsetImports;
        private final Map<String, String> metadataProps;

        Model(Path path) throws IOException {
            this.path = path;
            this.arena = Arena.ofConfined();
            this.opsetImports = new ArrayList<>();
            this.metadataProps = new HashMap<>();

            try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
                long fileSize = channel.size();
                this.data = arena.allocate(fileSize);

                ByteBuffer buffer = data.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
                channel.read(buffer);
                buffer.flip();

                // Parse protobuf
                ProtobufParser parser = new ProtobufParser(data);

                long irVer = 0;
                String prodName = "";
                String prodVer = "";
                String dom = "";
                long modVer = 0;
                String doc = "";
                Graph g = null;

                while (parser.hasMore()) {
                    int tag = parser.readTag();
                    int fieldNumber = tag >>> 3;
                    int wireType = tag & 0x7;

                    switch (fieldNumber) {
                        case 1 -> irVer = parser.readVarint();
                        case 2 -> prodName = parser.readString();
                        case 3 -> prodVer = parser.readString();
                        case 4 -> dom = parser.readString();
                        case 5 -> modVer = parser.readVarint();
                        case 6 -> doc = parser.readString();
                        case 7 -> g = parseGraph(parser.readBytes());
                        case 8 -> opsetImports.add(parseOpsetId(parser.readBytes()));
                        case 14 -> parseMetadataProp(parser.readBytes());
                        default -> parser.skip(wireType);
                    }
                }

                this.irVersion = irVer;
                this.producerName = prodName;
                this.producerVersion = prodVer;
                this.domain = dom;
                this.modelVersion = modVer;
                this.docString = doc;
                this.graph = g != null ? g : new Graph("", new ArrayList<>(), new ArrayList<>(),
                        new ArrayList<>(), new HashMap<>(), new ArrayList<>());
            }
        }

        private Graph parseGraph(MemorySegment bytes) {
            ProtobufParser parser = new ProtobufParser(bytes);
            String name = "";
            List<ValueInfo> inputs = new ArrayList<>();
            List<ValueInfo> outputs = new ArrayList<>();
            List<Node> nodes = new ArrayList<>();
            Map<String, TensorProto> initializers = new HashMap<>();
            List<ValueInfo> valueInfos = new ArrayList<>();

            while (parser.hasMore()) {
                int tag = parser.readTag();
                int fieldNumber = tag >>> 3;
                int wireType = tag & 0x7;

                switch (fieldNumber) {
                    case 1 -> nodes.add(parseNode(parser.readBytes()));
                    case 2 -> name = parser.readString();
                    case 5 -> {
                        TensorProto tensor = parseTensor(parser.readBytes());
                        initializers.put(tensor.name, tensor);
                    }
                    case 11 -> inputs.add(parseValueInfo(parser.readBytes()));
                    case 12 -> outputs.add(parseValueInfo(parser.readBytes()));
                    case 13 -> valueInfos.add(parseValueInfo(parser.readBytes()));
                    default -> parser.skip(wireType);
                }
            }

            return new Graph(name, inputs, outputs, nodes, initializers, valueInfos);
        }

        private Node parseNode(MemorySegment bytes) {
            ProtobufParser parser = new ProtobufParser(bytes);
            List<String> inputs = new ArrayList<>();
            List<String> outputs = new ArrayList<>();
            String name = "";
            String opType = "";
            String domain = "";

            while (parser.hasMore()) {
                int tag = parser.readTag();
                int fieldNumber = tag >>> 3;
                int wireType = tag & 0x7;

                switch (fieldNumber) {
                    case 1 -> inputs.add(parser.readString());
                    case 2 -> outputs.add(parser.readString());
                    case 3 -> name = parser.readString();
                    case 4 -> opType = parser.readString();
                    case 7 -> domain = parser.readString();
                    default -> parser.skip(wireType);
                }
            }

            return new Node(name, opType, domain, inputs, outputs);
        }

        private TensorProto parseTensor(MemorySegment bytes) {
            ProtobufParser parser = new ProtobufParser(bytes);
            List<Long> dims = new ArrayList<>();
            int dataType = FLOAT;
            String name = "";
            MemorySegment rawData = null;
            List<Float> floatData = new ArrayList<>();
            List<Double> doubleData = new ArrayList<>();
            List<Long> int64Data = new ArrayList<>();
            List<Integer> int32Data = new ArrayList<>();
            String externalDataLocation = null;

            while (parser.hasMore()) {
                int tag = parser.readTag();
                int fieldNumber = tag >>> 3;
                int wireType = tag & 0x7;

                switch (fieldNumber) {
                    case 1 -> dims.add(parser.readVarint());
                    case 2 -> dataType = (int) parser.readVarint();
                    case 8 -> name = parser.readString();
                    case 9 -> rawData = parser.readBytes();
                    case 4 -> floatData.add(parser.readFloat());
                    case 5 -> int32Data.add((int) parser.readVarint());
                    case 7 -> int64Data.add(parser.readVarint());
                    case 10 -> doubleData.add(parser.readDouble());
                    case 13 -> {
                        // External data - parse StringStringEntry
                        MemorySegment entry = parser.readBytes();
                        ProtobufParser entryParser = new ProtobufParser(entry);
                        String key = "";
                        String value = "";
                        while (entryParser.hasMore()) {
                            int entryTag = entryParser.readTag();
                            int entryField = entryTag >>> 3;
                            if (entryField == 1) key = entryParser.readString();
                            else if (entryField == 2) value = entryParser.readString();
                            else entryParser.skip(entryTag & 0x7);
                        }
                        if ("location".equals(key)) externalDataLocation = value;
                    }
                    default -> parser.skip(wireType);
                }
            }

            return new TensorProto(name, dims, dataType, rawData, floatData, doubleData,
                    int64Data, int32Data, externalDataLocation);
        }

        private ValueInfo parseValueInfo(MemorySegment bytes) {
            ProtobufParser parser = new ProtobufParser(bytes);
            String name = "";
            int elemType = FLOAT;
            List<Long> shape = new ArrayList<>();

            while (parser.hasMore()) {
                int tag = parser.readTag();
                int fieldNumber = tag >>> 3;
                int wireType = tag & 0x7;

                switch (fieldNumber) {
                    case 1 -> name = parser.readString();
                    case 2 -> {
                        // TypeProto - need to parse nested
                        MemorySegment typeBytes = parser.readBytes();
                        ProtobufParser typeParser = new ProtobufParser(typeBytes);
                        while (typeParser.hasMore()) {
                            int typeTag = typeParser.readTag();
                            int typeField = typeTag >>> 3;
                            if (typeField == 1) {
                                // tensor_type
                                MemorySegment tensorTypeBytes = typeParser.readBytes();
                                ProtobufParser ttParser = new ProtobufParser(tensorTypeBytes);
                                while (ttParser.hasMore()) {
                                    int ttTag = ttParser.readTag();
                                    int ttField = ttTag >>> 3;
                                    if (ttField == 1) elemType = (int) ttParser.readVarint();
                                    else if (ttField == 2) {
                                        // TensorShapeProto
                                        MemorySegment shapeBytes = ttParser.readBytes();
                                        ProtobufParser shapeParser = new ProtobufParser(shapeBytes);
                                        while (shapeParser.hasMore()) {
                                            int shapeTag = shapeParser.readTag();
                                            int shapeField = shapeTag >>> 3;
                                            if (shapeField == 1) {
                                                // Dimension
                                                MemorySegment dimBytes = shapeParser.readBytes();
                                                ProtobufParser dimParser = new ProtobufParser(dimBytes);
                                                while (dimParser.hasMore()) {
                                                    int dimTag = dimParser.readTag();
                                                    int dimField = dimTag >>> 3;
                                                    if (dimField == 1) shape.add(dimParser.readVarint());
                                                    else dimParser.skip(dimTag & 0x7);
                                                }
                                            } else shapeParser.skip(shapeTag & 0x7);
                                        }
                                    } else ttParser.skip(ttTag & 0x7);
                                }
                            } else typeParser.skip(typeTag & 0x7);
                        }
                    }
                    default -> parser.skip(wireType);
                }
            }

            return new ValueInfo(name, elemType, shape);
        }

        private OperatorSetId parseOpsetId(MemorySegment bytes) {
            ProtobufParser parser = new ProtobufParser(bytes);
            String domain = "";
            long version = 0;

            while (parser.hasMore()) {
                int tag = parser.readTag();
                int fieldNumber = tag >>> 3;
                int wireType = tag & 0x7;

                switch (fieldNumber) {
                    case 1 -> domain = parser.readString();
                    case 2 -> version = parser.readVarint();
                    default -> parser.skip(wireType);
                }
            }

            return new OperatorSetId(domain, version);
        }

        private void parseMetadataProp(MemorySegment bytes) {
            ProtobufParser parser = new ProtobufParser(bytes);
            String key = "";
            String value = "";

            while (parser.hasMore()) {
                int tag = parser.readTag();
                int fieldNumber = tag >>> 3;
                int wireType = tag & 0x7;

                switch (fieldNumber) {
                    case 1 -> key = parser.readString();
                    case 2 -> value = parser.readString();
                    default -> parser.skip(wireType);
                }
            }

            if (!key.isEmpty()) {
                metadataProps.put(key, value);
            }
        }

        // Public API

        public Path path() { return path; }
        public long irVersion() { return irVersion; }
        public String producer() { return producerName + " " + producerVersion; }
        public String producerName() { return producerName; }
        public String producerVersion() { return producerVersion; }
        public String domain() { return domain; }
        public long modelVersion() { return modelVersion; }
        public String docString() { return docString; }
        public List<OperatorSetId> opsetImports() { return List.copyOf(opsetImports); }
        public Map<String, String> metadata() { return Map.copyOf(metadataProps); }

        public String graphName() { return graph.name; }
        public List<ValueInfo> inputs() { return List.copyOf(graph.inputs); }
        public List<ValueInfo> outputs() { return List.copyOf(graph.outputs); }
        public List<Node> nodes() { return List.copyOf(graph.nodes); }
        public List<String> initializerNames() { return List.copyOf(graph.initializers.keySet()); }

        /**
         * Get an initializer (weight tensor) by name.
         */
        public TensorView initializer(String name) {
            TensorProto tensor = graph.initializers.get(name);
            if (tensor == null) return null;

            long[] shape = tensor.dims.stream().mapToLong(Long::longValue).toArray();
            DType dtype = toDType(tensor.dataType);

            long elementCount = 1;
            for (long dim : shape) elementCount *= dim;
            long byteSize = elementCount * dtype.byteSize();

            MemorySegment segment;
            if (tensor.rawData != null && tensor.rawData.byteSize() > 0) {
                segment = tensor.rawData;
            } else if (tensor.externalDataLocation != null) {
                // Load external data
                try {
                    Path externalPath = path.getParent().resolve(tensor.externalDataLocation);
                    segment = loadExternalData(externalPath, byteSize);
                } catch (IOException e) {
                    throw new RuntimeException("Failed to load external data: " + tensor.externalDataLocation, e);
                }
            } else {
                // Data in typed arrays - convert to raw
                segment = arena.allocate(byteSize);
                copyTypedDataToSegment(tensor, segment, dtype);
            }

            TensorInfo info = new TensorInfo(name, dtype, shape, 0, byteSize);
            return new TensorView(segment, info);
        }

        private MemorySegment loadExternalData(Path externalPath, long byteSize) throws IOException {
            try (FileChannel channel = FileChannel.open(externalPath, StandardOpenOption.READ)) {
                MemorySegment segment = arena.allocate(byteSize);
                ByteBuffer buffer = segment.asByteBuffer();
                channel.read(buffer);
                return segment;
            }
        }

        private void copyTypedDataToSegment(TensorProto tensor, MemorySegment segment, DType dtype) {
            switch (dtype) {
                case F32 -> {
                    for (int i = 0; i < tensor.floatData.size(); i++) {
                        segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, tensor.floatData.get(i));
                    }
                }
                case F64 -> {
                    for (int i = 0; i < tensor.doubleData.size(); i++) {
                        segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, tensor.doubleData.get(i));
                    }
                }
                case I64 -> {
                    for (int i = 0; i < tensor.int64Data.size(); i++) {
                        segment.setAtIndex(ValueLayout.JAVA_LONG, i, tensor.int64Data.get(i));
                    }
                }
                case I32 -> {
                    for (int i = 0; i < tensor.int32Data.size(); i++) {
                        segment.setAtIndex(ValueLayout.JAVA_INT, i, tensor.int32Data.get(i));
                    }
                }
                default -> throw new UnsupportedOperationException("Cannot copy typed data for " + dtype);
            }
        }

        /**
         * Check if an initializer exists.
         */
        public boolean hasInitializer(String name) {
            return graph.initializers.containsKey(name);
        }

        /**
         * Get total number of parameters.
         */
        public long parameterCount() {
            long count = 0;
            for (TensorProto tensor : graph.initializers.values()) {
                long elements = 1;
                for (long dim : tensor.dims) elements *= dim;
                count += elements;
            }
            return count;
        }

        @Override
        public void close() {
            arena.close();
        }
    }

    // Internal data structures

    private record Graph(
            String name,
            List<ValueInfo> inputs,
            List<ValueInfo> outputs,
            List<Node> nodes,
            Map<String, TensorProto> initializers,
            List<ValueInfo> valueInfos
    ) {}

    private record TensorProto(
            String name,
            List<Long> dims,
            int dataType,
            MemorySegment rawData,
            List<Float> floatData,
            List<Double> doubleData,
            List<Long> int64Data,
            List<Integer> int32Data,
            String externalDataLocation
    ) {}

    /**
     * Graph node (operation).
     */
    public record Node(
            String name,
            String opType,
            String domain,
            List<String> inputs,
            List<String> outputs
    ) {}

    /**
     * Value info (input/output specification).
     */
    public record ValueInfo(
            String name,
            int elemType,
            List<Long> shape
    ) {
        public DType dtype() {
            return toDType(elemType);
        }

        public long[] shapeArray() {
            return shape.stream().mapToLong(Long::longValue).toArray();
        }
    }

    /**
     * Operator set import.
     */
    public record OperatorSetId(String domain, long version) {}

    /**
     * Simple protobuf parser for ONNX format.
     */
    private static class ProtobufParser {
        private final MemorySegment data;
        private long pos;
        private final long end;

        ProtobufParser(MemorySegment data) {
            this.data = data;
            this.pos = 0;
            this.end = data.byteSize();
        }

        boolean hasMore() {
            return pos < end;
        }

        int readTag() {
            return (int) readVarint();
        }

        long readVarint() {
            long result = 0;
            int shift = 0;
            while (pos < end) {
                byte b = data.get(ValueLayout.JAVA_BYTE, pos++);
                result |= (long) (b & 0x7F) << shift;
                if ((b & 0x80) == 0) break;
                shift += 7;
            }
            return result;
        }

        String readString() {
            int length = (int) readVarint();
            if (length == 0) return "";
            byte[] bytes = new byte[length];
            MemorySegment.copy(data, ValueLayout.JAVA_BYTE, pos, bytes, 0, length);
            pos += length;
            return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
        }

        MemorySegment readBytes() {
            int length = (int) readVarint();
            if (length == 0) return MemorySegment.NULL;
            MemorySegment slice = data.asSlice(pos, length);
            pos += length;
            return slice;
        }

        float readFloat() {
            int bits = (int) readFixed32();
            return Float.intBitsToFloat(bits);
        }

        double readDouble() {
            long bits = readFixed64();
            return Double.longBitsToDouble(bits);
        }

        long readFixed32() {
            long value = data.get(ValueLayout.JAVA_INT_UNALIGNED, pos) & 0xFFFFFFFFL;
            pos += 4;
            return value;
        }

        long readFixed64() {
            long value = data.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);
            pos += 8;
            return value;
        }

        void skip(int wireType) {
            switch (wireType) {
                case 0 -> readVarint();
                case 1 -> pos += 8;
                case 2 -> {
                    int length = (int) readVarint();
                    pos += length;
                }
                case 5 -> pos += 4;
                default -> throw new IllegalStateException("Unknown wire type: " + wireType);
            }
        }
    }
}
