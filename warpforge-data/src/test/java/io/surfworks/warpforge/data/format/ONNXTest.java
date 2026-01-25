package io.surfworks.warpforge.data.format;

import io.surfworks.warpforge.data.DType;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ONNXTest {

    @TempDir
    Path tempDir;

    @Nested
    class DataTypeTests {

        @Test
        void testToDTypeFloat() {
            assertEquals(DType.F32, ONNX.toDType(ONNX.FLOAT));
        }

        @Test
        void testToDTypeDouble() {
            assertEquals(DType.F64, ONNX.toDType(ONNX.DOUBLE));
        }

        @Test
        void testToDTypeFloat16() {
            assertEquals(DType.F16, ONNX.toDType(ONNX.FLOAT16));
        }

        @Test
        void testToDTypeBFloat16() {
            assertEquals(DType.BF16, ONNX.toDType(ONNX.BFLOAT16));
        }

        @Test
        void testToDTypeInt8() {
            assertEquals(DType.I8, ONNX.toDType(ONNX.INT8));
        }

        @Test
        void testToDTypeInt16() {
            assertEquals(DType.I16, ONNX.toDType(ONNX.INT16));
        }

        @Test
        void testToDTypeInt32() {
            assertEquals(DType.I32, ONNX.toDType(ONNX.INT32));
        }

        @Test
        void testToDTypeInt64() {
            assertEquals(DType.I64, ONNX.toDType(ONNX.INT64));
        }

        @Test
        void testToDTypeUint8() {
            assertEquals(DType.U8, ONNX.toDType(ONNX.UINT8));
        }

        @Test
        void testToDTypeUnsupported() {
            assertThrows(IllegalArgumentException.class, () -> ONNX.toDType(ONNX.STRING));
            assertThrows(IllegalArgumentException.class, () -> ONNX.toDType(ONNX.COMPLEX64));
        }
    }

    @Nested
    class ValueInfoTests {

        @Test
        void testValueInfoDType() {
            ONNX.ValueInfo info = new ONNX.ValueInfo("test", ONNX.FLOAT, List.of(1L, 2L, 3L));

            assertEquals("test", info.name());
            assertEquals(DType.F32, info.dtype());
        }

        @Test
        void testValueInfoShapeArray() {
            ONNX.ValueInfo info = new ONNX.ValueInfo("test", ONNX.FLOAT, List.of(2L, 4L, 8L));

            long[] shape = info.shapeArray();
            assertEquals(3, shape.length);
            assertEquals(2L, shape[0]);
            assertEquals(4L, shape[1]);
            assertEquals(8L, shape[2]);
        }
    }

    @Nested
    class OperatorSetIdTests {

        @Test
        void testOperatorSetId() {
            ONNX.OperatorSetId opset = new ONNX.OperatorSetId("ai.onnx", 17);

            assertEquals("ai.onnx", opset.domain());
            assertEquals(17, opset.version());
        }

        @Test
        void testEmptyDomain() {
            ONNX.OperatorSetId opset = new ONNX.OperatorSetId("", 14);

            assertEquals("", opset.domain());
            assertEquals(14, opset.version());
        }
    }

    @Nested
    class NodeTests {

        @Test
        void testNode() {
            ONNX.Node node = new ONNX.Node(
                    "MatMul_0",
                    "MatMul",
                    "",
                    List.of("input", "weight"),
                    List.of("output")
            );

            assertEquals("MatMul_0", node.name());
            assertEquals("MatMul", node.opType());
            assertEquals(2, node.inputs().size());
            assertEquals(1, node.outputs().size());
        }
    }

    @Nested
    class MinimalModelTests {

        @Test
        void testLoadMinimalModel() throws IOException {
            // Create a minimal valid ONNX model (protobuf)
            Path modelPath = tempDir.resolve("minimal.onnx");
            byte[] minimalOnnx = createMinimalOnnxModel();
            Files.write(modelPath, minimalOnnx);

            try (ONNX.Model model = ONNX.load(modelPath)) {
                assertNotNull(model);
                assertEquals(modelPath, model.path());
            }
        }

        @Test
        void testModelMetadata() throws IOException {
            Path modelPath = tempDir.resolve("meta.onnx");
            byte[] onnxData = createOnnxModelWithMetadata();
            Files.write(modelPath, onnxData);

            try (ONNX.Model model = ONNX.load(modelPath)) {
                // IR version should be set
                assertTrue(model.irVersion() >= 0);
            }
        }

        /**
         * Create a minimal valid ONNX protobuf.
         * This is a simplified model with just required fields.
         */
        private byte[] createMinimalOnnxModel() {
            // Minimal ONNX model structure:
            // Field 1 (ir_version): varint
            // Field 7 (graph): length-delimited (GraphProto)

            ByteBuffer buf = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);

            // Field 1: ir_version = 8
            buf.put((byte) 0x08); // tag (field 1, wire type 0)
            buf.put((byte) 0x08); // value 8

            // Field 7: graph (empty graph)
            byte[] emptyGraph = createEmptyGraph();
            buf.put((byte) 0x3A); // tag (field 7, wire type 2)
            buf.put((byte) emptyGraph.length); // length
            buf.put(emptyGraph);

            buf.flip();
            byte[] result = new byte[buf.remaining()];
            buf.get(result);
            return result;
        }

        private byte[] createOnnxModelWithMetadata() {
            ByteBuffer buf = ByteBuffer.allocate(128).order(ByteOrder.LITTLE_ENDIAN);

            // Field 1: ir_version = 9
            buf.put((byte) 0x08);
            buf.put((byte) 0x09);

            // Field 2: producer_name = "test"
            buf.put((byte) 0x12);
            buf.put((byte) 0x04);
            buf.put("test".getBytes());

            // Field 3: producer_version = "1.0"
            buf.put((byte) 0x1A);
            buf.put((byte) 0x03);
            buf.put("1.0".getBytes());

            // Field 7: graph
            byte[] emptyGraph = createEmptyGraph();
            buf.put((byte) 0x3A);
            buf.put((byte) emptyGraph.length);
            buf.put(emptyGraph);

            buf.flip();
            byte[] result = new byte[buf.remaining()];
            buf.get(result);
            return result;
        }

        private byte[] createEmptyGraph() {
            ByteBuffer buf = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);

            // Field 2: name = "main"
            buf.put((byte) 0x12);
            buf.put((byte) 0x04);
            buf.put("main".getBytes());

            buf.flip();
            byte[] result = new byte[buf.remaining()];
            buf.get(result);
            return result;
        }
    }

    @Nested
    class ProtobufParsingTests {

        @Test
        void testVarintParsing() throws IOException {
            // Test that we can parse varints correctly
            Path modelPath = tempDir.resolve("varint.onnx");
            byte[] data = createModelWithLargeVarint();
            Files.write(modelPath, data);

            try (ONNX.Model model = ONNX.load(modelPath)) {
                // Should parse without error
                assertNotNull(model);
            }
        }

        private byte[] createModelWithLargeVarint() {
            ByteBuffer buf = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);

            // Field 1: ir_version = 300 (multi-byte varint: 0xAC 0x02)
            buf.put((byte) 0x08);
            buf.put((byte) 0xAC);
            buf.put((byte) 0x02);

            // Field 7: graph
            byte[] emptyGraph = createSimpleGraph();
            buf.put((byte) 0x3A);
            buf.put((byte) emptyGraph.length);
            buf.put(emptyGraph);

            buf.flip();
            byte[] result = new byte[buf.remaining()];
            buf.get(result);
            return result;
        }

        private byte[] createSimpleGraph() {
            ByteBuffer buf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            buf.put((byte) 0x12); // field 2
            buf.put((byte) 0x01); // length 1
            buf.put((byte) 'g');  // "g"
            buf.flip();
            byte[] result = new byte[buf.remaining()];
            buf.get(result);
            return result;
        }
    }

    @Nested
    class DataTypeConstantsTests {

        @Test
        void testDataTypeConstants() {
            assertEquals(0, ONNX.UNDEFINED);
            assertEquals(1, ONNX.FLOAT);
            assertEquals(2, ONNX.UINT8);
            assertEquals(3, ONNX.INT8);
            assertEquals(6, ONNX.INT32);
            assertEquals(7, ONNX.INT64);
            assertEquals(10, ONNX.FLOAT16);
            assertEquals(11, ONNX.DOUBLE);
            assertEquals(16, ONNX.BFLOAT16);
        }
    }
}
