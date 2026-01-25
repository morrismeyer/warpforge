package io.surfworks.warpforge.data.format;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ArrowIPCTest {

    @TempDir
    Path tempDir;

    @Nested
    class ValidationTests {

        @Test
        void testInvalidMagicBytesAtStart() throws IOException {
            Path file = tempDir.resolve("invalid.arrow");
            Files.write(file, new byte[100]);

            assertThrows(IOException.class, () -> ArrowIPC.open(file));
        }

        @Test
        void testInvalidMagicBytesAtEnd() throws IOException {
            Path file = tempDir.resolve("invalid_end.arrow");
            ByteBuffer buf = ByteBuffer.allocate(100);
            buf.put("ARROW1".getBytes()); // Valid start
            buf.position(94);
            buf.put(new byte[]{0, 0, 0, 0, 0, 0}); // Invalid end
            Files.write(file, buf.array());

            assertThrows(IOException.class, () -> ArrowIPC.open(file));
        }

        @Test
        void testValidArrowFile() throws IOException {
            Path file = tempDir.resolve("valid.arrow");
            ByteBuffer buf = createMinimalArrowFile();
            Files.write(file, buf.array());

            try (ArrowIPC.File arrowFile = ArrowIPC.open(file)) {
                assertNotNull(arrowFile);
                assertNotNull(arrowFile.schema());
            }
        }
    }

    @Nested
    class SchemaTests {

        @Test
        void testSchemaFields() throws IOException {
            Path file = tempDir.resolve("schema.arrow");
            Files.write(file, createMinimalArrowFile().array());

            try (ArrowIPC.File arrowFile = ArrowIPC.open(file)) {
                ArrowIPC.Schema schema = arrowFile.schema();

                assertNotNull(schema);
                assertTrue(schema.numFields() > 0);
                assertNotNull(schema.field(0));
            }
        }

        @Test
        void testFieldProperties() throws IOException {
            Path file = tempDir.resolve("fields.arrow");
            Files.write(file, createMinimalArrowFile().array());

            try (ArrowIPC.File arrowFile = ArrowIPC.open(file)) {
                ArrowIPC.Field field = arrowFile.schema().field(0);

                assertNotNull(field.name());
                assertNotNull(field.type());
            }
        }
    }

    @Nested
    class DataTypeTests {

        @Test
        void testByteWidth() {
            assertEquals(1, ArrowIPC.DataType.INT8.byteWidth());
            assertEquals(2, ArrowIPC.DataType.INT16.byteWidth());
            assertEquals(4, ArrowIPC.DataType.INT32.byteWidth());
            assertEquals(8, ArrowIPC.DataType.INT64.byteWidth());
            assertEquals(4, ArrowIPC.DataType.FLOAT32.byteWidth());
            assertEquals(8, ArrowIPC.DataType.FLOAT64.byteWidth());
        }

        @Test
        void testVariableWidthTypes() {
            assertTrue(ArrowIPC.DataType.STRING.isVariableWidth());
            assertTrue(ArrowIPC.DataType.BINARY.isVariableWidth());
            assertTrue(ArrowIPC.DataType.LIST.isVariableWidth());
            assertFalse(ArrowIPC.DataType.INT32.isVariableWidth());
            assertFalse(ArrowIPC.DataType.FLOAT64.isVariableWidth());
        }

        @Test
        void testVariableWidthByteWidth() {
            assertEquals(-1, ArrowIPC.DataType.STRING.byteWidth());
            assertEquals(-1, ArrowIPC.DataType.LIST.byteWidth());
        }
    }

    @Nested
    class FileInfoTests {

        @Test
        void testFilePath() throws IOException {
            Path file = tempDir.resolve("path.arrow");
            Files.write(file, createMinimalArrowFile().array());

            try (ArrowIPC.File arrowFile = ArrowIPC.open(file)) {
                assertEquals(file, arrowFile.path());
            }
        }

        @Test
        void testNumBatches() throws IOException {
            Path file = tempDir.resolve("batches.arrow");
            Files.write(file, createMinimalArrowFile().array());

            try (ArrowIPC.File arrowFile = ArrowIPC.open(file)) {
                // Minimal file has no batches
                assertEquals(0, arrowFile.numBatches());
            }
        }
    }

    @Nested
    class FieldRecordTests {

        @Test
        void testFieldMetadata() {
            ArrowIPC.Field field = new ArrowIPC.Field("test", ArrowIPC.DataType.INT32, true);

            assertEquals("test", field.name());
            assertEquals(ArrowIPC.DataType.INT32, field.type());
            assertTrue(field.nullable());
            assertNotNull(field.metadata());
        }
    }

    /**
     * Creates a minimal valid Arrow IPC file structure.
     */
    private ByteBuffer createMinimalArrowFile() {
        // Minimal Arrow file:
        // [6 bytes: ARROW1 magic]
        // [2 bytes: padding]
        // [N bytes: empty schema flatbuffer placeholder]
        // [footer flatbuffer placeholder]
        // [4 bytes: footer size]
        // [6 bytes: ARROW1 magic]

        int footerSize = 32; // Placeholder footer
        int totalSize = 6 + 2 + footerSize + 4 + 6;

        ByteBuffer buf = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN);

        // Start magic
        buf.put("ARROW1".getBytes());

        // Padding
        buf.putShort((short) 0);

        // Footer placeholder (would be FlatBuffer in real file)
        for (int i = 0; i < footerSize; i++) {
            buf.put((byte) 0);
        }

        // Footer size
        buf.putInt(footerSize);

        // End magic
        buf.put("ARROW1".getBytes());

        return buf;
    }
}
