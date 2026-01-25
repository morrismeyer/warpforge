package io.surfworks.warpforge.data.format;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GGUFTest {

    @TempDir
    Path tempDir;

    @Test
    void testReadSimpleGGUF() throws IOException {
        Path file = createSimpleGGUF();

        try (GGUF gguf = GGUF.open(file)) {
            assertEquals(3, gguf.version());
            assertEquals(1, gguf.tensorCount());
            assertTrue(gguf.hasTensor("test_tensor"));

            TensorInfo info = gguf.tensorInfo("test_tensor");
            assertEquals("test_tensor", info.name());
            assertEquals(DType.F32, info.dtype());
            assertArrayEquals(new long[]{2, 3}, info.shape());
            assertEquals(6, info.elementCount());

            TensorView view = gguf.tensor("test_tensor");
            assertNotNull(view);
            assertEquals(24, view.byteSize()); // 6 floats * 4 bytes

            // Check values
            assertEquals(1.0f, view.getFloat(0, 0), 0.001f);
            assertEquals(2.0f, view.getFloat(0, 1), 0.001f);
            assertEquals(3.0f, view.getFloat(0, 2), 0.001f);
            assertEquals(4.0f, view.getFloat(1, 0), 0.001f);
            assertEquals(5.0f, view.getFloat(1, 1), 0.001f);
            assertEquals(6.0f, view.getFloat(1, 2), 0.001f);
        }
    }

    @Test
    void testToFloatArray() throws IOException {
        Path file = createSimpleGGUF();

        try (GGUF gguf = GGUF.open(file)) {
            TensorView view = gguf.tensor("test_tensor");
            float[] array = view.toFloatArray();

            assertEquals(6, array.length);
            assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, array, 0.001f);
        }
    }

    @Test
    void testMetadata() throws IOException {
        Path file = createGGUFWithMetadata();

        try (GGUF gguf = GGUF.open(file)) {
            assertEquals("llama", gguf.architecture());
            assertEquals("Test Model", gguf.modelName());
            assertEquals("llama", gguf.metadata().get("general.architecture"));
            assertEquals("Test Model", gguf.metadata().get("general.name"));
        }
    }

    @Test
    void testF16Tensor() throws IOException {
        Path file = createF16GGUF();

        try (GGUF gguf = GGUF.open(file)) {
            TensorInfo info = gguf.tensorInfo("f16_tensor");
            assertEquals(DType.F16, info.dtype());
            assertEquals(4, info.elementCount());

            TensorView view = gguf.tensor("f16_tensor");
            assertEquals(8, view.byteSize()); // 4 f16 values * 2 bytes

            // Check F16 values (1.0, 2.0, 3.0, 4.0)
            assertEquals(1.0f, view.getFloat(0), 0.01f);
            assertEquals(2.0f, view.getFloat(1), 0.01f);
            assertEquals(3.0f, view.getFloat(2), 0.01f);
            assertEquals(4.0f, view.getFloat(3), 0.01f);
        }
    }

    @Test
    void testMultipleTensors() throws IOException {
        Path file = createMultiTensorGGUF();

        try (GGUF gguf = GGUF.open(file)) {
            assertEquals(2, gguf.tensorCount());
            assertTrue(gguf.hasTensor("weights"));
            assertTrue(gguf.hasTensor("bias"));
            assertFalse(gguf.hasTensor("nonexistent"));

            TensorInfo weightsInfo = gguf.tensorInfo("weights");
            assertEquals(DType.F32, weightsInfo.dtype());
            assertArrayEquals(new long[]{4}, weightsInfo.shape());

            TensorInfo biasInfo = gguf.tensorInfo("bias");
            assertEquals(DType.F32, biasInfo.dtype());
            assertArrayEquals(new long[]{2}, biasInfo.shape());

            // Check weights values
            TensorView weights = gguf.tensor("weights");
            assertEquals(1.0f, weights.getFloatFlat(0), 0.001f);
            assertEquals(2.0f, weights.getFloatFlat(1), 0.001f);
            assertEquals(3.0f, weights.getFloatFlat(2), 0.001f);
            assertEquals(4.0f, weights.getFloatFlat(3), 0.001f);

            // Check bias values
            TensorView bias = gguf.tensor("bias");
            assertEquals(0.5f, bias.getFloatFlat(0), 0.001f);
            assertEquals(0.25f, bias.getFloatFlat(1), 0.001f);
        }
    }

    @Test
    void testInvalidMagic() throws IOException {
        Path file = tempDir.resolve("invalid.gguf");
        Files.write(file, "NOT_GGUF_FILE_DATA_HERE!".getBytes());

        IOException ex = assertThrows(IOException.class, () -> GGUF.open(file));
        assertTrue(ex.getMessage().contains("Invalid GGUF magic"));
    }

    @Test
    void testFileTooSmall() throws IOException {
        Path file = tempDir.resolve("tiny.gguf");
        Files.write(file, new byte[10]); // Too small

        IOException ex = assertThrows(IOException.class, () -> GGUF.open(file));
        assertTrue(ex.getMessage().contains("too small"));
    }

    @Test
    void testTensorNotFound() throws IOException {
        Path file = createSimpleGGUF();

        try (GGUF gguf = GGUF.open(file)) {
            assertThrows(IllegalArgumentException.class, () -> gguf.tensorInfo("nonexistent"));
            assertThrows(IllegalArgumentException.class, () -> gguf.tensor("nonexistent"));
        }
    }

    @Test
    void testToString() throws IOException {
        Path file = createGGUFWithMetadata();

        try (GGUF gguf = GGUF.open(file)) {
            String str = gguf.toString();
            assertTrue(str.contains("GGUF"));
            assertTrue(str.contains("llama"));
            assertTrue(str.contains("v3"));
        }
    }

    @Test
    void testQ4_0Tensor() throws IOException {
        // Q4_0: 32 elements per block, 18 bytes per block (16 data + 2 scale)
        Path file = createQ4_0GGUF();

        try (GGUF gguf = GGUF.open(file)) {
            TensorInfo info = gguf.tensorInfo("q4_tensor");
            assertEquals(DType.Q4_0, info.dtype());
            assertArrayEquals(new long[]{32}, info.shape());
            // 32 elements = 1 block = 18 bytes
            assertEquals(18, info.size());
        }
    }

    @Test
    void testQ8_0Tensor() throws IOException {
        // Q8_0: 32 elements per block, 34 bytes per block (32 data + 2 scale)
        Path file = createQ8_0GGUF();

        try (GGUF gguf = GGUF.open(file)) {
            TensorInfo info = gguf.tensorInfo("q8_tensor");
            assertEquals(DType.Q8_0, info.dtype());
            assertArrayEquals(new long[]{64}, info.shape());
            // 64 elements = 2 blocks = 68 bytes
            assertEquals(68, info.size());
        }
    }

    @Test
    void testBF16Tensor() throws IOException {
        Path file = createBF16GGUF();

        try (GGUF gguf = GGUF.open(file)) {
            TensorInfo info = gguf.tensorInfo("bf16_tensor");
            assertEquals(DType.BF16, info.dtype());
            assertEquals(4, info.elementCount());

            TensorView view = gguf.tensor("bf16_tensor");
            assertEquals(8, view.byteSize()); // 4 bf16 values * 2 bytes
        }
    }

    @Test
    void testVersion2() throws IOException {
        Path file = createVersion2GGUF();

        try (GGUF gguf = GGUF.open(file)) {
            assertEquals(2, gguf.version());
            assertEquals(1, gguf.tensorCount());
            assertTrue(gguf.hasTensor("v2_tensor"));
        }
    }

    @Test
    void testIntegerMetadata() throws IOException {
        Path file = createGGUFWithIntMetadata();

        try (GGUF gguf = GGUF.open(file)) {
            Object blockCount = gguf.metadata().get("llama.block_count");
            assertNotNull(blockCount);
            // Integer metadata values
            assertTrue(blockCount instanceof Number);
            assertEquals(32, ((Number) blockCount).intValue());
        }
    }

    // =========================================================================
    // Test File Creators
    // =========================================================================

    /**
     * Create a simple GGUF v3 file with a single 2x3 F32 tensor.
     */
    private Path createSimpleGGUF() throws IOException {
        Path file = tempDir.resolve("simple.gguf");
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        try (Arena arena = Arena.ofConfined()) {
            GGUFWriter writer = new GGUFWriter(arena, 1024);

            // Header
            writer.writeInt(0x46554747); // Magic "GGUF"
            writer.writeInt(3);          // Version 3
            writer.writeLong(1);         // Tensor count
            writer.writeLong(0);         // Metadata KV count

            // Tensor info: "test_tensor", 2D [2,3], F32, offset 0
            writer.writeString("test_tensor");
            writer.writeInt(2);          // n_dims
            writer.writeLong(2);         // dim[0]
            writer.writeLong(3);         // dim[1]
            writer.writeInt(0);          // GGML_TYPE_F32
            writer.writeLong(0);         // offset

            // Padding to 32-byte alignment
            writer.alignTo(32);

            // Tensor data
            for (float v : values) {
                writer.writeFloat(v);
            }

            byte[] fileBytes = writer.toByteArray();
            Files.write(file, fileBytes);
        }

        return file;
    }

    /**
     * Create a GGUF file with metadata.
     */
    private Path createGGUFWithMetadata() throws IOException {
        Path file = tempDir.resolve("with_metadata.gguf");
        float[] values = {1.0f, 2.0f};

        try (Arena arena = Arena.ofConfined()) {
            GGUFWriter writer = new GGUFWriter(arena, 1024);

            // Header
            writer.writeInt(0x46554747); // Magic
            writer.writeInt(3);          // Version
            writer.writeLong(1);         // Tensor count
            writer.writeLong(2);         // Metadata KV count (architecture + name)

            // Metadata: general.architecture = "llama"
            writer.writeString("general.architecture");
            writer.writeInt(8);          // GGUF_TYPE_STRING
            writer.writeString("llama");

            // Metadata: general.name = "Test Model"
            writer.writeString("general.name");
            writer.writeInt(8);          // GGUF_TYPE_STRING
            writer.writeString("Test Model");

            // Tensor info
            writer.writeString("test_tensor");
            writer.writeInt(1);          // n_dims
            writer.writeLong(2);         // dim[0]
            writer.writeInt(0);          // GGML_TYPE_F32
            writer.writeLong(0);         // offset

            // Padding
            writer.alignTo(32);

            // Data
            for (float v : values) {
                writer.writeFloat(v);
            }

            Files.write(file, writer.toByteArray());
        }

        return file;
    }

    /**
     * Create a GGUF file with F16 tensor.
     */
    private Path createF16GGUF() throws IOException {
        Path file = tempDir.resolve("f16.gguf");

        try (Arena arena = Arena.ofConfined()) {
            GGUFWriter writer = new GGUFWriter(arena, 1024);

            // Header
            writer.writeInt(0x46554747);
            writer.writeInt(3);
            writer.writeLong(1);
            writer.writeLong(0);

            // Tensor info: F16
            writer.writeString("f16_tensor");
            writer.writeInt(1);
            writer.writeLong(4);
            writer.writeInt(1);          // GGML_TYPE_F16
            writer.writeLong(0);

            writer.alignTo(32);

            // F16 data: 1.0, 2.0, 3.0, 4.0
            writer.writeShort((short) 0x3C00); // 1.0 in F16
            writer.writeShort((short) 0x4000); // 2.0 in F16
            writer.writeShort((short) 0x4200); // 3.0 in F16
            writer.writeShort((short) 0x4400); // 4.0 in F16

            Files.write(file, writer.toByteArray());
        }

        return file;
    }

    /**
     * Create a GGUF file with Q4_0 quantized tensor.
     */
    private Path createQ4_0GGUF() throws IOException {
        Path file = tempDir.resolve("q4_0.gguf");

        try (Arena arena = Arena.ofConfined()) {
            GGUFWriter writer = new GGUFWriter(arena, 1024);

            // Header
            writer.writeInt(0x46554747);
            writer.writeInt(3);
            writer.writeLong(1);
            writer.writeLong(0);

            // Tensor info: Q4_0 type (GGML type 2)
            writer.writeString("q4_tensor");
            writer.writeInt(1);
            writer.writeLong(32);        // 32 elements = 1 block
            writer.writeInt(2);          // GGML_TYPE_Q4_0
            writer.writeLong(0);

            writer.alignTo(32);

            // Q4_0 block: 2 bytes scale + 16 bytes data (32 4-bit values)
            writer.writeShort((short) 0x3C00); // scale (F16 1.0)
            for (int i = 0; i < 16; i++) {
                writer.writeByte((byte) 0x00);
            }

            Files.write(file, writer.toByteArray());
        }

        return file;
    }

    /**
     * Create a GGUF file with Q8_0 quantized tensor.
     */
    private Path createQ8_0GGUF() throws IOException {
        Path file = tempDir.resolve("q8_0.gguf");

        try (Arena arena = Arena.ofConfined()) {
            GGUFWriter writer = new GGUFWriter(arena, 1024);

            // Header
            writer.writeInt(0x46554747);
            writer.writeInt(3);
            writer.writeLong(1);
            writer.writeLong(0);

            // Tensor info: Q8_0 type (GGML type 8)
            writer.writeString("q8_tensor");
            writer.writeInt(1);
            writer.writeLong(64);        // 64 elements = 2 blocks
            writer.writeInt(8);          // GGML_TYPE_Q8_0
            writer.writeLong(0);

            writer.alignTo(32);

            // Two Q8_0 blocks: 2 bytes scale + 32 bytes data each
            for (int block = 0; block < 2; block++) {
                writer.writeShort((short) 0x3C00); // scale
                for (int i = 0; i < 32; i++) {
                    writer.writeByte((byte) 0x00);
                }
            }

            Files.write(file, writer.toByteArray());
        }

        return file;
    }

    /**
     * Create a GGUF file with BF16 tensor.
     */
    private Path createBF16GGUF() throws IOException {
        Path file = tempDir.resolve("bf16.gguf");

        try (Arena arena = Arena.ofConfined()) {
            GGUFWriter writer = new GGUFWriter(arena, 1024);

            // Header
            writer.writeInt(0x46554747);
            writer.writeInt(3);
            writer.writeLong(1);
            writer.writeLong(0);

            // Tensor info: BF16 (GGML type 18)
            writer.writeString("bf16_tensor");
            writer.writeInt(1);
            writer.writeLong(4);
            writer.writeInt(18);         // GGML_TYPE_BF16
            writer.writeLong(0);

            writer.alignTo(32);

            // BF16 values (truncated F32)
            writer.writeShort((short) 0x3F80); // 1.0 in BF16
            writer.writeShort((short) 0x4000); // 2.0 in BF16
            writer.writeShort((short) 0x4040); // 3.0 in BF16
            writer.writeShort((short) 0x4080); // 4.0 in BF16

            Files.write(file, writer.toByteArray());
        }

        return file;
    }

    /**
     * Create a GGUF v2 file.
     */
    private Path createVersion2GGUF() throws IOException {
        Path file = tempDir.resolve("v2.gguf");

        try (Arena arena = Arena.ofConfined()) {
            GGUFWriter writer = new GGUFWriter(arena, 1024);

            // Header with version 2
            writer.writeInt(0x46554747);
            writer.writeInt(2);          // Version 2
            writer.writeLong(1);
            writer.writeLong(0);

            // Tensor info
            writer.writeString("v2_tensor");
            writer.writeInt(1);
            writer.writeLong(4);
            writer.writeInt(0);          // F32
            writer.writeLong(0);

            writer.alignTo(32);

            // Data
            for (int i = 0; i < 4; i++) {
                writer.writeFloat((float) i);
            }

            Files.write(file, writer.toByteArray());
        }

        return file;
    }

    /**
     * Create a GGUF file with integer metadata.
     */
    private Path createGGUFWithIntMetadata() throws IOException {
        Path file = tempDir.resolve("int_metadata.gguf");

        try (Arena arena = Arena.ofConfined()) {
            GGUFWriter writer = new GGUFWriter(arena, 1024);

            // Header
            writer.writeInt(0x46554747);
            writer.writeInt(3);
            writer.writeLong(1);
            writer.writeLong(1);         // 1 metadata entry

            // Metadata: llama.block_count = 32 (INT32)
            writer.writeString("llama.block_count");
            writer.writeInt(5);          // GGUF_TYPE_INT32
            writer.writeInt(32);

            // Tensor info
            writer.writeString("test_tensor");
            writer.writeInt(1);
            writer.writeLong(2);
            writer.writeInt(0);
            writer.writeLong(0);

            writer.alignTo(32);

            // Data
            writer.writeFloat(1.0f);
            writer.writeFloat(2.0f);

            Files.write(file, writer.toByteArray());
        }

        return file;
    }

    /**
     * Create a GGUF file with multiple tensors.
     */
    private Path createMultiTensorGGUF() throws IOException {
        Path file = tempDir.resolve("multi.gguf");

        try (Arena arena = Arena.ofConfined()) {
            GGUFWriter writer = new GGUFWriter(arena, 1024);

            // Header
            writer.writeInt(0x46554747);
            writer.writeInt(3);
            writer.writeLong(2);         // 2 tensors
            writer.writeLong(0);

            // Tensor 1: weights [4] F32, offset 0
            writer.writeString("weights");
            writer.writeInt(1);
            writer.writeLong(4);
            writer.writeInt(0);          // F32
            writer.writeLong(0);

            // Tensor 2: bias [2] F32, offset 16 (after 4 floats)
            writer.writeString("bias");
            writer.writeInt(1);
            writer.writeLong(2);
            writer.writeInt(0);          // F32
            writer.writeLong(16);        // offset after weights

            writer.alignTo(32);

            // Weights data
            writer.writeFloat(1.0f);
            writer.writeFloat(2.0f);
            writer.writeFloat(3.0f);
            writer.writeFloat(4.0f);

            // Bias data
            writer.writeFloat(0.5f);
            writer.writeFloat(0.25f);

            Files.write(file, writer.toByteArray());
        }

        return file;
    }

    // =========================================================================
    // Helper Writer Class
    // =========================================================================

    private static class GGUFWriter {
        private final MemorySegment buffer;
        private long pos;

        GGUFWriter(Arena arena, int capacity) {
            this.buffer = arena.allocate(capacity);
            this.pos = 0;
        }

        void writeInt(int value) {
            buffer.set(ValueLayout.JAVA_INT_UNALIGNED, pos, value);
            pos += 4;
        }

        void writeLong(long value) {
            buffer.set(ValueLayout.JAVA_LONG_UNALIGNED, pos, value);
            pos += 8;
        }

        void writeFloat(float value) {
            buffer.set(ValueLayout.JAVA_FLOAT_UNALIGNED, pos, value);
            pos += 4;
        }

        void writeShort(short value) {
            buffer.set(ValueLayout.JAVA_SHORT_UNALIGNED, pos, value);
            pos += 2;
        }

        void writeByte(byte value) {
            buffer.set(ValueLayout.JAVA_BYTE, pos, value);
            pos += 1;
        }

        void writeString(String s) {
            byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
            writeLong(bytes.length);
            MemorySegment.copy(bytes, 0, buffer, ValueLayout.JAVA_BYTE, pos, bytes.length);
            pos += bytes.length;
        }

        void alignTo(int alignment) {
            long aligned = ((pos + alignment - 1) / alignment) * alignment;
            while (pos < aligned) {
                buffer.set(ValueLayout.JAVA_BYTE, pos, (byte) 0);
                pos++;
            }
        }

        byte[] toByteArray() {
            byte[] result = new byte[(int) pos];
            MemorySegment.copy(buffer, ValueLayout.JAVA_BYTE, 0, result, 0, (int) pos);
            return result;
        }
    }
}
