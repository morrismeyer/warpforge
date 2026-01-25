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
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SafeTensorsTest {

    @TempDir
    Path tempDir;

    @Test
    void testReadSimpleSafeTensors() throws IOException {
        // Create a minimal SafeTensors file with a single 2x3 F32 tensor
        Path file = createSimpleSafeTensors();

        try (SafeTensors st = SafeTensors.open(file)) {
            assertEquals(1, st.tensorCount());
            assertTrue(st.hasTensor("test_tensor"));

            TensorInfo info = st.tensorInfo("test_tensor");
            assertEquals("test_tensor", info.name());
            assertEquals(DType.F32, info.dtype());
            assertArrayEquals(new long[]{2, 3}, info.shape());
            assertEquals(6, info.elementCount());

            TensorView view = st.tensor("test_tensor");
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
        Path file = createSimpleSafeTensors();

        try (SafeTensors st = SafeTensors.open(file)) {
            TensorView view = st.tensor("test_tensor");
            float[] array = view.toFloatArray();

            assertEquals(6, array.length);
            assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, array, 0.001f);
        }
    }

    @Test
    void testTensorSlice() throws IOException {
        Path file = createSimpleSafeTensors();

        try (SafeTensors st = SafeTensors.open(file)) {
            TensorView view = st.tensor("test_tensor");

            // Slice first row
            TensorView slice = view.slice(0, 1);
            assertEquals(1, slice.info().shape()[0]);
            assertEquals(3, slice.info().shape()[1]);

            float[] sliceData = slice.toFloatArray();
            assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f}, sliceData, 0.001f);
        }
    }

    @Test
    void testMetadata() throws IOException {
        Path file = createSafeTensorsWithMetadata();

        try (SafeTensors st = SafeTensors.open(file)) {
            var metadata = st.metadata();
            assertEquals("test", metadata.get("format"));
            assertEquals("1.0", metadata.get("version"));
        }
    }

    /**
     * Create a simple SafeTensors file with a 2x3 F32 tensor containing [1,2,3,4,5,6].
     */
    private Path createSimpleSafeTensors() throws IOException {
        // Header JSON
        String header = """
                {"test_tensor":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}""";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);

        // Tensor data: 6 floats
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        // Build the file
        Path file = tempDir.resolve("test.safetensors");

        try (Arena arena = Arena.ofConfined()) {
            long totalSize = 8 + headerBytes.length + (values.length * 4);
            MemorySegment buffer = arena.allocate(totalSize);

            // Header length (little-endian u64)
            buffer.set(ValueLayout.JAVA_LONG_UNALIGNED, 0, headerBytes.length);

            // Header bytes
            MemorySegment.copy(headerBytes, 0, buffer, ValueLayout.JAVA_BYTE, 8, headerBytes.length);

            // Tensor data (use unaligned access since header length may not be aligned)
            long dataOffset = 8 + headerBytes.length;
            for (int i = 0; i < values.length; i++) {
                buffer.set(ValueLayout.JAVA_FLOAT_UNALIGNED, dataOffset + (i * 4L), values[i]);
            }

            // Write to file
            byte[] fileBytes = buffer.toArray(ValueLayout.JAVA_BYTE);
            Files.write(file, fileBytes);
        }

        return file;
    }

    /**
     * Create a SafeTensors file with metadata.
     */
    private Path createSafeTensorsWithMetadata() throws IOException {
        // Header JSON with metadata
        String header = """
                {"__metadata__":{"format":"test","version":"1.0"},"test_tensor":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}""";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);

        float[] values = {1.0f, 2.0f};

        Path file = tempDir.resolve("test_meta.safetensors");

        try (Arena arena = Arena.ofConfined()) {
            long totalSize = 8 + headerBytes.length + (values.length * 4);
            MemorySegment buffer = arena.allocate(totalSize);

            buffer.set(ValueLayout.JAVA_LONG_UNALIGNED, 0, headerBytes.length);
            MemorySegment.copy(headerBytes, 0, buffer, ValueLayout.JAVA_BYTE, 8, headerBytes.length);

            long dataOffset = 8 + headerBytes.length;
            for (int i = 0; i < values.length; i++) {
                buffer.set(ValueLayout.JAVA_FLOAT_UNALIGNED, dataOffset + (i * 4L), values[i]);
            }

            byte[] fileBytes = buffer.toArray(ValueLayout.JAVA_BYTE);
            Files.write(file, fileBytes);
        }

        return file;
    }
}
