package io.surfworks.warpforge.core.io;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class NpyIOTest {

    @TempDir
    Path tempDir;

    @Nested
    @DisplayName("Round Trip Tests")
    class RoundTripTests {

        @Test
        void roundTrip1DTensor() throws IOException {
            float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            try (Tensor original = Tensor.fromFloatArray(data, 5)) {
                Path file = tempDir.resolve("test_1d.npy");

                // Write
                NpyIO.write(original, file);

                // Read back
                try (Tensor loaded = NpyIO.read(file)) {
                    assertArrayEquals(original.shape(), loaded.shape());
                    assertEquals(original.dtype(), loaded.dtype());
                    assertArrayEquals(data, loaded.toFloatArray(), 1e-9f);
                }
            }
        }

        @Test
        void roundTrip2DTensor() throws IOException {
            float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            try (Tensor original = Tensor.fromFloatArray(data, 2, 3)) {
                Path file = tempDir.resolve("test_2d.npy");

                NpyIO.write(original, file);

                try (Tensor loaded = NpyIO.read(file)) {
                    assertArrayEquals(new int[]{2, 3}, loaded.shape());
                    assertEquals(ScalarType.F32, loaded.dtype());
                    assertArrayEquals(data, loaded.toFloatArray(), 1e-9f);
                }
            }
        }

        @Test
        void roundTrip3DTensor() throws IOException {
            float[] data = new float[24];
            for (int i = 0; i < 24; i++) data[i] = i;

            try (Tensor original = Tensor.fromFloatArray(data, 2, 3, 4)) {
                Path file = tempDir.resolve("test_3d.npy");

                NpyIO.write(original, file);

                try (Tensor loaded = NpyIO.read(file)) {
                    assertArrayEquals(new int[]{2, 3, 4}, loaded.shape());
                    assertArrayEquals(data, loaded.toFloatArray(), 1e-9f);
                }
            }
        }

        @Test
        void roundTripDoubleTensor() throws IOException {
            double[] data = {1.0, 2.0, 3.0, 4.0};
            try (Tensor original = Tensor.fromDoubleArray(data, 2, 2)) {
                Path file = tempDir.resolve("test_double.npy");

                NpyIO.write(original, file);

                try (Tensor loaded = NpyIO.read(file)) {
                    assertEquals(ScalarType.F64, loaded.dtype());
                    assertArrayEquals(data, loaded.toDoubleArray(), 1e-15);
                }
            }
        }

        @Test
        void roundTripIntTensor() throws IOException {
            int[] data = {1, 2, 3, 4, 5, 6};
            try (Tensor original = Tensor.fromIntArray(data, 3, 2)) {
                Path file = tempDir.resolve("test_int.npy");

                NpyIO.write(original, file);

                try (Tensor loaded = NpyIO.read(file)) {
                    assertEquals(ScalarType.I32, loaded.dtype());
                    assertArrayEquals(data, loaded.toIntArray());
                }
            }
        }

        @Test
        void roundTripLargeTensor() throws IOException {
            int size = 1000;
            float[] data = new float[size];
            for (int i = 0; i < size; i++) data[i] = (float) Math.sin(i * 0.01);

            try (Tensor original = Tensor.fromFloatArray(data, 10, 100)) {
                Path file = tempDir.resolve("test_large.npy");

                NpyIO.write(original, file);

                try (Tensor loaded = NpyIO.read(file)) {
                    assertArrayEquals(new int[]{10, 100}, loaded.shape());
                    assertArrayEquals(data, loaded.toFloatArray(), 1e-6f);
                }
            }
        }
    }

    @Nested
    @DisplayName("Header Parsing")
    class HeaderParsingTests {

        @Test
        void parseSimpleHeader() {
            String header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
            NpyHeader parsed = NpyHeader.parse(1, 0, header);

            assertEquals(ScalarType.F32, parsed.dtype());
            assertEquals(ByteOrder.LITTLE_ENDIAN, parsed.byteOrder());
            assertFalse(parsed.fortranOrder());
            assertArrayEquals(new int[]{2, 3}, parsed.shape());
        }

        @Test
        void parseHeaderWithBigEndian() {
            String header = "{'descr': '>f8', 'fortran_order': False, 'shape': (10,), }";
            NpyHeader parsed = NpyHeader.parse(1, 0, header);

            assertEquals(ScalarType.F64, parsed.dtype());
            assertEquals(ByteOrder.BIG_ENDIAN, parsed.byteOrder());
        }

        @Test
        void parseHeaderWithFortranOrder() {
            String header = "{'descr': '<f4', 'fortran_order': True, 'shape': (3, 4), }";
            NpyHeader parsed = NpyHeader.parse(1, 0, header);

            assertTrue(parsed.fortranOrder());
        }

        @Test
        void parseScalarHeader() {
            String header = "{'descr': '<f4', 'fortran_order': False, 'shape': (), }";
            NpyHeader parsed = NpyHeader.parse(1, 0, header);

            assertArrayEquals(new int[0], parsed.shape());
            assertEquals(1, parsed.elementCount());
        }

        @Test
        void parse1DHeader() {
            // NumPy uses (n,) for 1D arrays
            String header = "{'descr': '<i4', 'fortran_order': False, 'shape': (5,), }";
            NpyHeader parsed = NpyHeader.parse(1, 0, header);

            assertArrayEquals(new int[]{5}, parsed.shape());
            assertEquals(5, parsed.elementCount());
        }

        @Test
        void headerToStringRoundTrip() {
            NpyHeader original = new NpyHeader(
                1, 0,
                ScalarType.F32,
                ByteOrder.LITTLE_ENDIAN,
                false,
                new int[]{2, 3, 4}
            );

            String headerStr = original.toHeaderString();
            NpyHeader parsed = NpyHeader.parse(1, 0, headerStr);

            assertEquals(original.dtype(), parsed.dtype());
            assertArrayEquals(original.shape(), parsed.shape());
            assertEquals(original.fortranOrder(), parsed.fortranOrder());
        }
    }

    @Nested
    @DisplayName("Stream I/O")
    class StreamIOTests {

        @Test
        void writeAndReadFromStream() throws IOException {
            float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
            try (Tensor original = Tensor.fromFloatArray(data, 2, 2)) {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                NpyIO.write(original, baos);

                byte[] bytes = baos.toByteArray();
                assertTrue(bytes.length > 0);

                ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
                try (Tensor loaded = NpyIO.read(bais)) {
                    assertArrayEquals(data, loaded.toFloatArray(), 1e-9f);
                }
            }
        }

        @Test
        void readHeaderFromStream() throws IOException {
            try (Tensor tensor = Tensor.fromFloatArray(new float[]{1, 2, 3, 4, 5, 6}, 2, 3)) {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                NpyIO.write(tensor, baos);

                ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
                NpyHeader header = NpyIO.readHeader(bais);

                assertEquals(ScalarType.F32, header.dtype());
                assertArrayEquals(new int[]{2, 3}, header.shape());
            }
        }
    }

    @Nested
    @DisplayName("File I/O")
    class FileIOTests {

        @Test
        void readHeaderFromFile() throws IOException {
            try (Tensor tensor = Tensor.fromFloatArray(new float[]{1, 2, 3, 4}, 4)) {
                Path file = tempDir.resolve("header_test.npy");
                NpyIO.write(tensor, file);

                NpyHeader header = NpyIO.readHeader(file);
                assertArrayEquals(new int[]{4}, header.shape());
                assertEquals(ScalarType.F32, header.dtype());
            }
        }
    }

    @Nested
    @DisplayName("Error Handling")
    class ErrorHandlingTests {

        @Test
        void invalidMagicNumberThrows() {
            byte[] invalidData = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
            ByteArrayInputStream bais = new ByteArrayInputStream(invalidData);

            assertThrows(IOException.class, () -> NpyIO.read(bais));
        }

        @Test
        void missingDescrThrows() {
            String badHeader = "{'fortran_order': False, 'shape': (2, 3), }";
            assertThrows(IllegalArgumentException.class, () ->
                NpyHeader.parse(1, 0, badHeader)
            );
        }

        @Test
        void missingShapeThrows() {
            String badHeader = "{'descr': '<f4', 'fortran_order': False}";
            assertThrows(IllegalArgumentException.class, () ->
                NpyHeader.parse(1, 0, badHeader)
            );
        }
    }

    @Test
    @DisplayName("NpyHeader needsByteSwap detects different endianness")
    void needsByteSwapDetectsEndianness() {
        NpyHeader littleEndian = new NpyHeader(1, 0, ScalarType.F32, ByteOrder.LITTLE_ENDIAN, false, new int[]{2});
        NpyHeader bigEndian = new NpyHeader(1, 0, ScalarType.F32, ByteOrder.BIG_ENDIAN, false, new int[]{2});

        // On little-endian systems (most modern systems), little-endian files don't need swap
        if (ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN) {
            assertFalse(littleEndian.needsByteSwap());
            assertTrue(bigEndian.needsByteSwap());
        } else {
            assertTrue(littleEndian.needsByteSwap());
            assertFalse(bigEndian.needsByteSwap());
        }

        // Single-byte types never need swapping
        NpyHeader singleByte = new NpyHeader(1, 0, ScalarType.I8, ByteOrder.BIG_ENDIAN, false, new int[]{2});
        assertFalse(singleByte.needsByteSwap());
    }
}
