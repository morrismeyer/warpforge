package io.surfworks.warpforge.data.dataset;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ParquetDatasetTest {

    @TempDir
    Path tempDir;

    @Nested
    class ValidationTests {

        @Test
        void testInvalidMagicBytesAtStart() throws IOException {
            Path file = tempDir.resolve("invalid.parquet");
            Files.write(file, new byte[]{0, 0, 0, 0}); // Not "PAR1"

            assertThrows(IOException.class, () ->
                    ParquetDataset.load("test", file));
        }

        @Test
        void testInvalidMagicBytesAtEnd() throws IOException {
            Path file = tempDir.resolve("invalid_end.parquet");
            // Valid start magic but invalid end
            ByteBuffer buf = ByteBuffer.allocate(20);
            buf.put("PAR1".getBytes());
            buf.put(new byte[12]); // padding
            buf.put(new byte[]{0, 0, 0, 0}); // invalid end magic
            Files.write(file, buf.array());

            assertThrows(IOException.class, () ->
                    ParquetDataset.load("test", file));
        }

        @Test
        void testValidParquetStructure() throws IOException {
            // Create a minimal valid Parquet file structure
            Path file = tempDir.resolve("minimal.parquet");
            ByteBuffer buf = createMinimalParquetFile();
            Files.write(file, buf.array());

            // Should not throw - file has valid structure
            ParquetDataset dataset = ParquetDataset.load("test", file);

            assertNotNull(dataset);
            assertEquals("test", dataset.id());
        }
    }

    @Nested
    class DatasetSourceTests {

        @Test
        void testBasicProperties() throws IOException {
            Path file = tempDir.resolve("test.parquet");
            Files.write(file, createMinimalParquetFile().array());

            ParquetDataset dataset = ParquetDataset.load("my-dataset", file);

            assertEquals("my-dataset", dataset.id());
            assertTrue(dataset.splits().contains("default"));
            assertEquals(dataset, dataset.split("any")); // Returns self
        }

        @Test
        void testColumnNames() throws IOException {
            Path file = tempDir.resolve("test.parquet");
            Files.write(file, createMinimalParquetFile().array());

            ParquetDataset dataset = ParquetDataset.load("test", file);

            // Minimal parser creates placeholder column
            assertNotNull(dataset.columnNames());
        }
    }

    @Nested
    class IterationTests {

        @Test
        void testEmptyIteration() throws IOException {
            Path file = tempDir.resolve("empty.parquet");
            Files.write(file, createMinimalParquetFile().array());

            ParquetDataset dataset = ParquetDataset.load("test", file);

            // Minimal parser returns empty data for now
            assertEquals(0, dataset.size());
            assertEquals(0, dataset.stream().count());
        }

        @Test
        void testCloseIsNoOp() throws IOException {
            Path file = tempDir.resolve("test.parquet");
            Files.write(file, createMinimalParquetFile().array());

            ParquetDataset dataset = ParquetDataset.load("test", file);
            dataset.close(); // Should not throw
        }
    }

    /**
     * Creates a minimal valid Parquet file structure.
     * This is a bare-bones file with valid magic bytes and footer.
     */
    private ByteBuffer createMinimalParquetFile() {
        // Minimal Parquet file:
        // [4 bytes: PAR1 magic]
        // [N bytes: row groups (empty for minimal)]
        // [M bytes: footer metadata (simplified)]
        // [4 bytes: footer length]
        // [4 bytes: PAR1 magic]

        // Create a simplified footer (not valid Thrift, but has correct structure)
        byte[] footerMetadata = new byte[32]; // Minimal footer placeholder

        int totalSize = 4 + footerMetadata.length + 4 + 4;
        ByteBuffer buf = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN);

        // Start magic
        buf.put("PAR1".getBytes());

        // Footer metadata
        buf.put(footerMetadata);

        // Footer length
        buf.putInt(footerMetadata.length);

        // End magic
        buf.put("PAR1".getBytes());

        return buf;
    }
}
