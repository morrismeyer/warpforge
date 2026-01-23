package io.surfworks.warpforge.core.formats;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for block-scaled floating-point formats (NVFP4, OCP MX).
 */
@DisplayName("Block-Scaled Formats")
class BlockFormatTest {

    @Nested
    @DisplayName("NVFP4 Block Format")
    class NvFp4BlockTest {

        @Test
        @DisplayName("Block size is 16")
        void blockSize() {
            assertEquals(16, NvFp4Block.INSTANCE.blockSize());
        }

        @Test
        @DisplayName("Byte size calculation")
        void byteSize() {
            // 4 bytes tensor scale + 9 bytes per block (8 for values, 1 for scale)
            assertEquals(4 + 9, NvFp4Block.INSTANCE.byteSize(16)); // 1 block
            assertEquals(4 + 18, NvFp4Block.INSTANCE.byteSize(32)); // 2 blocks
            assertEquals(4 + 9, NvFp4Block.INSTANCE.byteSize(1)); // Partial block
        }

        @Test
        @DisplayName("Round-trip small values")
        void roundTripSmall() {
            float[] source = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.5f, -1, -1.5f, -2, -3, -4, -6, 0};

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment encoded = arena.allocate(NvFp4Block.INSTANCE.byteSize(source.length));
                NvFp4Block.INSTANCE.encode(source, encoded);

                float[] decoded = new float[source.length];
                NvFp4Block.INSTANCE.decode(encoded, decoded);

                // With block scaling, relative error should be small
                for (int i = 0; i < source.length; i++) {
                    if (source[i] == 0) {
                        assertEquals(0, decoded[i], 0.1f, "Index " + i);
                    } else {
                        float relError = Math.abs(source[i] - decoded[i]) / Math.abs(source[i]);
                        assertTrue(relError < 0.3f, "Index " + i + ": " + source[i] + " vs " + decoded[i]);
                    }
                }
            }
        }

        @Test
        @DisplayName("Round-trip large values with scaling")
        void roundTripLarge() {
            float[] source = new float[32];
            for (int i = 0; i < 32; i++) {
                // Use moderate range that FP4 with scaling can handle
                source[i] = (i - 16) * 10.0f; // Range [-160, 150]
            }

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment encoded = arena.allocate(NvFp4Block.INSTANCE.byteSize(source.length));
                NvFp4Block.INSTANCE.encode(source, encoded);

                float[] decoded = new float[source.length];
                NvFp4Block.INSTANCE.decode(encoded, decoded);

                // Block scaling with FP4 has limited precision
                for (int i = 0; i < source.length; i++) {
                    if (Math.abs(source[i]) > 1) {
                        float relError = Math.abs(source[i] - decoded[i]) / Math.abs(source[i]);
                        assertTrue(relError < 0.5f, "Index " + i + ": " + source[i] + " vs " + decoded[i]);
                    } else {
                        // For small values, check absolute error
                        float absError = Math.abs(source[i] - decoded[i]);
                        assertTrue(absError < 5.0f, "Index " + i + ": " + source[i] + " vs " + decoded[i]);
                    }
                }
            }
        }

        @Test
        @DisplayName("Tensor scale is stored correctly")
        void tensorScale() {
            float[] source = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
                              -1000, -2000, -3000, -4000, -5000, -6000, -7000, -8000};

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment encoded = arena.allocate(NvFp4Block.INSTANCE.byteSize(source.length));
                NvFp4Block.INSTANCE.encode(source, encoded);

                float tensorScale = NvFp4Block.INSTANCE.getTensorScale(encoded);
                assertTrue(tensorScale > 0, "Tensor scale should be positive");

                float blockScale = NvFp4Block.INSTANCE.getBlockScale(encoded, 0);
                assertTrue(blockScale > 0, "Block scale should be positive");
            }
        }
    }

    @Nested
    @DisplayName("OCP MX Block Formats")
    class MxBlockTest {

        @Test
        @DisplayName("MXFP4 block size is 32")
        void mxfp4BlockSize() {
            assertEquals(32, MxBlock.MXFP4.blockSize());
        }

        @Test
        @DisplayName("MXFP8 E4M3 byte size calculation")
        void mxfp8E4m3ByteSize() {
            // 32 bytes for values + 1 byte for scale = 33 bytes per block
            assertEquals(33, MxBlock.MXFP8_E4M3.byteSize(32));
            assertEquals(66, MxBlock.MXFP8_E4M3.byteSize(64));
        }

        @Test
        @DisplayName("MXFP4 byte size calculation")
        void mxfp4ByteSize() {
            // 16 bytes for values (32 * 4 bits / 8) + 1 byte for scale = 17 bytes per block
            assertEquals(17, MxBlock.MXFP4.byteSize(32));
            assertEquals(34, MxBlock.MXFP4.byteSize(64));
        }

        @Test
        @DisplayName("MXFP4 round-trip")
        void mxfp4RoundTrip() {
            float[] source = new float[64];
            for (int i = 0; i < 64; i++) {
                source[i] = (i % 8) - 3.5f; // Small values that fit in FP4
            }

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment encoded = arena.allocate(MxBlock.MXFP4.byteSize(source.length));
                MxBlock.MXFP4.encode(source, encoded);

                float[] decoded = new float[source.length];
                MxBlock.MXFP4.decode(encoded, decoded);

                for (int i = 0; i < source.length; i++) {
                    float expected = source[i];
                    float actual = decoded[i];
                    // FP4 has limited precision, but with scaling should be reasonable
                    float tolerance = Math.max(0.5f, Math.abs(expected) * 0.3f);
                    assertEquals(expected, actual, tolerance, "Index " + i);
                }
            }
        }

        @Test
        @DisplayName("MXFP8 E4M3 round-trip")
        void mxfp8E4m3RoundTrip() {
            float[] source = new float[64];
            for (int i = 0; i < 64; i++) {
                // Use smaller values to stay within E4M3 range after scaling
                source[i] = (float) Math.sin(i * 0.1) * 5;
            }

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment encoded = arena.allocate(MxBlock.MXFP8_E4M3.byteSize(source.length));
                MxBlock.MXFP8_E4M3.encode(source, encoded);

                float[] decoded = new float[source.length];
                MxBlock.MXFP8_E4M3.decode(encoded, decoded);

                for (int i = 0; i < source.length; i++) {
                    float expected = source[i];
                    float actual = decoded[i];
                    // E4M3 has limited precision, so allow reasonable tolerance
                    float tolerance = Math.max(0.2f, Math.abs(expected) * 0.15f);
                    assertEquals(expected, actual, tolerance, "Index " + i);
                }
            }
        }

        @Test
        @DisplayName("MXFP8 E5M2 round-trip with wide range")
        void mxfp8E5m2RoundTrip() {
            float[] source = new float[64];
            Random rng = new Random(42);
            for (int i = 0; i < 64; i++) {
                // Limit exponent range to stay well within E5M2 capabilities after scaling
                source[i] = (float) Math.pow(2, rng.nextInt(10) - 5) * (rng.nextBoolean() ? 1 : -1);
            }

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment encoded = arena.allocate(MxBlock.MXFP8_E5M2.byteSize(source.length));
                MxBlock.MXFP8_E5M2.encode(source, encoded);

                float[] decoded = new float[source.length];
                MxBlock.MXFP8_E5M2.decode(encoded, decoded);

                for (int i = 0; i < source.length; i++) {
                    float expected = source[i];
                    float actual = decoded[i];
                    // E5M2 has wider range but lower precision
                    float tolerance = Math.max(0.5f, Math.abs(expected) * 0.3f);
                    assertEquals(expected, actual, tolerance, "Index " + i);
                }
            }
        }

        @Test
        @DisplayName("Scale exponent is retrieved correctly")
        void scaleExponent() {
            float[] source = new float[32];
            for (int i = 0; i < 32; i++) {
                source[i] = 1000 * (i + 1); // Large values requiring scaling
            }

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment encoded = arena.allocate(MxBlock.MXFP8_E4M3.byteSize(source.length));
                MxBlock.MXFP8_E4M3.encode(source, encoded);

                int exp = MxBlock.MXFP8_E4M3.getScaleExponent(encoded, 0);
                float scale = MxBlock.MXFP8_E4M3.getScale(encoded, 0);

                // Scale should be a power of 2
                assertEquals(Math.scalb(1.0f, exp), scale, 1e-6f);
                // Scale should be large enough to handle the max value
                assertTrue(scale > 1, "Scale should be > 1 for large values");
            }
        }
    }

    @Nested
    @DisplayName("Format Converter with Block Formats")
    class FormatConverterBlockTest {

        @Test
        @DisplayName("Measure NVFP4 quantization error")
        void nvfp4QuantizationError() {
            float[] source = new float[256];
            Random rng = new Random(123);
            for (int i = 0; i < source.length; i++) {
                source[i] = (float) (rng.nextGaussian() * 10);
            }

            FormatConverter.QuantizationStats stats =
                FormatConverter.measureBlockQuantizationError(source, NvFp4Block.INSTANCE);

            // NVFP4 with dual scaling should achieve reasonable SNR
            assertTrue(stats.snrDb() > 10, "SNR should be > 10 dB, got " + stats.snrDb());
        }

        @Test
        @DisplayName("Measure MXFP4 quantization error")
        void mxfp4QuantizationError() {
            float[] source = new float[256];
            Random rng = new Random(456);
            for (int i = 0; i < source.length; i++) {
                source[i] = (float) (rng.nextGaussian() * 10);
            }

            FormatConverter.QuantizationStats stats =
                FormatConverter.measureBlockQuantizationError(source, MxBlock.MXFP4);

            assertTrue(stats.snrDb() > 5, "SNR should be > 5 dB, got " + stats.snrDb());
        }

        @Test
        @DisplayName("Measure MXFP8 E4M3 quantization error")
        void mxfp8E4m3QuantizationError() {
            float[] source = new float[256];
            Random rng = new Random(789);
            for (int i = 0; i < source.length; i++) {
                // Use smaller values for better quantization
                source[i] = (float) (rng.nextGaussian() * 5);
            }

            FormatConverter.QuantizationStats stats =
                FormatConverter.measureBlockQuantizationError(source, MxBlock.MXFP8_E4M3);

            // FP8 with scaling should achieve reasonable SNR
            assertTrue(stats.snrDb() > 10, "SNR should be > 10 dB, got " + stats.snrDb());
        }
    }
}
