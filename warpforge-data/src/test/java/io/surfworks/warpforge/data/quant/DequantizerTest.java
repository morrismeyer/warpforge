package io.surfworks.warpforge.data.quant;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DequantizerTest {

    @Nested
    class F32Tests {

        @Test
        void testPassthrough() {
            try (Arena arena = Arena.ofConfined()) {
                float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
                MemorySegment segment = arena.allocate(16);
                for (int i = 0; i < 4; i++) {
                    segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, input[i]);
                }

                Dequantizer dequant = Dequantizer.forType(QuantizationType.F32);
                float[] output = new float[4];
                dequant.dequantize(segment, output, 4);

                assertArrayEquals(input, output, 1e-6f);
            }
        }

        @Test
        void testBlockSize() {
            Dequantizer dequant = Dequantizer.forType(QuantizationType.F32);
            assertEquals(1, dequant.blockSize());
            assertEquals(4, dequant.bytesPerBlock());
        }
    }

    @Nested
    class F16Tests {

        @Test
        void testDequantization() {
            try (Arena arena = Arena.ofConfined()) {
                // Pack some FP16 values
                MemorySegment segment = arena.allocate(8);

                // 1.0f in FP16: sign=0, exp=15 (0b01111), mant=0 -> 0x3C00
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x3C00);
                // 2.0f in FP16: sign=0, exp=16 (0b10000), mant=0 -> 0x4000
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x4000);
                // 0.5f in FP16: sign=0, exp=14 (0b01110), mant=0 -> 0x3800
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 2, (short) 0x3800);
                // -1.0f in FP16: sign=1, exp=15, mant=0 -> 0xBC00
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 3, (short) 0xBC00);

                Dequantizer dequant = Dequantizer.forType(QuantizationType.F16);
                float[] output = new float[4];
                dequant.dequantize(segment, output, 4);

                assertEquals(1.0f, output[0], 1e-3f);
                assertEquals(2.0f, output[1], 1e-3f);
                assertEquals(0.5f, output[2], 1e-3f);
                assertEquals(-1.0f, output[3], 1e-3f);
            }
        }

        @Test
        void testSpecialValues() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment segment = arena.allocate(6);

                // Zero: 0x0000
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x0000);
                // Positive infinity: 0x7C00
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x7C00);
                // NaN: 0x7C01
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 2, (short) 0x7C01);

                Dequantizer dequant = Dequantizer.forType(QuantizationType.F16);
                float[] output = new float[3];
                dequant.dequantize(segment, output, 3);

                assertEquals(0.0f, output[0], 0f);
                assertTrue(Float.isInfinite(output[1]));
                assertTrue(Float.isNaN(output[2]));
            }
        }
    }

    @Nested
    class BF16Tests {

        @Test
        void testDequantization() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment segment = arena.allocate(4);

                // 1.0f in BF16: top 16 bits of 0x3F800000 = 0x3F80
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x3F80);
                // 2.0f in BF16: top 16 bits of 0x40000000 = 0x4000
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x4000);

                Dequantizer dequant = Dequantizer.forType(QuantizationType.BF16);
                float[] output = new float[2];
                dequant.dequantize(segment, output, 2);

                assertEquals(1.0f, output[0], 1e-2f);
                assertEquals(2.0f, output[1], 1e-2f);
            }
        }
    }

    @Nested
    class Q8_0Tests {

        @Test
        void testDequantization() {
            try (Arena arena = Arena.ofConfined()) {
                // Q8_0 block: 2 bytes scale (fp16) + 32 bytes data
                int blockSize = 32;
                int bytesPerBlock = 2 + blockSize;
                MemorySegment segment = arena.allocate(bytesPerBlock);

                // Scale = 0.1f in FP16 (approximately 0x2E66)
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x2E66);

                // Data: 0, 1, 2, ..., 31
                for (int i = 0; i < blockSize; i++) {
                    segment.set(ValueLayout.JAVA_BYTE, 2 + i, (byte) i);
                }

                Dequantizer dequant = Dequantizer.forType(QuantizationType.Q8_0);
                float[] output = new float[blockSize];
                dequant.dequantize(segment, output, blockSize);

                // Each value should be approximately i * 0.1
                for (int i = 0; i < blockSize; i++) {
                    assertEquals(i * 0.1f, output[i], 0.01f);
                }
            }
        }

        @Test
        void testBlockSize() {
            Dequantizer dequant = Dequantizer.forType(QuantizationType.Q8_0);
            assertEquals(32, dequant.blockSize());
            assertEquals(34, dequant.bytesPerBlock());
        }
    }

    @Nested
    class Q4_0Tests {

        @Test
        void testDequantization() {
            try (Arena arena = Arena.ofConfined()) {
                // Q4_0 block: 2 bytes scale (fp16) + 16 bytes data (32 x 4-bit)
                int bytesPerBlock = 2 + 16;
                MemorySegment segment = arena.allocate(bytesPerBlock);

                // Scale = 1.0f in FP16 (0x3C00)
                segment.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x3C00);

                // Data: pack values. Each byte has two 4-bit values.
                // Values are stored as unsigned 0-15, then subtracted by 8 to get -8 to 7
                // To get value 0 after dequant, we store 8 (since 8 - 8 = 0)
                for (int i = 0; i < 16; i++) {
                    // Lower nibble = 8 (gives 0), upper nibble = 9 (gives 1)
                    segment.set(ValueLayout.JAVA_BYTE, 2 + i, (byte) 0x98);
                }

                Dequantizer dequant = Dequantizer.forType(QuantizationType.Q4_0);
                float[] output = new float[32];
                dequant.dequantize(segment, output, 32);

                // All even indices should be 0 (8-8=0), odd indices should be 1 (9-8=1)
                for (int i = 0; i < 32; i += 2) {
                    assertEquals(0.0f, output[i], 0.01f);
                    assertEquals(1.0f, output[i + 1], 0.01f);
                }
            }
        }

        @Test
        void testBlockSize() {
            Dequantizer dequant = Dequantizer.forType(QuantizationType.Q4_0);
            assertEquals(32, dequant.blockSize());
            assertEquals(18, dequant.bytesPerBlock());
        }
    }

    @Nested
    class Int8Tests {

        @Test
        void testSymmetricDequantization() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment segment = arena.allocate(4);
                segment.set(ValueLayout.JAVA_BYTE, 0, (byte) 0);
                segment.set(ValueLayout.JAVA_BYTE, 1, (byte) 127);
                segment.set(ValueLayout.JAVA_BYTE, 2, (byte) -128);
                segment.set(ValueLayout.JAVA_BYTE, 3, (byte) 64);

                Dequantizer.Int8Dequantizer dequant = new Dequantizer.Int8Dequantizer(0.1f);
                float[] output = new float[4];
                dequant.dequantize(segment, output, 4);

                assertEquals(0.0f, output[0], 1e-6f);
                assertEquals(12.7f, output[1], 1e-3f);
                assertEquals(-12.8f, output[2], 1e-3f);
                assertEquals(6.4f, output[3], 1e-3f);
            }
        }
    }

    @Nested
    class Int4Tests {

        @Test
        void testSymmetricDequantization() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment segment = arena.allocate(2);
                // Pack: lower=0, upper=7 (both positive)
                segment.set(ValueLayout.JAVA_BYTE, 0, (byte) 0x70);
                // Pack: lower=8 (becomes -8), upper=15 (becomes -1)
                segment.set(ValueLayout.JAVA_BYTE, 1, (byte) 0xF8);

                Dequantizer.Int4Dequantizer dequant = new Dequantizer.Int4Dequantizer(1.0f);
                float[] output = new float[4];
                dequant.dequantize(segment, output, 4);

                assertEquals(0.0f, output[0], 1e-6f);
                assertEquals(7.0f, output[1], 1e-6f);
                assertEquals(-8.0f, output[2], 1e-6f);
                assertEquals(-1.0f, output[3], 1e-6f);
            }
        }
    }

    @Nested
    class FactoryTests {

        @Test
        void testForType() {
            assertEquals(QuantizationType.F32, Dequantizer.forType(QuantizationType.F32).type());
            assertEquals(QuantizationType.F16, Dequantizer.forType(QuantizationType.F16).type());
            assertEquals(QuantizationType.BF16, Dequantizer.forType(QuantizationType.BF16).type());
            assertEquals(QuantizationType.Q8_0, Dequantizer.forType(QuantizationType.Q8_0).type());
            assertEquals(QuantizationType.Q4_0, Dequantizer.forType(QuantizationType.Q4_0).type());
            assertEquals(QuantizationType.Q4_1, Dequantizer.forType(QuantizationType.Q4_1).type());
            assertEquals(QuantizationType.INT8, Dequantizer.forType(QuantizationType.INT8).type());
            assertEquals(QuantizationType.INT4, Dequantizer.forType(QuantizationType.INT4).type());
        }

        @Test
        void testUnsupportedType() {
            assertThrows(UnsupportedOperationException.class, () ->
                    Dequantizer.forType(QuantizationType.Q6_K));
        }
    }
}
