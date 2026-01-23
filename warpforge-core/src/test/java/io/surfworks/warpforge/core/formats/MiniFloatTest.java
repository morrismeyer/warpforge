package io.surfworks.warpforge.core.formats;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for P3109 small floating-point format implementation.
 */
@DisplayName("MiniFloat P3109 Implementation")
class MiniFloatTest {

    @Nested
    @DisplayName("FormatParameters")
    class FormatParametersTest {

        @Test
        @DisplayName("FP8 E5M2 parameters are correct")
        void fp8E5m2Parameters() {
            FormatParameters p = FormatParameters.FP8_E5M2;
            assertEquals(8, p.bitWidth());
            assertEquals(3, p.precision());
            assertEquals(5, p.exponentWidth());
            assertEquals(2, p.mantissaWidth());
            assertEquals(15, p.exponentBias());
            assertTrue(p.signed());
            assertFalse(p.finiteOnly());
            assertEquals("Binary8p3se", p.canonicalName());
            assertEquals("E5M2", p.shortName());
        }

        @Test
        @DisplayName("FP8 E4M3 parameters are correct")
        void fp8E4m3Parameters() {
            FormatParameters p = FormatParameters.FP8_E4M3;
            assertEquals(8, p.bitWidth());
            assertEquals(4, p.precision());
            assertEquals(4, p.exponentWidth());
            assertEquals(3, p.mantissaWidth());
            assertEquals(7, p.exponentBias());
            assertTrue(p.signed());
            assertFalse(p.finiteOnly());
        }

        @Test
        @DisplayName("FP4 E2M1 parameters are correct")
        void fp4E2m1Parameters() {
            FormatParameters p = FormatParameters.FP4_E2M1;
            assertEquals(4, p.bitWidth());
            assertEquals(2, p.precision());
            assertEquals(2, p.exponentWidth());
            assertEquals(1, p.mantissaWidth());
            assertEquals(1, p.exponentBias());
            assertTrue(p.signed());
            assertTrue(p.finiteOnly()); // Finite-only for ML use (MXFP4/NVFP4)
        }

        @Test
        @DisplayName("FP8 E8M0 (scale format) parameters are correct")
        void fp8E8m0Parameters() {
            FormatParameters p = FormatParameters.FP8_E8M0;
            assertEquals(8, p.bitWidth());
            assertEquals(1, p.precision());
            assertEquals(8, p.exponentWidth());
            assertEquals(0, p.mantissaWidth());
            assertEquals(127, p.exponentBias());
            assertFalse(p.signed());
            assertTrue(p.finiteOnly());
        }
    }

    @Nested
    @DisplayName("FP8 E5M2 Encoding/Decoding")
    class Float8E5M2Test {

        private static final FormatParameters PARAMS = FormatParameters.FP8_E5M2;

        @Test
        @DisplayName("Zero encodes to 0x00")
        void encodeZero() {
            assertEquals(0x00, MiniFloat.encode(0.0f, PARAMS));
            assertEquals(0x00, MiniFloat.encode(-0.0f, PARAMS)); // No negative zero in P3109
        }

        @Test
        @DisplayName("One encodes correctly")
        void encodeOne() {
            int bits = MiniFloat.encode(1.0f, PARAMS);
            float decoded = MiniFloat.decodeToFloat(bits, PARAMS);
            assertEquals(1.0f, decoded, 1e-6f);
        }

        @ParameterizedTest
        @ValueSource(floats = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f})
        @DisplayName("Powers of two round-trip correctly")
        void powersOfTwo(float value) {
            int bits = MiniFloat.encode(value, PARAMS);
            float decoded = MiniFloat.decodeToFloat(bits, PARAMS);
            assertEquals(value, decoded, 1e-6f);
        }

        @Test
        @DisplayName("NaN encodes to sign bit set")
        void encodeNaN() {
            int bits = MiniFloat.encode(Float.NaN, PARAMS);
            assertTrue(MiniFloat.isNaNBits(bits, PARAMS));
            assertEquals(0x80, bits); // P3109 NaN for signed: 2^(K-1)
        }

        @Test
        @DisplayName("Infinity encodes correctly")
        void encodeInfinity() {
            int posInfBits = MiniFloat.encode(Float.POSITIVE_INFINITY, PARAMS);
            int negInfBits = MiniFloat.encode(Float.NEGATIVE_INFINITY, PARAMS);

            assertTrue(MiniFloat.isInfinityBits(posInfBits, PARAMS));
            assertTrue(MiniFloat.isInfinityBits(negInfBits, PARAMS));
            assertFalse(MiniFloat.isNaNBits(posInfBits, PARAMS));
            assertFalse(MiniFloat.isNaNBits(negInfBits, PARAMS));
        }

        @Test
        @DisplayName("Negative values encode with sign bit")
        void encodeNegative() {
            int posBits = MiniFloat.encode(1.0f, PARAMS);
            int negBits = MiniFloat.encode(-1.0f, PARAMS);
            assertEquals(-1.0f, MiniFloat.decodeToFloat(negBits, PARAMS), 1e-6f);
            // Sign bit should be set (bit 7)
            assertEquals(0x80, (negBits ^ posBits) & 0x80);
        }

        @ParameterizedTest
        @CsvSource({
            "0.25, 0.25",
            "0.125, 0.125",
            "1.5, 1.5",
            "3.0, 3.0"
        })
        @DisplayName("Common values round-trip within precision")
        void commonValues(float input, float expected) {
            int bits = MiniFloat.encode(input, PARAMS);
            float decoded = MiniFloat.decodeToFloat(bits, PARAMS);
            assertEquals(expected, decoded, expected * 0.1f); // 10% tolerance
        }
    }

    @Nested
    @DisplayName("FP8 E4M3 Encoding/Decoding")
    class Float8E4M3Test {

        private static final FormatParameters PARAMS = FormatParameters.FP8_E4M3;

        @Test
        @DisplayName("Zero encodes to 0x00")
        void encodeZero() {
            assertEquals(0x00, MiniFloat.encode(0.0f, PARAMS));
        }

        @Test
        @DisplayName("One encodes correctly")
        void encodeOne() {
            int bits = MiniFloat.encode(1.0f, PARAMS);
            float decoded = MiniFloat.decodeToFloat(bits, PARAMS);
            assertEquals(1.0f, decoded, 1e-6f);
        }

        @Test
        @DisplayName("Higher precision than E5M2 near 1.0")
        void higherPrecision() {
            // E4M3 has 3 mantissa bits vs E5M2's 2, so better precision
            float value = 1.125f; // 1 + 1/8, exactly representable with 3+ mantissa bits
            int bits = MiniFloat.encode(value, PARAMS);
            float decoded = MiniFloat.decodeToFloat(bits, PARAMS);
            assertEquals(value, decoded, 1e-6f);
        }
    }

    @Nested
    @DisplayName("FP4 E2M1 Encoding/Decoding")
    class Float4E2M1Test {

        private static final FormatParameters PARAMS = FormatParameters.FP4_E2M1;

        @Test
        @DisplayName("All 16 code points decode correctly")
        void allCodePoints() {
            // E2M1 finite-only has exactly 16 finite values
            // Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6
            // Note: Code 8 in finite-only is negative zero (treated as 0)
            float[] expected = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, 0, -0.5f, -1, -1.5f, -2, -3, -4, -6};

            for (int code = 0; code < 16; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);
                // For finite-only formats, no NaN or infinity
                assertFalse(Float.isNaN(decoded), "Code " + code + " should not be NaN (finite-only)");
                assertFalse(Float.isInfinite(decoded), "Code " + code + " should not be Inf (finite-only)");
                assertEquals(expected[code], decoded, 0.01f, "Code " + code);
            }
        }

        @Test
        @DisplayName("Round-trip common values")
        void roundTrip() {
            float[] testValues = {0, 0.5f, 1, 1.5f, 2, 3, 4, -1, -2};
            for (float val : testValues) {
                int bits = MiniFloat.encode(val, PARAMS);
                float decoded = MiniFloat.decodeToFloat(bits, PARAMS);
                assertEquals(val, decoded, 0.01f, "Failed for " + val);
            }
        }
    }

    @Nested
    @DisplayName("Arithmetic Operations")
    class ArithmeticTest {

        private static final FormatParameters PARAMS = FormatParameters.FP8_E5M2;

        @Test
        @DisplayName("Add two values")
        void addition() {
            int a = MiniFloat.encode(1.0f, PARAMS);
            int b = MiniFloat.encode(2.0f, PARAMS);
            int sum = MiniFloat.add(a, b, PARAMS);
            float result = MiniFloat.decodeToFloat(sum, PARAMS);
            assertEquals(3.0f, result, 0.1f);
        }

        @Test
        @DisplayName("Subtract two values")
        void subtraction() {
            int a = MiniFloat.encode(5.0f, PARAMS);
            int b = MiniFloat.encode(2.0f, PARAMS);
            int diff = MiniFloat.subtract(a, b, PARAMS);
            float result = MiniFloat.decodeToFloat(diff, PARAMS);
            assertEquals(3.0f, result, 0.1f);
        }

        @Test
        @DisplayName("Multiply two values")
        void multiplication() {
            int a = MiniFloat.encode(2.0f, PARAMS);
            int b = MiniFloat.encode(3.0f, PARAMS);
            int prod = MiniFloat.multiply(a, b, PARAMS);
            float result = MiniFloat.decodeToFloat(prod, PARAMS);
            assertEquals(6.0f, result, 0.1f);
        }

        @Test
        @DisplayName("Divide two values")
        void division() {
            int a = MiniFloat.encode(6.0f, PARAMS);
            int b = MiniFloat.encode(2.0f, PARAMS);
            int quot = MiniFloat.divide(a, b, PARAMS);
            float result = MiniFloat.decodeToFloat(quot, PARAMS);
            assertEquals(3.0f, result, 0.1f);
        }

        @Test
        @DisplayName("Fused multiply-add")
        void fma() {
            int a = MiniFloat.encode(2.0f, PARAMS);
            int b = MiniFloat.encode(3.0f, PARAMS);
            int c = MiniFloat.encode(1.0f, PARAMS);
            int result = MiniFloat.fma(a, b, c, PARAMS);
            float decoded = MiniFloat.decodeToFloat(result, PARAMS);
            assertEquals(7.0f, decoded, 0.1f); // 2*3+1 = 7
        }

        @Test
        @DisplayName("Negate")
        void negate() {
            int pos = MiniFloat.encode(5.0f, PARAMS);
            int neg = MiniFloat.negate(pos, PARAMS);
            float result = MiniFloat.decodeToFloat(neg, PARAMS);
            assertEquals(-5.0f, result, 0.1f);
        }

        @Test
        @DisplayName("Absolute value")
        void absoluteValue() {
            int neg = MiniFloat.encode(-5.0f, PARAMS);
            int abs = MiniFloat.abs(neg, PARAMS);
            float result = MiniFloat.decodeToFloat(abs, PARAMS);
            assertEquals(5.0f, result, 0.1f);
        }

        @Test
        @DisplayName("Min and Max")
        void minMax() {
            int a = MiniFloat.encode(2.0f, PARAMS);
            int b = MiniFloat.encode(5.0f, PARAMS);

            assertEquals(2.0f, MiniFloat.decodeToFloat(MiniFloat.min(a, b, PARAMS), PARAMS), 0.1f);
            assertEquals(5.0f, MiniFloat.decodeToFloat(MiniFloat.max(a, b, PARAMS), PARAMS), 0.1f);
        }
    }

    @Nested
    @DisplayName("Bulk Operations")
    class BulkOperationsTest {

        @Test
        @DisplayName("Bulk encode/decode FP8")
        void bulkFp8() {
            FormatParameters params = FormatParameters.FP8_E5M2;
            float[] source = {0, 1, 2, 3, 4, 5, 6, 7, -1, -2};

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment dest = arena.allocate(MiniFloat.byteSize(source.length, params));
                MiniFloat.encodeBulk(source, dest, params);

                float[] decoded = new float[source.length];
                MiniFloat.decodeBulk(dest, decoded, params);

                for (int i = 0; i < source.length; i++) {
                    assertEquals(source[i], decoded[i], 0.1f, "Index " + i);
                }
            }
        }

        @Test
        @DisplayName("Bulk encode/decode FP4 (packed)")
        void bulkFp4() {
            FormatParameters params = FormatParameters.FP4_E2M1;
            float[] source = {0, 0.5f, 1, 1.5f, 2, 3, 4, -1, -2, -3};

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment dest = arena.allocate(MiniFloat.byteSize(source.length, params));
                MiniFloat.encodeBulk(source, dest, params);

                float[] decoded = new float[source.length];
                MiniFloat.decodeBulk(dest, decoded, params);

                for (int i = 0; i < source.length; i++) {
                    assertEquals(source[i], decoded[i], 0.1f, "Index " + i);
                }
            }
        }

        @Test
        @DisplayName("Bulk encode/decode FP6 (packed)")
        void bulkFp6() {
            FormatParameters params = FormatParameters.FP6_E3M2;
            // Use values within the FP6 E3M2 finite-only range
            float[] source = {0, 0.5f, 1, 2, 4, -1, -2, -4, 0.25f, 0.5f, 1.5f, 3};

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment dest = arena.allocate(MiniFloat.byteSize(source.length, params));
                MiniFloat.encodeBulk(source, dest, params);

                float[] decoded = new float[source.length];
                MiniFloat.decodeBulk(dest, decoded, params);

                for (int i = 0; i < source.length; i++) {
                    float tolerance = Math.max(0.2f, Math.abs(source[i]) * 0.3f);
                    assertEquals(source[i], decoded[i], tolerance, "Index " + i);
                }
            }
        }
    }

    @Nested
    @DisplayName("Cross-Format Conversion")
    class CrossFormatTest {

        @Test
        @DisplayName("Convert E5M2 to E4M3")
        void e5m2ToE4m3() {
            float value = 2.5f;
            int e5m2 = MiniFloat.encode(value, FormatParameters.FP8_E5M2);
            int e4m3 = MiniFloat.convert(e5m2, FormatParameters.FP8_E5M2, FormatParameters.FP8_E4M3);
            float decoded = MiniFloat.decodeToFloat(e4m3, FormatParameters.FP8_E4M3);
            assertEquals(value, decoded, 0.2f);
        }

        @Test
        @DisplayName("Convert FP4 to FP8")
        void fp4ToFp8() {
            float value = 1.5f;
            int fp4 = MiniFloat.encode(value, FormatParameters.FP4_E2M1);
            int fp8 = MiniFloat.convert(fp4, FormatParameters.FP4_E2M1, FormatParameters.FP8_E4M3);
            float decoded = MiniFloat.decodeToFloat(fp8, FormatParameters.FP8_E4M3);
            assertEquals(value, decoded, 0.1f);
        }
    }
}
