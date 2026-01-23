package io.surfworks.warpforge.core.formats;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exhaustive tests for all P3109 floating-point formats.
 *
 * <p>These tests verify every single code point for each format to ensure
 * correct encoding and decoding behavior. The expected values are computed
 * directly from the P3109 specification formulas.
 *
 * <h2>P3109 Value Computation</h2>
 * <pre>
 * For a K-bit format with P precision bits:
 * - E = K - P (exponent width for signed)
 * - M = P - 1 (mantissa width)
 * - bias = 2^(E-1) - 1
 *
 * Subnormal (biased_exp = 0):
 *   value = 2^(1-bias) × (0.mantissa)
 *
 * Normal (biased_exp > 0):
 *   value = 2^(biased_exp - bias) × (1.mantissa)
 * </pre>
 */
@DisplayName("Exhaustive Format Tests")
class ExhaustiveFormatTest {

    // ==================== FP4 E2M1 (16 values) ====================

    @Nested
    @DisplayName("FP4 E2M1 - All 16 Code Points")
    class Fp4E2m1ExhaustiveTest {

        private static final FormatParameters PARAMS = FormatParameters.FP4_E2M1;

        /**
         * FP4 E2M1 Complete Value Table (finite-only):
         *
         * Code | Binary | Sign | Exp | Mant | Formula                  | Value
         * -----|--------|------|-----|------|--------------------------|-------
         *  0   | 0000   |  0   |  0  |   0  | 0                        |  0.0
         *  1   | 0001   |  0   |  0  |   1  | 2^0 × 0.5 = 0.5          |  0.5
         *  2   | 0010   |  0   |  1  |   0  | 2^0 × 1.0 = 1.0          |  1.0
         *  3   | 0011   |  0   |  1  |   1  | 2^0 × 1.5 = 1.5          |  1.5
         *  4   | 0100   |  0   |  2  |   0  | 2^1 × 1.0 = 2.0          |  2.0
         *  5   | 0101   |  0   |  2  |   1  | 2^1 × 1.5 = 3.0          |  3.0
         *  6   | 0110   |  0   |  3  |   0  | 2^2 × 1.0 = 4.0          |  4.0
         *  7   | 0111   |  0   |  3  |   1  | 2^2 × 1.5 = 6.0          |  6.0
         *  8   | 1000   |  1   |  0  |   0  | -0 (= 0 in P3109)        |  0.0
         *  9   | 1001   |  1   |  0  |   1  | -(2^0 × 0.5) = -0.5      | -0.5
         * 10   | 1010   |  1   |  1  |   0  | -(2^0 × 1.0) = -1.0      | -1.0
         * 11   | 1011   |  1   |  1  |   1  | -(2^0 × 1.5) = -1.5      | -1.5
         * 12   | 1100   |  1   |  2  |   0  | -(2^1 × 1.0) = -2.0      | -2.0
         * 13   | 1101   |  1   |  2  |   1  | -(2^1 × 1.5) = -3.0      | -3.0
         * 14   | 1110   |  1   |  3  |   0  | -(2^2 × 1.0) = -4.0      | -4.0
         * 15   | 1111   |  1   |  3  |   1  | -(2^2 × 1.5) = -6.0      | -6.0
         */
        @ParameterizedTest(name = "Code {0} (0b{1}) = {2}")
        @CsvSource({
            // Positive values
            "0,  0000,  0.0",
            "1,  0001,  0.5",
            "2,  0010,  1.0",
            "3,  0011,  1.5",
            "4,  0100,  2.0",
            "5,  0101,  3.0",
            "6,  0110,  4.0",
            "7,  0111,  6.0",
            // Negative values (code 8 is -0 = 0 in P3109)
            "8,  1000,  0.0",
            "9,  1001, -0.5",
            "10, 1010, -1.0",
            "11, 1011, -1.5",
            "12, 1100, -2.0",
            "13, 1101, -3.0",
            "14, 1110, -4.0",
            "15, 1111, -6.0"
        })
        @DisplayName("Decode")
        void decodeAllCodePoints(int code, String binary, float expected) {
            float actual = MiniFloat.decodeToFloat(code, PARAMS);
            assertEquals(expected, actual, 1e-6f,
                "Code " + code + " (0b" + binary + ") should decode to " + expected);
        }

        @ParameterizedTest(name = "{0} -> Code {1}")
        @CsvSource({
            // Exact values encode to their code points
            " 0.0,  0",
            " 0.5,  1",
            " 1.0,  2",
            " 1.5,  3",
            " 2.0,  4",
            " 3.0,  5",
            " 4.0,  6",
            " 6.0,  7",
            "-0.5,  9",
            "-1.0, 10",
            "-1.5, 11",
            "-2.0, 12",
            "-3.0, 13",
            "-4.0, 14",
            "-6.0, 15"
        })
        @DisplayName("Encode exact values")
        void encodeExactValues(float value, int expectedCode) {
            int actual = MiniFloat.encode(value, PARAMS);
            assertEquals(expectedCode, actual,
                value + " should encode to code " + expectedCode);
        }

        @ParameterizedTest(name = "{0} rounds to {1}")
        @CsvSource({
            // Values between representable points should round to nearest
            " 0.25,  1",   // 0.25 -> 0.5 (nearest)
            " 0.74,  1",   // 0.74 -> 0.5 (nearest)
            " 0.76,  2",   // 0.76 -> 1.0 (nearest)
            " 1.25,  3",   // 1.25 -> 1.5 (nearest)
            " 1.74,  3",   // 1.74 -> 1.5 (nearest)
            " 1.76,  4",   // 1.76 -> 2.0 (nearest)
            " 2.5,   5",   // 2.5  -> 3.0 (nearest)
            " 3.5,   6",   // 3.5  -> 4.0 (nearest)
            " 5.0,   7",   // 5.0  -> 6.0 (nearest)
            "-0.25,  9",   // -0.25 -> -0.5 (nearest)
            "-0.76, 10",   // -0.76 -> -1.0 (nearest)
            "-2.5,  13",   // -2.5  -> -3.0 (nearest)
        })
        @DisplayName("Encode with rounding")
        void encodeWithRounding(float value, int expectedCode) {
            int actual = MiniFloat.encode(value, PARAMS);
            assertEquals(expectedCode, actual,
                value + " should round to code " + expectedCode);
        }

        @Test
        @DisplayName("All codes round-trip correctly")
        void roundTripAllCodes() {
            for (int code = 0; code < 16; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);
                if (code == 8) {
                    // Code 8 is -0, decodes to 0, re-encodes to 0
                    assertEquals(0, MiniFloat.encode(decoded, PARAMS),
                        "Code 8 (-0) should round-trip through 0");
                } else {
                    int reencoded = MiniFloat.encode(decoded, PARAMS);
                    assertEquals(code, reencoded,
                        "Code " + code + " (value " + decoded + ") should round-trip");
                }
            }
        }

        @Test
        @DisplayName("No NaN or Infinity in finite-only format")
        void noSpecialValues() {
            for (int code = 0; code < 16; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);
                assertFalse(Float.isNaN(decoded), "Code " + code + " should not be NaN");
                assertFalse(Float.isInfinite(decoded), "Code " + code + " should not be Inf");
            }
        }

        @Test
        @DisplayName("Overflow clamps to max")
        void overflowClampsToMax() {
            // Values > 6.0 should clamp to 6.0 (code 7)
            assertEquals(7, MiniFloat.encode(7.0f, PARAMS));
            assertEquals(7, MiniFloat.encode(100.0f, PARAMS));
            assertEquals(7, MiniFloat.encode(Float.MAX_VALUE, PARAMS));
            // Negative overflow
            assertEquals(15, MiniFloat.encode(-7.0f, PARAMS));
            assertEquals(15, MiniFloat.encode(-100.0f, PARAMS));
        }

        @Test
        @DisplayName("Underflow rounds to zero or min subnormal")
        void underflowBehavior() {
            // Values < 0.25 should round to 0
            assertEquals(0, MiniFloat.encode(0.1f, PARAMS));
            assertEquals(0, MiniFloat.encode(0.24f, PARAMS));
            // Values >= 0.25 should round to 0.5
            assertEquals(1, MiniFloat.encode(0.25f, PARAMS));
            assertEquals(1, MiniFloat.encode(0.26f, PARAMS));
        }

        @Test
        @DisplayName("NaN input encodes to zero (finite-only)")
        void nanEncodesToZero() {
            assertEquals(0, MiniFloat.encode(Float.NaN, PARAMS));
        }

        @Test
        @DisplayName("Infinity input clamps to max (finite-only)")
        void infinityClampsToMax() {
            assertEquals(7, MiniFloat.encode(Float.POSITIVE_INFINITY, PARAMS));
            assertEquals(15, MiniFloat.encode(Float.NEGATIVE_INFINITY, PARAMS));
        }
    }

    // ==================== FP4 E1M2 (16 values) ====================

    @Nested
    @DisplayName("FP4 E1M2 - All 16 Code Points")
    class Fp4E1m2ExhaustiveTest {

        private static final FormatParameters PARAMS = FormatParameters.FP4_E1M2;

        /**
         * FP4 E1M2 Complete Value Table (finite-only):
         *
         * E1M2: 1-bit exponent, 2-bit mantissa, bias = 0
         *
         * Subnormal (exp=0): value = 2^(1-0) × (0.mm) = 2 × (mm/4) = mm/2
         * Normal (exp=1): value = 2^(1-0) × (1.mm) = 2 × (1 + mm/4)
         *
         * Code | Binary | Sign | Exp | Mant | Value
         * -----|--------|------|-----|------|-------
         *  0   | 0000   |  0   |  0  |  00  |  0.0
         *  1   | 0001   |  0   |  0  |  01  |  0.5
         *  2   | 0010   |  0   |  0  |  10  |  1.0
         *  3   | 0011   |  0   |  0  |  11  |  1.5
         *  4   | 0100   |  0   |  1  |  00  |  2.0
         *  5   | 0101   |  0   |  1  |  01  |  2.5
         *  6   | 0110   |  0   |  1  |  10  |  3.0
         *  7   | 0111   |  0   |  1  |  11  |  3.5
         *  8   | 1000   |  1   |  0  |  00  |  0.0 (-0)
         *  9   | 1001   |  1   |  0  |  01  | -0.5
         * 10   | 1010   |  1   |  0  |  10  | -1.0
         * 11   | 1011   |  1   |  0  |  11  | -1.5
         * 12   | 1100   |  1   |  1  |  00  | -2.0
         * 13   | 1101   |  1   |  1  |  01  | -2.5
         * 14   | 1110   |  1   |  1  |  10  | -3.0
         * 15   | 1111   |  1   |  1  |  11  | -3.5
         */
        @ParameterizedTest(name = "Code {0} (0b{1}) = {2}")
        @CsvSource({
            "0,  0000,  0.0",
            "1,  0001,  0.5",
            "2,  0010,  1.0",
            "3,  0011,  1.5",
            "4,  0100,  2.0",
            "5,  0101,  2.5",
            "6,  0110,  3.0",
            "7,  0111,  3.5",
            "8,  1000,  0.0",
            "9,  1001, -0.5",
            "10, 1010, -1.0",
            "11, 1011, -1.5",
            "12, 1100, -2.0",
            "13, 1101, -2.5",
            "14, 1110, -3.0",
            "15, 1111, -3.5"
        })
        @DisplayName("Decode")
        void decodeAllCodePoints(int code, String binary, float expected) {
            float actual = MiniFloat.decodeToFloat(code, PARAMS);
            assertEquals(expected, actual, 1e-6f,
                "Code " + code + " (0b" + binary + ") should decode to " + expected);
        }

        @ParameterizedTest(name = "{0} -> Code {1}")
        @CsvSource({
            " 0.0,  0",
            " 0.5,  1",
            " 1.0,  2",
            " 1.5,  3",
            " 2.0,  4",
            " 2.5,  5",
            " 3.0,  6",
            " 3.5,  7",
            "-0.5,  9",
            "-1.0, 10",
            "-1.5, 11",
            "-2.0, 12",
            "-2.5, 13",
            "-3.0, 14",
            "-3.5, 15"
        })
        @DisplayName("Encode exact values")
        void encodeExactValues(float value, int expectedCode) {
            int actual = MiniFloat.encode(value, PARAMS);
            assertEquals(expectedCode, actual,
                value + " should encode to code " + expectedCode);
        }

        @Test
        @DisplayName("All codes round-trip correctly")
        void roundTripAllCodes() {
            for (int code = 0; code < 16; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);
                if (code == 8) {
                    assertEquals(0, MiniFloat.encode(decoded, PARAMS));
                } else {
                    assertEquals(code, MiniFloat.encode(decoded, PARAMS),
                        "Code " + code + " should round-trip");
                }
            }
        }

        @Test
        @DisplayName("No NaN or Infinity in finite-only format")
        void noSpecialValues() {
            for (int code = 0; code < 16; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);
                assertFalse(Float.isNaN(decoded), "Code " + code + " should not be NaN");
                assertFalse(Float.isInfinite(decoded), "Code " + code + " should not be Inf");
            }
        }
    }

    // ==================== FP6 E3M2 (64 values) ====================

    @Nested
    @DisplayName("FP6 E3M2 - All 64 Code Points")
    class Fp6E3m2ExhaustiveTest {

        private static final FormatParameters PARAMS = FormatParameters.FP6_E3M2;

        /**
         * FP6 E3M2: 3-bit exponent, 2-bit mantissa, bias = 3
         *
         * Subnormal (exp=0): value = 2^(1-3) × (0.mm) = 0.25 × (mm/4)
         * Normal (exp=1-7): value = 2^(exp-3) × (1.mm)
         *
         * Max value (exp=7, mant=11): 2^4 × 1.75 = 28.0
         */
        @Test
        @DisplayName("Decode all 64 positive code points")
        void decodeAllPositiveCodes() {
            // Verify computed values match expected for all positive codes (0-31)
            float[] expected = computeE3M2Values(false);
            for (int code = 0; code < 32; code++) {
                float actual = MiniFloat.decodeToFloat(code, PARAMS);
                assertEquals(expected[code], actual, 1e-6f,
                    "Code " + code + " should decode to " + expected[code]);
            }
        }

        @Test
        @DisplayName("Decode all 64 negative code points")
        void decodeAllNegativeCodes() {
            float[] positiveExpected = computeE3M2Values(false);
            for (int code = 32; code < 64; code++) {
                float actual = MiniFloat.decodeToFloat(code, PARAMS);
                int positiveCode = code - 32;
                float expected = (positiveCode == 0) ? 0.0f : -positiveExpected[positiveCode];
                assertEquals(expected, actual, 1e-6f,
                    "Code " + code + " should decode to " + expected);
            }
        }

        private float[] computeE3M2Values(boolean negative) {
            // E3M2: bias=3, subnormal scale = 2^(1-3) = 0.25
            float[] values = new float[32];
            values[0] = 0.0f; // Zero

            // Subnormals (exp=0, mant=1,2,3)
            values[1] = 0.25f * (1.0f / 4.0f);  // 0.0625
            values[2] = 0.25f * (2.0f / 4.0f);  // 0.125
            values[3] = 0.25f * (3.0f / 4.0f);  // 0.1875

            // Normals (exp=1-7)
            for (int exp = 1; exp <= 7; exp++) {
                float scale = (float) Math.pow(2, exp - 3);
                for (int mant = 0; mant < 4; mant++) {
                    int code = (exp << 2) | mant;
                    values[code] = scale * (1.0f + mant / 4.0f);
                }
            }
            return values;
        }

        @Test
        @DisplayName("Key values decode correctly")
        void keyValuesDecodeCorrectly() {
            // Zero
            assertEquals(0.0f, MiniFloat.decodeToFloat(0, PARAMS), 1e-6f);

            // Smallest subnormal: 0.0625
            assertEquals(0.0625f, MiniFloat.decodeToFloat(1, PARAMS), 1e-6f);

            // One (exp=3, mant=0): 2^0 × 1.0 = 1.0
            assertEquals(1.0f, MiniFloat.decodeToFloat(0b001100, PARAMS), 1e-6f);

            // Max positive (exp=7, mant=3): 2^4 × 1.75 = 28.0
            assertEquals(28.0f, MiniFloat.decodeToFloat(0b011111, PARAMS), 1e-6f);

            // Max negative
            assertEquals(-28.0f, MiniFloat.decodeToFloat(0b111111, PARAMS), 1e-6f);
        }

        @Test
        @DisplayName("All codes round-trip correctly")
        void roundTripAllCodes() {
            for (int code = 0; code < 64; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);
                if (code == 32) {
                    // -0 encodes as 0
                    assertEquals(0, MiniFloat.encode(decoded, PARAMS));
                } else {
                    assertEquals(code, MiniFloat.encode(decoded, PARAMS),
                        "Code " + code + " (value " + decoded + ") should round-trip");
                }
            }
        }

        @Test
        @DisplayName("No NaN or Infinity in finite-only format")
        void noSpecialValues() {
            for (int code = 0; code < 64; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);
                assertFalse(Float.isNaN(decoded), "Code " + code + " should not be NaN");
                assertFalse(Float.isInfinite(decoded), "Code " + code + " should not be Inf");
            }
        }
    }

    // ==================== FP6 E2M3 (64 values) ====================

    @Nested
    @DisplayName("FP6 E2M3 - All 64 Code Points")
    class Fp6E2m3ExhaustiveTest {

        private static final FormatParameters PARAMS = FormatParameters.FP6_E2M3;

        /**
         * FP6 E2M3: 2-bit exponent, 3-bit mantissa, bias = 1
         *
         * Subnormal (exp=0): value = 2^(1-1) × (0.mmm) = mmm/8
         * Normal (exp=1-3): value = 2^(exp-1) × (1.mmm)
         *
         * Max value (exp=3, mant=111): 2^2 × 1.875 = 7.5
         */
        @Test
        @DisplayName("Decode all 64 positive code points")
        void decodeAllPositiveCodes() {
            float[] expected = computeE2M3Values();
            for (int code = 0; code < 32; code++) {
                float actual = MiniFloat.decodeToFloat(code, PARAMS);
                assertEquals(expected[code], actual, 1e-6f,
                    "Code " + code + " should decode to " + expected[code]);
            }
        }

        @Test
        @DisplayName("Decode all 64 negative code points")
        void decodeAllNegativeCodes() {
            float[] positiveExpected = computeE2M3Values();
            for (int code = 32; code < 64; code++) {
                float actual = MiniFloat.decodeToFloat(code, PARAMS);
                int positiveCode = code - 32;
                float expected = (positiveCode == 0) ? 0.0f : -positiveExpected[positiveCode];
                assertEquals(expected, actual, 1e-6f,
                    "Code " + code + " should decode to " + expected);
            }
        }

        private float[] computeE2M3Values() {
            // E2M3: bias=1, subnormal scale = 2^(1-1) = 1.0
            float[] values = new float[32];
            values[0] = 0.0f;

            // Subnormals (exp=0, mant=1-7)
            for (int mant = 1; mant < 8; mant++) {
                values[mant] = mant / 8.0f;
            }

            // Normals (exp=1-3)
            for (int exp = 1; exp <= 3; exp++) {
                float scale = (float) Math.pow(2, exp - 1);
                for (int mant = 0; mant < 8; mant++) {
                    int code = (exp << 3) | mant;
                    values[code] = scale * (1.0f + mant / 8.0f);
                }
            }
            return values;
        }

        @Test
        @DisplayName("Key values decode correctly")
        void keyValuesDecodeCorrectly() {
            // Zero
            assertEquals(0.0f, MiniFloat.decodeToFloat(0, PARAMS), 1e-6f);

            // Smallest subnormal: 0.125
            assertEquals(0.125f, MiniFloat.decodeToFloat(1, PARAMS), 1e-6f);

            // One (exp=1, mant=0): 2^0 × 1.0 = 1.0
            assertEquals(1.0f, MiniFloat.decodeToFloat(0b01000, PARAMS), 1e-6f);

            // Max positive (exp=3, mant=7): 2^2 × 1.875 = 7.5
            assertEquals(7.5f, MiniFloat.decodeToFloat(0b11111, PARAMS), 1e-6f);

            // Max negative
            assertEquals(-7.5f, MiniFloat.decodeToFloat(0b111111, PARAMS), 1e-6f);
        }

        @Test
        @DisplayName("All codes round-trip correctly")
        void roundTripAllCodes() {
            for (int code = 0; code < 64; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);
                if (code == 32) {
                    assertEquals(0, MiniFloat.encode(decoded, PARAMS));
                } else {
                    assertEquals(code, MiniFloat.encode(decoded, PARAMS),
                        "Code " + code + " (value " + decoded + ") should round-trip");
                }
            }
        }

        @Test
        @DisplayName("No NaN or Infinity in finite-only format")
        void noSpecialValues() {
            for (int code = 0; code < 64; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);
                assertFalse(Float.isNaN(decoded), "Code " + code + " should not be NaN");
                assertFalse(Float.isInfinite(decoded), "Code " + code + " should not be Inf");
            }
        }
    }

    // ==================== FP8 E5M2 (256 values) ====================

    @Nested
    @DisplayName("FP8 E5M2 - All 256 Code Points")
    class Fp8E5m2ExhaustiveTest {

        private static final FormatParameters PARAMS = FormatParameters.FP8_E5M2;

        /**
         * FP8 E5M2: 5-bit exponent, 2-bit mantissa, bias = 15
         *
         * Extended format (has NaN and Inf):
         * - NaN: code 0x80 (sign=1, exp=0, mant=0)
         * - +Inf: exp=31, mant=3 (code 0x7F)
         * - -Inf: sign=1, exp=31, mant=3 (code 0xFF)
         *
         * Max finite: exp=30, mant=3 = 2^15 × 1.75 = 57344
         */
        @Test
        @DisplayName("Decode all 256 code points")
        void decodeAllCodes() {
            for (int code = 0; code < 256; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);

                // Special values
                if (code == 0x80) {
                    assertTrue(Float.isNaN(decoded), "Code 0x80 should be NaN");
                } else if (code == 0x7F) {
                    assertEquals(Float.POSITIVE_INFINITY, decoded, "Code 0x7F should be +Inf");
                } else if (code == 0xFF) {
                    assertEquals(Float.NEGATIVE_INFINITY, decoded, "Code 0xFF should be -Inf");
                } else {
                    // Normal/subnormal - verify it's finite
                    assertTrue(Float.isFinite(decoded),
                        "Code " + code + " should be finite, got " + decoded);
                }
            }
        }

        @Test
        @DisplayName("Key values decode correctly")
        void keyValuesDecodeCorrectly() {
            // Zero
            assertEquals(0.0f, MiniFloat.decodeToFloat(0x00, PARAMS), 1e-10f);

            // One: exp=15, mant=0 -> 2^0 × 1.0 = 1.0
            // Code = 0_01111_00 = 0x3C
            assertEquals(1.0f, MiniFloat.decodeToFloat(0x3C, PARAMS), 1e-6f);

            // Two: exp=16, mant=0 -> 2^1 × 1.0 = 2.0
            // Code = 0_10000_00 = 0x40
            assertEquals(2.0f, MiniFloat.decodeToFloat(0x40, PARAMS), 1e-6f);

            // 0.5: exp=14, mant=0 -> 2^(-1) × 1.0 = 0.5
            // Code = 0_01110_00 = 0x38
            assertEquals(0.5f, MiniFloat.decodeToFloat(0x38, PARAMS), 1e-6f);

            // NaN
            assertTrue(Float.isNaN(MiniFloat.decodeToFloat(0x80, PARAMS)));

            // +Inf
            assertEquals(Float.POSITIVE_INFINITY, MiniFloat.decodeToFloat(0x7F, PARAMS));

            // -Inf
            assertEquals(Float.NEGATIVE_INFINITY, MiniFloat.decodeToFloat(0xFF, PARAMS));
        }

        @Test
        @DisplayName("All finite codes round-trip correctly")
        void roundTripAllFiniteCodes() {
            for (int code = 0; code < 256; code++) {
                // Skip NaN (doesn't round-trip by identity) and infinities
                if (code == 0x80 || code == 0x7F || code == 0xFF) continue;

                float decoded = MiniFloat.decodeToFloat(code, PARAMS);

                // Skip -0 which encodes as +0
                if (code == 0x80) continue;

                int reencoded = MiniFloat.encode(decoded, PARAMS);

                // Special case: negative zero (code with sign bit but zero value)
                // should re-encode as positive zero
                if (decoded == 0.0f) {
                    assertEquals(0, reencoded, "Zero should encode to code 0");
                } else {
                    assertEquals(code, reencoded,
                        "Code " + code + " (0x" + Integer.toHexString(code) +
                        ", value " + decoded + ") should round-trip");
                }
            }
        }

        @Test
        @DisplayName("Subnormal values are correct")
        void subnormalValues() {
            // Subnormals: exp=0, mant=1,2,3
            // Scale = 2^(1-15) = 2^(-14)
            float scale = (float) Math.pow(2, -14);

            assertEquals(scale * 0.25f, MiniFloat.decodeToFloat(0x01, PARAMS), 1e-10f);
            assertEquals(scale * 0.5f, MiniFloat.decodeToFloat(0x02, PARAMS), 1e-10f);
            assertEquals(scale * 0.75f, MiniFloat.decodeToFloat(0x03, PARAMS), 1e-10f);
        }

        @Test
        @DisplayName("Powers of two encode correctly")
        void powersOfTwo() {
            // Test a range of powers of 2
            for (int exp = -14; exp <= 15; exp++) {
                float value = (float) Math.pow(2, exp);
                int bits = MiniFloat.encode(value, PARAMS);
                float decoded = MiniFloat.decodeToFloat(bits, PARAMS);
                assertEquals(value, decoded, value * 1e-6f,
                    "2^" + exp + " should round-trip correctly");
            }
        }
    }

    // ==================== FP8 E4M3 (256 values) ====================

    @Nested
    @DisplayName("FP8 E4M3 - All 256 Code Points")
    class Fp8E4m3ExhaustiveTest {

        private static final FormatParameters PARAMS = FormatParameters.FP8_E4M3;

        /**
         * FP8 E4M3: 4-bit exponent, 3-bit mantissa, bias = 7
         *
         * Extended format (has NaN and Inf):
         * - NaN: code 0x80
         * - +Inf: exp=15, mant=7 (code 0x7F)
         * - -Inf: sign=1, exp=15, mant=7 (code 0xFF)
         *
         * Max finite: exp=14, mant=7 = 2^7 × 1.875 = 240
         */
        @Test
        @DisplayName("Decode all 256 code points")
        void decodeAllCodes() {
            for (int code = 0; code < 256; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);

                if (code == 0x80) {
                    assertTrue(Float.isNaN(decoded), "Code 0x80 should be NaN");
                } else if (code == 0x7F) {
                    assertEquals(Float.POSITIVE_INFINITY, decoded, "Code 0x7F should be +Inf");
                } else if (code == 0xFF) {
                    assertEquals(Float.NEGATIVE_INFINITY, decoded, "Code 0xFF should be -Inf");
                } else {
                    assertTrue(Float.isFinite(decoded),
                        "Code " + code + " should be finite");
                }
            }
        }

        @Test
        @DisplayName("Key values decode correctly")
        void keyValuesDecodeCorrectly() {
            // Zero
            assertEquals(0.0f, MiniFloat.decodeToFloat(0x00, PARAMS), 1e-10f);

            // One: exp=7, mant=0 -> 2^0 × 1.0 = 1.0
            // Code = 0_0111_000 = 0x38
            assertEquals(1.0f, MiniFloat.decodeToFloat(0x38, PARAMS), 1e-6f);

            // Max finite: exp=15, mant=6 = 2^8 × 1.75 = 448
            // Code = 0_1111_110 = 0x7E
            // (Code 0x7F = exp=15, mant=7 is +Inf)
            assertEquals(448.0f, MiniFloat.decodeToFloat(0x7E, PARAMS), 1e-4f);

            // -448
            assertEquals(-448.0f, MiniFloat.decodeToFloat(0xFE, PARAMS), 1e-4f);

            // Verify code 0x77 (old expected max) is actually 240
            assertEquals(240.0f, MiniFloat.decodeToFloat(0x77, PARAMS), 1e-4f);
        }

        @Test
        @DisplayName("All finite codes round-trip correctly")
        void roundTripAllFiniteCodes() {
            for (int code = 0; code < 256; code++) {
                if (code == 0x80 || code == 0x7F || code == 0xFF) continue;

                float decoded = MiniFloat.decodeToFloat(code, PARAMS);

                if (decoded == 0.0f) {
                    assertEquals(0, MiniFloat.encode(decoded, PARAMS));
                } else {
                    assertEquals(code, MiniFloat.encode(decoded, PARAMS),
                        "Code " + code + " (value " + decoded + ") should round-trip");
                }
            }
        }

        @Test
        @DisplayName("E4M3 has higher precision than E5M2 near 1.0")
        void higherPrecisionThanE5M2() {
            // E4M3 has 3 mantissa bits (8 levels between powers of 2)
            // E5M2 has 2 mantissa bits (4 levels between powers of 2)

            // 1.125 = 1 + 1/8 is exactly representable in E4M3 but not E5M2
            float value = 1.125f;
            int e4m3Bits = MiniFloat.encode(value, PARAMS);
            float e4m3Decoded = MiniFloat.decodeToFloat(e4m3Bits, PARAMS);

            int e5m2Bits = MiniFloat.encode(value, FormatParameters.FP8_E5M2);
            float e5m2Decoded = MiniFloat.decodeToFloat(e5m2Bits, FormatParameters.FP8_E5M2);

            // E4M3 should be exact
            assertEquals(1.125f, e4m3Decoded, 1e-6f);
            // E5M2 will round to nearest representable: 1.0, 1.25, or 1.5
            assertTrue(e5m2Decoded == 1.0f || e5m2Decoded == 1.25f || e5m2Decoded == 1.5f,
                "E5M2 decoded to " + e5m2Decoded);
            // But E5M2 should NOT be exact for 1.125
            assertTrue(Math.abs(e5m2Decoded - 1.125f) > 0.001f,
                "E5M2 should not exactly represent 1.125");
        }
    }

    // ==================== FP8 E4M3FN (256 values, finite-only) ====================

    @Nested
    @DisplayName("FP8 E4M3FN - All 256 Code Points (Finite-Only)")
    class Fp8E4m3fnExhaustiveTest {

        private static final FormatParameters PARAMS = FormatParameters.FP8_E4M3FN;

        /**
         * FP8 E4M3FN: Same as E4M3 but finite-only (no NaN/Inf)
         *
         * Max value: exp=15, mant=7 = 2^8 × 1.875 = 480
         */
        @Test
        @DisplayName("All 256 codes are finite")
        void allCodesAreFinite() {
            for (int code = 0; code < 256; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);
                assertFalse(Float.isNaN(decoded), "Code " + code + " should not be NaN");
                assertFalse(Float.isInfinite(decoded), "Code " + code + " should not be Inf");
            }
        }

        @Test
        @DisplayName("Key values decode correctly")
        void keyValuesDecodeCorrectly() {
            assertEquals(0.0f, MiniFloat.decodeToFloat(0x00, PARAMS), 1e-10f);

            // One: exp=7, mant=0
            assertEquals(1.0f, MiniFloat.decodeToFloat(0x38, PARAMS), 1e-6f);

            // Max positive: exp=15, mant=7 = 2^8 × 1.875 = 480
            assertEquals(480.0f, MiniFloat.decodeToFloat(0x7F, PARAMS), 1e-4f);

            // Max negative
            assertEquals(-480.0f, MiniFloat.decodeToFloat(0xFF, PARAMS), 1e-4f);
        }

        @Test
        @DisplayName("All codes round-trip correctly")
        void roundTripAllCodes() {
            for (int code = 0; code < 256; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);

                if (decoded == 0.0f) {
                    assertEquals(0, MiniFloat.encode(decoded, PARAMS));
                } else {
                    assertEquals(code, MiniFloat.encode(decoded, PARAMS),
                        "Code " + code + " (value " + decoded + ") should round-trip");
                }
            }
        }

        @Test
        @DisplayName("Finite-only has larger max than extended E4M3")
        void largerMaxThanExtended() {
            // E4M3FN max = 480 (uses exp=15)
            // E4M3 extended max = 240 (exp=14, because exp=15 is for inf)
            assertTrue(PARAMS.maxValue() > FormatParameters.FP8_E4M3.maxValue());
        }
    }

    // ==================== FP8 E8M0 (256 values, pure exponent) ====================

    @Nested
    @DisplayName("FP8 E8M0 - All 256 Code Points (Pure Exponent)")
    class Fp8E8m0ExhaustiveTest {

        private static final FormatParameters PARAMS = FormatParameters.FP8_E8M0;

        /**
         * FP8 E8M0: Pure exponent format (unsigned, no mantissa)
         *
         * This is used for scale factors in NVFP4 and MX formats.
         *
         * value = 2^(code - 127) for code 1-254
         * code 0 = 2^(-127)
         * code 255 = NaN (in extended format)
         */
        @Test
        @DisplayName("Decode all 256 code points")
        void decodeAllCodes() {
            for (int code = 0; code < 256; code++) {
                float decoded = MiniFloat.decodeToFloat(code, PARAMS);

                if (code == 255 && !PARAMS.finiteOnly()) {
                    assertTrue(Float.isNaN(decoded), "Code 255 should be NaN");
                } else if (code == 0) {
                    // Code 0 is either 0 or 2^(-127) depending on interpretation
                    assertTrue(decoded >= 0, "Code 0 should be non-negative");
                } else {
                    // Normal values: 2^(code - 127)
                    float expected = (float) Math.pow(2, code - 127);
                    assertEquals(expected, decoded, expected * 1e-6f,
                        "Code " + code + " should decode to 2^" + (code - 127));
                }
            }
        }

        @Test
        @DisplayName("Key scale values")
        void keyScaleValues() {
            // 2^0 = 1.0 at code 127
            assertEquals(1.0f, MiniFloat.decodeToFloat(127, PARAMS), 1e-6f);

            // 2^1 = 2.0 at code 128
            assertEquals(2.0f, MiniFloat.decodeToFloat(128, PARAMS), 1e-6f);

            // 2^(-1) = 0.5 at code 126
            assertEquals(0.5f, MiniFloat.decodeToFloat(126, PARAMS), 1e-6f);

            // 2^7 = 128 at code 134
            assertEquals(128.0f, MiniFloat.decodeToFloat(134, PARAMS), 1e-4f);
        }

        @Test
        @DisplayName("All powers of 2 encode correctly")
        void powersOfTwoEncode() {
            for (int exp = -126; exp <= 127; exp++) {
                float value = (float) Math.pow(2, exp);
                int bits = MiniFloat.encode(value, PARAMS);
                int expectedCode = exp + 127;
                assertEquals(expectedCode, bits,
                    "2^" + exp + " should encode to code " + expectedCode);
            }
        }
    }

    // ==================== Cross-format consistency tests ====================

    @Nested
    @DisplayName("Cross-Format Consistency")
    class CrossFormatConsistencyTest {

        @Test
        @DisplayName("All formats agree on encoding zero")
        void allFormatsEncodeZero() {
            assertEquals(0, MiniFloat.encode(0.0f, FormatParameters.FP4_E2M1));
            assertEquals(0, MiniFloat.encode(0.0f, FormatParameters.FP4_E1M2));
            assertEquals(0, MiniFloat.encode(0.0f, FormatParameters.FP6_E3M2));
            assertEquals(0, MiniFloat.encode(0.0f, FormatParameters.FP6_E2M3));
            assertEquals(0, MiniFloat.encode(0.0f, FormatParameters.FP8_E5M2));
            assertEquals(0, MiniFloat.encode(0.0f, FormatParameters.FP8_E4M3));
            assertEquals(0, MiniFloat.encode(0.0f, FormatParameters.FP8_E4M3FN));
        }

        @Test
        @DisplayName("All formats decode zero from code 0")
        void allFormatsDecodeZero() {
            assertEquals(0.0f, MiniFloat.decodeToFloat(0, FormatParameters.FP4_E2M1));
            assertEquals(0.0f, MiniFloat.decodeToFloat(0, FormatParameters.FP4_E1M2));
            assertEquals(0.0f, MiniFloat.decodeToFloat(0, FormatParameters.FP6_E3M2));
            assertEquals(0.0f, MiniFloat.decodeToFloat(0, FormatParameters.FP6_E2M3));
            assertEquals(0.0f, MiniFloat.decodeToFloat(0, FormatParameters.FP8_E5M2));
            assertEquals(0.0f, MiniFloat.decodeToFloat(0, FormatParameters.FP8_E4M3));
            assertEquals(0.0f, MiniFloat.decodeToFloat(0, FormatParameters.FP8_E4M3FN));
            assertEquals(0.0f, MiniFloat.decodeToFloat(0, FormatParameters.FP8_E8M0));
        }

        @Test
        @DisplayName("Finite-only formats reject NaN input gracefully")
        void finiteOnlyFormatsRejectNaN() {
            // NaN should encode to 0 for finite-only formats
            assertEquals(0, MiniFloat.encode(Float.NaN, FormatParameters.FP4_E2M1));
            assertEquals(0, MiniFloat.encode(Float.NaN, FormatParameters.FP4_E1M2));
            assertEquals(0, MiniFloat.encode(Float.NaN, FormatParameters.FP6_E3M2));
            assertEquals(0, MiniFloat.encode(Float.NaN, FormatParameters.FP6_E2M3));
            assertEquals(0, MiniFloat.encode(Float.NaN, FormatParameters.FP8_E4M3FN));
        }

        @Test
        @DisplayName("Extended formats encode NaN to sign bit pattern")
        void extendedFormatsEncodeNaN() {
            // P3109 NaN for signed: 2^(K-1) = sign bit only
            assertEquals(0x80, MiniFloat.encode(Float.NaN, FormatParameters.FP8_E5M2));
            assertEquals(0x80, MiniFloat.encode(Float.NaN, FormatParameters.FP8_E4M3));
        }

        @Test
        @DisplayName("Value 1.0 encodes correctly in all formats")
        void oneEncodesCorrectly() {
            // 1.0 should round-trip in all formats
            float[] formats = {
                MiniFloat.decodeToFloat(MiniFloat.encode(1.0f, FormatParameters.FP4_E2M1), FormatParameters.FP4_E2M1),
                MiniFloat.decodeToFloat(MiniFloat.encode(1.0f, FormatParameters.FP4_E1M2), FormatParameters.FP4_E1M2),
                MiniFloat.decodeToFloat(MiniFloat.encode(1.0f, FormatParameters.FP6_E3M2), FormatParameters.FP6_E3M2),
                MiniFloat.decodeToFloat(MiniFloat.encode(1.0f, FormatParameters.FP6_E2M3), FormatParameters.FP6_E2M3),
                MiniFloat.decodeToFloat(MiniFloat.encode(1.0f, FormatParameters.FP8_E5M2), FormatParameters.FP8_E5M2),
                MiniFloat.decodeToFloat(MiniFloat.encode(1.0f, FormatParameters.FP8_E4M3), FormatParameters.FP8_E4M3),
                MiniFloat.decodeToFloat(MiniFloat.encode(1.0f, FormatParameters.FP8_E4M3FN), FormatParameters.FP8_E4M3FN),
                MiniFloat.decodeToFloat(MiniFloat.encode(1.0f, FormatParameters.FP8_E8M0), FormatParameters.FP8_E8M0),
            };

            for (float decoded : formats) {
                assertEquals(1.0f, decoded, 1e-6f);
            }
        }
    }
}
