package io.surfworks.warpforge.data;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DTypeTest {

    // =========================================================================
    // Basic Type Properties
    // =========================================================================

    @Test
    void testByteSizes() {
        assertEquals(4, DType.F32.byteSize());
        assertEquals(8, DType.F64.byteSize());
        assertEquals(2, DType.F16.byteSize());
        assertEquals(2, DType.BF16.byteSize());
        assertEquals(1, DType.I8.byteSize());
        assertEquals(4, DType.I32.byteSize());
        assertEquals(8, DType.I64.byteSize());
        // FP8 types
        assertEquals(1, DType.F8_E5M2.byteSize());
        assertEquals(1, DType.F8_E4M3.byteSize());
        assertEquals(1, DType.F8_E4M3FN.byteSize());
        // FP4/FP6 report 1 for element access
        assertEquals(1, DType.F4_E2M1.byteSize());
        assertEquals(1, DType.F6_E3M2.byteSize());
    }

    @Test
    void testBitWidths() {
        assertEquals(32, DType.F32.bitWidth());
        assertEquals(16, DType.F16.bitWidth());
        assertEquals(8, DType.F8_E5M2.bitWidth());
        assertEquals(8, DType.F8_E4M3.bitWidth());
        assertEquals(6, DType.F6_E3M2.bitWidth());
        assertEquals(6, DType.F6_E2M3.bitWidth());
        assertEquals(4, DType.F4_E2M1.bitWidth());
        assertEquals(4, DType.F4_E1M2.bitWidth());
    }

    @Test
    void testIsFloating() {
        assertTrue(DType.F32.isFloating());
        assertTrue(DType.F64.isFloating());
        assertTrue(DType.F16.isFloating());
        assertTrue(DType.BF16.isFloating());
        assertTrue(DType.F8_E5M2.isFloating());
        assertTrue(DType.F8_E4M3.isFloating());
        assertTrue(DType.F4_E2M1.isFloating());
        assertFalse(DType.I32.isFloating());
        assertFalse(DType.I64.isFloating());
        assertFalse(DType.BOOL.isFloating());
    }

    @Test
    void testIsP3109() {
        assertFalse(DType.F32.isP3109());
        assertFalse(DType.F16.isP3109());
        assertFalse(DType.BF16.isP3109());
        assertTrue(DType.F8_E5M2.isP3109());
        assertTrue(DType.F8_E4M3.isP3109());
        assertTrue(DType.F8_E4M3FN.isP3109());
        assertTrue(DType.F8_E8M0.isP3109());
        assertTrue(DType.F6_E3M2.isP3109());
        assertTrue(DType.F6_E2M3.isP3109());
        assertTrue(DType.F4_E2M1.isP3109());
        assertTrue(DType.F4_E1M2.isP3109());
    }

    @Test
    void testIsSubByte() {
        assertFalse(DType.F32.isSubByte());
        assertFalse(DType.F8_E5M2.isSubByte());
        assertTrue(DType.F6_E3M2.isSubByte());
        assertTrue(DType.F6_E2M3.isSubByte());
        assertTrue(DType.F4_E2M1.isSubByte());
        assertTrue(DType.F4_E1M2.isSubByte());
    }

    @Test
    void testIsBlockQuantized() {
        assertFalse(DType.F32.isBlockQuantized());
        assertFalse(DType.F8_E5M2.isBlockQuantized());
        assertFalse(DType.F4_E2M1.isBlockQuantized());
        assertTrue(DType.Q4_0.isBlockQuantized());
        assertTrue(DType.Q4_K_M.isBlockQuantized());
        assertTrue(DType.Q8_0.isBlockQuantized());
    }

    @Test
    void testPackedByteSize() {
        // Standard types
        assertEquals(40, DType.F32.packedByteSize(10));
        assertEquals(10, DType.F8_E5M2.packedByteSize(10));

        // FP4: 2 values per byte
        assertEquals(5, DType.F4_E2M1.packedByteSize(10));
        assertEquals(1, DType.F4_E2M1.packedByteSize(2));
        assertEquals(1, DType.F4_E2M1.packedByteSize(1)); // ceiling

        // FP6: 4 values in 3 bytes
        assertEquals(8, DType.F6_E3M2.packedByteSize(10)); // 60 bits -> 8 bytes
        assertEquals(3, DType.F6_E3M2.packedByteSize(4));  // 24 bits -> 3 bytes
    }

    // =========================================================================
    // SafeTensors Parsing
    // =========================================================================

    @Test
    void testFromSafeTensorsStandard() {
        assertEquals(DType.F32, DType.fromSafeTensors("F32"));
        assertEquals(DType.F32, DType.fromSafeTensors("FLOAT32"));
        assertEquals(DType.F16, DType.fromSafeTensors("F16"));
        assertEquals(DType.F16, DType.fromSafeTensors("FLOAT16"));
        assertEquals(DType.BF16, DType.fromSafeTensors("BF16"));
        assertEquals(DType.BF16, DType.fromSafeTensors("BFLOAT16"));
        assertEquals(DType.I32, DType.fromSafeTensors("I32"));
        assertEquals(DType.I32, DType.fromSafeTensors("INT32"));
        assertEquals(DType.I64, DType.fromSafeTensors("I64"));
        assertEquals(DType.I64, DType.fromSafeTensors("INT64"));
    }

    @Test
    void testFromSafeTensorsFP8() {
        assertEquals(DType.F8_E5M2, DType.fromSafeTensors("F8_E5M2"));
        assertEquals(DType.F8_E5M2, DType.fromSafeTensors("FLOAT8_E5M2"));
        assertEquals(DType.F8_E5M2, DType.fromSafeTensors("E5M2"));
        assertEquals(DType.F8_E4M3, DType.fromSafeTensors("F8_E4M3"));
        assertEquals(DType.F8_E4M3, DType.fromSafeTensors("FLOAT8_E4M3"));
        assertEquals(DType.F8_E4M3FN, DType.fromSafeTensors("F8_E4M3FN"));
        assertEquals(DType.F8_E4M3FN, DType.fromSafeTensors("FLOAT8_E4M3FN"));
    }

    @Test
    void testFromSafeTensorsFP4FP6() {
        assertEquals(DType.F4_E2M1, DType.fromSafeTensors("F4_E2M1"));
        assertEquals(DType.F4_E2M1, DType.fromSafeTensors("FLOAT4_E2M1"));
        assertEquals(DType.F4_E2M1, DType.fromSafeTensors("E2M1"));
        assertEquals(DType.F4_E1M2, DType.fromSafeTensors("F4_E1M2"));
        assertEquals(DType.F6_E3M2, DType.fromSafeTensors("F6_E3M2"));
        assertEquals(DType.F6_E2M3, DType.fromSafeTensors("F6_E2M3"));
    }

    @Test
    void testFromSafeTensorsUnknown() {
        assertThrows(IllegalArgumentException.class, () -> DType.fromSafeTensors("UNKNOWN"));
    }

    // =========================================================================
    // BF16 Conversions
    // =========================================================================

    @Nested
    class BF16ConversionTests {

        @Test
        void testZero() {
            assertEquals(0.0f, DType.bf16ToFloat((short) 0), 0.0f);
            assertEquals(-0.0f, DType.bf16ToFloat((short) 0x8000), 0.0f);
        }

        @Test
        void testOne() {
            short oneBits = DType.floatToBf16(1.0f);
            assertEquals(1.0f, DType.bf16ToFloat(oneBits), 0.001f);
        }

        @ParameterizedTest
        @ValueSource(floats = {0.5f, 1.0f, 1.5f, 2.0f, 100.0f, -1.0f, -0.5f})
        void testRoundTrip(float value) {
            short bits = DType.floatToBf16(value);
            float converted = DType.bf16ToFloat(bits);
            // BF16 has ~3 decimal digits of precision
            assertEquals(value, converted, Math.abs(value) * 0.01f + 0.001f);
        }
    }

    // =========================================================================
    // F16 Conversions
    // =========================================================================

    @Nested
    class F16ConversionTests {

        @Test
        void testZero() {
            assertEquals(0.0f, DType.f16ToFloat((short) 0), 0.0f);
        }

        @Test
        void testNegativeZero() {
            assertEquals(-0.0f, DType.f16ToFloat((short) 0x8000), 0.0f);
        }

        @Test
        void testOne() {
            // F16 encoding for 1.0: sign=0, exp=15 (biased), mantissa=0
            // Bits: 0 01111 0000000000 = 0x3C00
            assertEquals(1.0f, DType.f16ToFloat((short) 0x3C00), 0.001f);
        }

        @Test
        void testTwo() {
            // F16 encoding for 2.0: sign=0, exp=16 (biased), mantissa=0
            // Bits: 0 10000 0000000000 = 0x4000
            assertEquals(2.0f, DType.f16ToFloat((short) 0x4000), 0.001f);
        }

        @Test
        void testNegativeOne() {
            // F16 encoding for -1.0: sign=1, exp=15, mantissa=0
            // Bits: 1 01111 0000000000 = 0xBC00
            assertEquals(-1.0f, DType.f16ToFloat((short) 0xBC00), 0.001f);
        }

        @Test
        void testInfinity() {
            // +Inf: exp=31, mantissa=0 -> 0x7C00
            assertEquals(Float.POSITIVE_INFINITY, DType.f16ToFloat((short) 0x7C00), 0.0f);
            // -Inf: 0xFC00
            assertEquals(Float.NEGATIVE_INFINITY, DType.f16ToFloat((short) 0xFC00), 0.0f);
        }

        @Test
        void testNaN() {
            // NaN: exp=31, mantissa!=0 -> 0x7C01
            assertTrue(Float.isNaN(DType.f16ToFloat((short) 0x7C01)));
        }

        @Test
        void testSubnormal() {
            // Smallest positive subnormal: 0x0001 = 2^(-14) * (1/1024) ≈ 5.96e-8
            float subnormal = DType.f16ToFloat((short) 0x0001);
            assertTrue(subnormal > 0);
            assertTrue(subnormal < 1e-6f);
        }
    }

    // =========================================================================
    // FP8 E5M2 Conversions
    // =========================================================================

    @Nested
    class F8E5M2ConversionTests {

        @Test
        void testZero() {
            assertEquals(0.0f, DType.f8e5m2ToFloat((byte) 0x00), 0.0f);
        }

        @Test
        void testNegativeZero() {
            assertEquals(-0.0f, DType.f8e5m2ToFloat((byte) 0x80), 0.0f);
        }

        @Test
        void testOne() {
            // E5M2 encoding for 1.0: sign=0, exp=15 (bias=15), mantissa=0
            // Bits: 0 01111 00 = 0x3C
            assertEquals(1.0f, DType.f8e5m2ToFloat((byte) 0x3C), 0.001f);
        }

        @Test
        void testTwo() {
            // E5M2 encoding for 2.0: sign=0, exp=16, mantissa=0
            // Bits: 0 10000 00 = 0x40
            assertEquals(2.0f, DType.f8e5m2ToFloat((byte) 0x40), 0.001f);
        }

        @Test
        void testNegativeOne() {
            // sign=1, exp=15, mantissa=0 -> 0xBC
            assertEquals(-1.0f, DType.f8e5m2ToFloat((byte) 0xBC), 0.001f);
        }

        @Test
        void testInfinity() {
            // +Inf: exp=31, mantissa=0 -> 0x7C
            assertEquals(Float.POSITIVE_INFINITY, DType.f8e5m2ToFloat((byte) 0x7C), 0.0f);
            // -Inf: 0xFC
            assertEquals(Float.NEGATIVE_INFINITY, DType.f8e5m2ToFloat((byte) 0xFC), 0.0f);
        }

        @Test
        void testNaN() {
            // NaN: exp=31, mantissa!=0 -> 0x7D, 0x7E, 0x7F
            assertTrue(Float.isNaN(DType.f8e5m2ToFloat((byte) 0x7F)));
            assertTrue(Float.isNaN(DType.f8e5m2ToFloat((byte) 0x7D)));
        }

        @Test
        void testRoundTrip() {
            // Test round-trip for values representable in E5M2
            float[] testValues = {0.5f, 1.0f, 2.0f, 4.0f, -1.0f, -2.0f};
            for (float v : testValues) {
                byte bits = DType.floatToF8e5m2(v);
                float converted = DType.f8e5m2ToFloat(bits);
                assertEquals(v, converted, Math.abs(v) * 0.3f + 0.01f,
                    "Round-trip failed for " + v);
            }
        }

        @Test
        void testConversionSpecialValues() {
            // NaN -> NaN
            byte nanBits = DType.floatToF8e5m2(Float.NaN);
            assertTrue(Float.isNaN(DType.f8e5m2ToFloat(nanBits)));

            // Inf -> Inf
            byte infBits = DType.floatToF8e5m2(Float.POSITIVE_INFINITY);
            assertEquals(Float.POSITIVE_INFINITY, DType.f8e5m2ToFloat(infBits), 0.0f);

            byte negInfBits = DType.floatToF8e5m2(Float.NEGATIVE_INFINITY);
            assertEquals(Float.NEGATIVE_INFINITY, DType.f8e5m2ToFloat(negInfBits), 0.0f);
        }
    }

    // =========================================================================
    // FP8 E4M3 Conversions
    // =========================================================================

    @Nested
    class F8E4M3ConversionTests {

        @Test
        void testZero() {
            assertEquals(0.0f, DType.f8e4m3ToFloat((byte) 0x00), 0.0f);
        }

        @Test
        void testNegativeZero() {
            assertEquals(-0.0f, DType.f8e4m3ToFloat((byte) 0x80), 0.0f);
        }

        @Test
        void testOne() {
            // E4M3 encoding for 1.0: sign=0, exp=7 (bias=7), mantissa=0
            // Bits: 0 0111 000 = 0x38
            assertEquals(1.0f, DType.f8e4m3ToFloat((byte) 0x38), 0.001f);
        }

        @Test
        void testTwo() {
            // E4M3 encoding for 2.0: sign=0, exp=8, mantissa=0
            // Bits: 0 1000 000 = 0x40
            assertEquals(2.0f, DType.f8e4m3ToFloat((byte) 0x40), 0.001f);
        }

        @Test
        void testNaN() {
            // In E4M3, NaN is at exp=15, mantissa=7 -> 0x7F and 0xFF
            assertTrue(Float.isNaN(DType.f8e4m3ToFloat((byte) 0x7F)));
            assertTrue(Float.isNaN(DType.f8e4m3ToFloat((byte) 0xFF)));
        }

        @Test
        void testMaxFinite() {
            // E4M3 max finite: exp=15, mantissa=6 -> 0x7E
            // Value = 2^8 * (1 + 6/8) = 256 * 1.75 = 448
            float maxFinite = DType.f8e4m3ToFloat((byte) 0x7E);
            assertEquals(448.0f, maxFinite, 1.0f);
        }

        @Test
        void testRoundTrip() {
            float[] testValues = {0.5f, 1.0f, 2.0f, 4.0f, -1.0f, -2.0f, 0.125f};
            for (float v : testValues) {
                byte bits = DType.floatToF8e4m3(v);
                float converted = DType.f8e4m3ToFloat(bits);
                assertEquals(v, converted, Math.abs(v) * 0.2f + 0.01f,
                    "Round-trip failed for " + v);
            }
        }
    }

    // =========================================================================
    // FP8 E4M3FN (Finite-Only) Conversions
    // =========================================================================

    @Nested
    class F8E4M3FNConversionTests {

        @Test
        void testZero() {
            assertEquals(0.0f, DType.f8e4m3fnToFloat((byte) 0x00), 0.0f);
        }

        @Test
        void testOne() {
            assertEquals(1.0f, DType.f8e4m3fnToFloat((byte) 0x38), 0.001f);
        }

        @Test
        void testAllCodesAreFinite() {
            // In E4M3FN, ALL 256 codes are finite (no NaN, no Inf)
            for (int i = 0; i < 256; i++) {
                float val = DType.f8e4m3fnToFloat((byte) i);
                assertTrue(Float.isFinite(val), "Code " + i + " should be finite");
            }
        }

        @Test
        void testMaxValue() {
            // 0x7F = exp=15, mantissa=7 -> 2^8 * (1 + 7/8) = 256 * 1.875 = 480
            float maxPos = DType.f8e4m3fnToFloat((byte) 0x7F);
            assertEquals(480.0f, maxPos, 1.0f);

            // 0xFF = negative max
            float maxNeg = DType.f8e4m3fnToFloat((byte) 0xFF);
            assertEquals(-480.0f, maxNeg, 1.0f);
        }
    }

    // =========================================================================
    // FP4 E2M1 Conversions
    // =========================================================================

    @Nested
    class F4E2M1ConversionTests {

        @Test
        void testZero() {
            assertEquals(0.0f, DType.f4e2m1ToFloat(0x0), 0.0f);
        }

        @Test
        void testNegativeZero() {
            assertEquals(-0.0f, DType.f4e2m1ToFloat(0x8), 0.0f);
        }

        @Test
        void testSubnormal() {
            // 0x1 = subnormal, value = 0.5
            assertEquals(0.5f, DType.f4e2m1ToFloat(0x1), 0.001f);
            // 0x9 = negative subnormal
            assertEquals(-0.5f, DType.f4e2m1ToFloat(0x9), 0.001f);
        }

        @Test
        void testOne() {
            // E2M1: 1.0 = sign=0, exp=1, mantissa=0
            // Bits: 0 01 0 = 0x2
            assertEquals(1.0f, DType.f4e2m1ToFloat(0x2), 0.001f);
        }

        @Test
        void testAllValues() {
            // E2M1 has only 16 possible values
            // Let's enumerate them: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
            float[] expectedPositive = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            for (int i = 0; i < 8; i++) {
                assertEquals(expectedPositive[i], DType.f4e2m1ToFloat(i), 0.001f,
                    "Positive value at " + i);
                if (i > 0) {
                    assertEquals(-expectedPositive[i], DType.f4e2m1ToFloat(i | 0x8), 0.001f,
                        "Negative value at " + (i | 0x8));
                }
            }
        }

        @Test
        void testRoundTripQuantization() {
            // Test that conversion quantizes correctly
            assertEquals(0x0, DType.floatToF4e2m1(0.0f));
            assertEquals(0x1, DType.floatToF4e2m1(0.5f));
            assertEquals(0x2, DType.floatToF4e2m1(1.0f));
            assertEquals(0x3, DType.floatToF4e2m1(1.5f));
            assertEquals(0x4, DType.floatToF4e2m1(2.0f));
            assertEquals(0x5, DType.floatToF4e2m1(3.0f));
            assertEquals(0x7, DType.floatToF4e2m1(6.0f));
            assertEquals(0x7, DType.floatToF4e2m1(100.0f)); // Clamp to max
        }

        @Test
        void testNegativeConversion() {
            assertEquals(0x8, DType.floatToF4e2m1(-0.1f)); // Rounds to -0
            assertEquals(0x9, DType.floatToF4e2m1(-0.5f));
            assertEquals(0xA, DType.floatToF4e2m1(-1.0f));
            assertEquals(0xF, DType.floatToF4e2m1(-100.0f)); // Clamp to -max
        }
    }

    // =========================================================================
    // FP4 E1M2 Conversions
    // =========================================================================

    @Nested
    class F4E1M2ConversionTests {

        @Test
        void testZero() {
            assertEquals(0.0f, DType.f4e1m2ToFloat(0x0), 0.0f);
        }

        @Test
        void testSubnormals() {
            // E1M2 subnormals: exp=0, mantissa=1,2,3 -> 0.25, 0.5, 0.75
            assertEquals(0.25f, DType.f4e1m2ToFloat(0x1), 0.001f);
            assertEquals(0.5f, DType.f4e1m2ToFloat(0x2), 0.001f);
            assertEquals(0.75f, DType.f4e1m2ToFloat(0x3), 0.001f);
        }

        @Test
        void testNormals() {
            // E1M2 normals: exp=1, 2^1 * (1 + m/4)
            // m=0: 2.0, m=1: 2.5, m=2: 3.0, m=3: 3.5
            assertEquals(2.0f, DType.f4e1m2ToFloat(0x4), 0.001f);
            assertEquals(2.5f, DType.f4e1m2ToFloat(0x5), 0.001f);
            assertEquals(3.0f, DType.f4e1m2ToFloat(0x6), 0.001f);
            assertEquals(3.5f, DType.f4e1m2ToFloat(0x7), 0.001f);
        }

        @Test
        void testNegatives() {
            assertEquals(-0.25f, DType.f4e1m2ToFloat(0x9), 0.001f);
            assertEquals(-2.0f, DType.f4e1m2ToFloat(0xC), 0.001f);
            assertEquals(-3.5f, DType.f4e1m2ToFloat(0xF), 0.001f);
        }

        @Test
        void testAllValues() {
            // E1M2: 0, 0.25, 0.5, 0.75, 2.0, 2.5, 3.0, 3.5 (and negatives)
            float[] expectedPositive = {0.0f, 0.25f, 0.5f, 0.75f, 2.0f, 2.5f, 3.0f, 3.5f};
            for (int i = 0; i < 8; i++) {
                assertEquals(expectedPositive[i], DType.f4e1m2ToFloat(i), 0.001f,
                    "Positive value at " + i);
            }
        }
    }

    // =========================================================================
    // Edge Cases and Comprehensive Coverage
    // =========================================================================

    @Nested
    class EdgeCaseTests {

        @Test
        void testAllFP8E5M2Codes() {
            // Verify all 256 codes produce valid floats (finite, inf, or nan)
            int finiteCount = 0;
            int infCount = 0;
            int nanCount = 0;

            for (int i = 0; i < 256; i++) {
                float val = DType.f8e5m2ToFloat((byte) i);
                if (Float.isNaN(val)) {
                    nanCount++;
                } else if (Float.isInfinite(val)) {
                    infCount++;
                } else {
                    finiteCount++;
                }
            }

            // E5M2 has: exp=31, mantissa=0 for ±Inf (0x7C, 0xFC)
            // exp=31, mantissa!=0 for NaN (0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF = 6 NaN codes)
            // Rest are finite (including 2 zeros)
            assertTrue(finiteCount > 240, "Should have many finite values: " + finiteCount);
            assertEquals(2, infCount, "Should have exactly 2 infinity values (+inf and -inf)");
            assertEquals(6, nanCount, "Should have 6 NaN values (3 positive, 3 negative mantissa patterns)");
        }

        @Test
        void testAllFP8E4M3Codes() {
            int finiteCount = 0;
            int nanCount = 0;

            for (int i = 0; i < 256; i++) {
                float val = DType.f8e4m3ToFloat((byte) i);
                if (Float.isNaN(val)) {
                    nanCount++;
                } else {
                    finiteCount++;
                }
            }

            // E4M3: only 0x7F and 0xFF are NaN
            assertEquals(2, nanCount, "E4M3 should have exactly 2 NaN codes");
            assertEquals(254, finiteCount, "E4M3 should have 254 finite codes");
        }

        @Test
        void testAllFP4E2M1Codes() {
            // All 16 FP4 codes should produce finite values
            for (int i = 0; i < 16; i++) {
                float val = DType.f4e2m1ToFloat(i);
                assertTrue(Float.isFinite(val), "FP4 E2M1 code " + i + " should be finite");
            }
        }

        @Test
        void testAllFP4E1M2Codes() {
            // All 16 FP4 codes should produce finite values
            for (int i = 0; i < 16; i++) {
                float val = DType.f4e1m2ToFloat(i);
                assertTrue(Float.isFinite(val), "FP4 E1M2 code " + i + " should be finite");
            }
        }

        @Test
        void testSymmetry() {
            // Positive and negative values should be symmetric
            for (int i = 1; i < 8; i++) {
                float pos4e2m1 = DType.f4e2m1ToFloat(i);
                float neg4e2m1 = DType.f4e2m1ToFloat(i | 0x8);
                assertEquals(pos4e2m1, -neg4e2m1, 0.0001f, "E2M1 symmetry at " + i);

                float pos4e1m2 = DType.f4e1m2ToFloat(i);
                float neg4e1m2 = DType.f4e1m2ToFloat(i | 0x8);
                assertEquals(pos4e1m2, -neg4e1m2, 0.0001f, "E1M2 symmetry at " + i);
            }
        }
    }
}
