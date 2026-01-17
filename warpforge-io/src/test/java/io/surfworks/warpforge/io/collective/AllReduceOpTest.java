package io.surfworks.warpforge.io.collective;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for AllReduceOp enum.
 */
@Tag("unit")
@DisplayName("AllReduceOp Unit Tests")
class AllReduceOpTest {

    @Test
    @DisplayName("Enum should have all expected values")
    void testEnumValues() {
        AllReduceOp[] ops = AllReduceOp.values();
        assertTrue(ops.length >= 12, "Should have at least 12 reduction operations");

        // Verify key operations exist
        assertNotNull(AllReduceOp.valueOf("SUM"));
        assertNotNull(AllReduceOp.valueOf("PROD"));
        assertNotNull(AllReduceOp.valueOf("MIN"));
        assertNotNull(AllReduceOp.valueOf("MAX"));
        assertNotNull(AllReduceOp.valueOf("AVG"));
        assertNotNull(AllReduceOp.valueOf("BAND"));
        assertNotNull(AllReduceOp.valueOf("BOR"));
        assertNotNull(AllReduceOp.valueOf("BXOR"));
        assertNotNull(AllReduceOp.valueOf("LAND"));
        assertNotNull(AllReduceOp.valueOf("LOR"));
        assertNotNull(AllReduceOp.valueOf("MINLOC"));
        assertNotNull(AllReduceOp.valueOf("MAXLOC"));
    }

    @Test
    @DisplayName("SUM should have UCC code 0")
    void testSumUccCode() {
        assertEquals(0, AllReduceOp.SUM.uccCode());
    }

    @Test
    @DisplayName("PROD should have UCC code 1")
    void testProdUccCode() {
        assertEquals(1, AllReduceOp.PROD.uccCode());
    }

    @Test
    @DisplayName("MIN should have UCC code 2")
    void testMinUccCode() {
        assertEquals(2, AllReduceOp.MIN.uccCode());
    }

    @Test
    @DisplayName("MAX should have UCC code 3")
    void testMaxUccCode() {
        assertEquals(3, AllReduceOp.MAX.uccCode());
    }

    @Test
    @DisplayName("AVG should have UCC code 4")
    void testAvgUccCode() {
        assertEquals(4, AllReduceOp.AVG.uccCode());
    }

    @Test
    @DisplayName("BAND should have UCC code 5")
    void testBandUccCode() {
        assertEquals(5, AllReduceOp.BAND.uccCode());
    }

    @Test
    @DisplayName("BOR should have UCC code 6")
    void testBorUccCode() {
        assertEquals(6, AllReduceOp.BOR.uccCode());
    }

    @Test
    @DisplayName("BXOR should have UCC code 7")
    void testBxorUccCode() {
        assertEquals(7, AllReduceOp.BXOR.uccCode());
    }

    @Test
    @DisplayName("LAND should have UCC code 8")
    void testLandUccCode() {
        assertEquals(8, AllReduceOp.LAND.uccCode());
    }

    @Test
    @DisplayName("LOR should have UCC code 9")
    void testLorUccCode() {
        assertEquals(9, AllReduceOp.LOR.uccCode());
    }

    @Test
    @DisplayName("MINLOC should have UCC code 10")
    void testMinlocUccCode() {
        assertEquals(10, AllReduceOp.MINLOC.uccCode());
    }

    @Test
    @DisplayName("MAXLOC should have UCC code 11")
    void testMaxlocUccCode() {
        assertEquals(11, AllReduceOp.MAXLOC.uccCode());
    }

    @ParameterizedTest
    @EnumSource(AllReduceOp.class)
    @DisplayName("fromUccCode should roundtrip with uccCode")
    void testFromUccCodeRoundtrip(AllReduceOp op) {
        int code = op.uccCode();
        AllReduceOp result = AllReduceOp.fromUccCode(code);
        assertEquals(op, result);
    }

    @Test
    @DisplayName("fromUccCode should throw on invalid code")
    void testFromUccCodeInvalid() {
        assertThrows(IllegalArgumentException.class, () -> AllReduceOp.fromUccCode(-1));
        assertThrows(IllegalArgumentException.class, () -> AllReduceOp.fromUccCode(100));
        assertThrows(IllegalArgumentException.class, () -> AllReduceOp.fromUccCode(12));
    }

    @ParameterizedTest
    @EnumSource(AllReduceOp.class)
    @DisplayName("UCC codes should be unique and non-negative")
    void testUccCodesAreValid(AllReduceOp op) {
        int code = op.uccCode();
        assertTrue(code >= 0, "UCC code should be non-negative");
        assertTrue(code < 20, "UCC code should be reasonable");
    }

    @Test
    @DisplayName("All UCC codes should be unique")
    void testUccCodesAreUnique() {
        AllReduceOp[] ops = AllReduceOp.values();
        int[] codes = new int[ops.length];

        for (int i = 0; i < ops.length; i++) {
            codes[i] = ops[i].uccCode();
        }

        // Check for duplicates
        for (int i = 0; i < codes.length; i++) {
            for (int j = i + 1; j < codes.length; j++) {
                assertTrue(codes[i] != codes[j],
                    String.format("Duplicate UCC code %d for %s and %s",
                        codes[i], ops[i], ops[j]));
            }
        }
    }

    @Test
    @DisplayName("Common operations should be well-defined")
    void testCommonOperations() {
        // These are the most commonly used operations
        assertNotNull(AllReduceOp.SUM);
        assertNotNull(AllReduceOp.AVG);
        assertNotNull(AllReduceOp.MIN);
        assertNotNull(AllReduceOp.MAX);

        // Verify they have expected ordinals for quick reference
        assertEquals(0, AllReduceOp.SUM.uccCode());
        assertEquals(4, AllReduceOp.AVG.uccCode());
    }
}
