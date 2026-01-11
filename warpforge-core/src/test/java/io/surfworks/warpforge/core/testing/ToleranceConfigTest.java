package io.surfworks.warpforge.core.testing;

import io.surfworks.warpforge.core.tensor.ScalarType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ToleranceConfigTest {

    @Nested
    @DisplayName("isClose")
    class IsCloseTests {

        @Test
        void exactMatch() {
            ToleranceConfig tol = new ToleranceConfig(0, 0);
            assertTrue(tol.isClose(1.0, 1.0));
            assertTrue(tol.isClose(0.0, 0.0));
            assertTrue(tol.isClose(-5.5, -5.5));
        }

        @Test
        void withinAbsoluteTolerance() {
            ToleranceConfig tol = new ToleranceConfig(1e-5, 0);
            assertTrue(tol.isClose(1.0, 1.000005));
            assertTrue(tol.isClose(1.0, 0.999995));
            assertFalse(tol.isClose(1.0, 1.00002));
        }

        @Test
        void withinRelativeTolerance() {
            ToleranceConfig tol = new ToleranceConfig(0, 1e-3);
            assertTrue(tol.isClose(1000.0, 1000.5));
            assertTrue(tol.isClose(1000.0, 999.5));
            assertFalse(tol.isClose(1000.0, 1002.0));
        }

        @Test
        void combinedTolerance() {
            ToleranceConfig tol = new ToleranceConfig(1e-5, 1e-3);
            // Large value: relative tolerance dominates
            assertTrue(tol.isClose(1000.0, 1000.5));
            // Small value: absolute tolerance dominates
            assertTrue(tol.isClose(0.00001, 0.000015));
        }

        @Test
        void nanValues() {
            ToleranceConfig tol = new ToleranceConfig(1e-5, 1e-5);
            // NaN == NaN returns true (special case for testing)
            assertTrue(tol.isClose(Double.NaN, Double.NaN));
            // NaN != any number
            assertFalse(tol.isClose(Double.NaN, 1.0));
            assertFalse(tol.isClose(1.0, Double.NaN));
        }

        @Test
        void infinityValues() {
            ToleranceConfig tol = new ToleranceConfig(1e-5, 1e-5);
            assertTrue(tol.isClose(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY));
            assertTrue(tol.isClose(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY));
            // Different sign infinities are not close
            assertFalse(tol.isClose(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY));
            // Infinity != finite
            assertFalse(tol.isClose(Double.POSITIVE_INFINITY, 1e308));
        }
    }

    @Nested
    @DisplayName("forOp")
    class ForOpTests {

        @Test
        void elementwiseOps() {
            ToleranceConfig addTol = ToleranceConfig.forOp("add");
            assertEquals(1e-6, addTol.atol(), 1e-10);
            assertEquals(1e-5, addTol.rtol(), 1e-10);
        }

        @Test
        void transcendentalOps() {
            ToleranceConfig expTol = ToleranceConfig.forOp("exp");
            assertEquals(1e-5, expTol.atol(), 1e-10);
            assertEquals(1e-4, expTol.rtol(), 1e-10);
        }

        @Test
        void matrixOps() {
            ToleranceConfig dotTol = ToleranceConfig.forOp("dot_general");
            assertEquals(1e-4, dotTol.atol(), 1e-10);
            assertEquals(1e-3, dotTol.rtol(), 1e-10);
        }

        @Test
        void exactOps() {
            ToleranceConfig reshapeTol = ToleranceConfig.forOp("reshape");
            assertEquals(0.0, reshapeTol.atol());
            assertEquals(0.0, reshapeTol.rtol());
        }

        @Test
        void stablehloPrefix() {
            // Should work with or without stablehlo. prefix
            ToleranceConfig tol1 = ToleranceConfig.forOp("add");
            ToleranceConfig tol2 = ToleranceConfig.forOp("stablehlo.add");
            assertEquals(tol1.atol(), tol2.atol());
            assertEquals(tol1.rtol(), tol2.rtol());
        }

        @Test
        void caseInsensitive() {
            ToleranceConfig tol1 = ToleranceConfig.forOp("ADD");
            ToleranceConfig tol2 = ToleranceConfig.forOp("add");
            assertEquals(tol1.atol(), tol2.atol());
        }

        @Test
        void unknownOpGetsDefault() {
            ToleranceConfig tol = ToleranceConfig.forOp("unknown_op");
            assertEquals(1e-5, tol.atol(), 1e-10);
            assertEquals(1e-4, tol.rtol(), 1e-10);
        }
    }

    @Nested
    @DisplayName("forDtype")
    class ForDtypeTests {

        @Test
        void float32() {
            ToleranceConfig tol = ToleranceConfig.forDtype(ScalarType.F32);
            assertEquals(1e-5, tol.atol(), 1e-10);
            assertEquals(1e-4, tol.rtol(), 1e-10);
        }

        @Test
        void float64() {
            ToleranceConfig tol = ToleranceConfig.forDtype(ScalarType.F64);
            assertEquals(1e-10, tol.atol(), 1e-15);
            assertEquals(1e-9, tol.rtol(), 1e-14);
        }

        @Test
        void float16() {
            ToleranceConfig tol = ToleranceConfig.forDtype(ScalarType.F16);
            assertEquals(1e-2, tol.atol(), 1e-5);
            assertEquals(1e-2, tol.rtol(), 1e-5);
        }

        @Test
        void integers() {
            ToleranceConfig tol = ToleranceConfig.forDtype(ScalarType.I32);
            assertEquals(0.0, tol.atol());
            assertEquals(0.0, tol.rtol());
        }
    }

    @Nested
    @DisplayName("forOp with dtype")
    class ForOpWithDtypeTests {

        @Test
        void takesMaxTolerance() {
            // F16 has higher tolerance than add
            ToleranceConfig tol = ToleranceConfig.forOp("add", ScalarType.F16);
            assertEquals(1e-2, tol.atol(), 1e-5); // F16 dominates
            assertEquals(1e-2, tol.rtol(), 1e-5); // F16 dominates
        }

        @Test
        void opCanDominate() {
            // dot_general has higher tolerance than F64
            ToleranceConfig tol = ToleranceConfig.forOp("dot_general", ScalarType.F64);
            assertEquals(1e-4, tol.atol(), 1e-10); // dot_general dominates
        }
    }

    @Nested
    @DisplayName("combinators")
    class CombinatorTests {

        @Test
        void scaled() {
            ToleranceConfig original = new ToleranceConfig(1e-5, 1e-4);
            ToleranceConfig scaled = original.scaled(10);
            assertEquals(1e-4, scaled.atol(), 1e-10);
            assertEquals(1e-3, scaled.rtol(), 1e-10);
        }

        @Test
        void or() {
            ToleranceConfig a = new ToleranceConfig(1e-5, 1e-3);
            ToleranceConfig b = new ToleranceConfig(1e-4, 1e-4);
            ToleranceConfig combined = a.or(b);
            assertEquals(1e-4, combined.atol(), 1e-10); // max
            assertEquals(1e-3, combined.rtol(), 1e-10); // max
        }

        @Test
        void and() {
            ToleranceConfig a = new ToleranceConfig(1e-5, 1e-3);
            ToleranceConfig b = new ToleranceConfig(1e-4, 1e-4);
            ToleranceConfig combined = a.and(b);
            assertEquals(1e-5, combined.atol(), 1e-10); // min
            assertEquals(1e-4, combined.rtol(), 1e-10); // min
        }
    }

    @Test
    void constants() {
        assertEquals(0.0, ToleranceConfig.STRICT.atol());
        assertEquals(0.0, ToleranceConfig.STRICT.rtol());

        assertEquals(1e-2, ToleranceConfig.LOOSE.atol(), 1e-5);
        assertEquals(1e-2, ToleranceConfig.LOOSE.rtol(), 1e-5);
    }

    @Test
    void toStringFormat() {
        String str = new ToleranceConfig(1e-5, 1e-4).toString();
        assertTrue(str.contains("atol"));
        assertTrue(str.contains("rtol"));
    }
}
