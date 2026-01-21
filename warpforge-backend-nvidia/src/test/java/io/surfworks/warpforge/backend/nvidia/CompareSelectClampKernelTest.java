package io.surfworks.warpforge.backend.nvidia;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaKernels;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaRuntime;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for comparison and selection CUDA kernels.
 *
 * <p>Operations: Compare (EQ, NE, LT, LE, GT, GE), Select, Clamp
 *
 * <p>Each test prints [TEST] and [PASS] markers for visibility in CI logs.
 */
@DisplayName("Comparison and Selection CUDA Kernels")
class CompareSelectClampKernelTest {

    private NvidiaBackend backend;

    @BeforeEach
    void setUp() {
        backend = new NvidiaBackend(0, CudaKernels.SALT_NONE);
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    // ==================== PTX Generation Tests (No CUDA Required) ====================

    @Test
    @DisplayName("PTX: Compare EQ generates valid output")
    void testCompareEqPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Compare EQ");
        String ptx = CudaKernels.generateCompareF32("EQ", CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry compare_eq_f32"));
        assertTrue(ptx.contains("setp.eq.f32"));
        assertTrue(ptx.contains("selp.f32"));
        System.out.println("[PASS] Compare EQ PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Compare NE generates valid output")
    void testCompareNePtxGeneration() {
        System.out.println("[TEST] PTX Generation: Compare NE");
        String ptx = CudaKernels.generateCompareF32("NE", CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry compare_ne_f32"));
        assertTrue(ptx.contains("setp.ne.f32"));
        System.out.println("[PASS] Compare NE PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Compare LT generates valid output")
    void testCompareLtPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Compare LT");
        String ptx = CudaKernels.generateCompareF32("LT", CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry compare_lt_f32"));
        assertTrue(ptx.contains("setp.lt.f32"));
        System.out.println("[PASS] Compare LT PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Compare LE generates valid output")
    void testCompareLePtxGeneration() {
        System.out.println("[TEST] PTX Generation: Compare LE");
        String ptx = CudaKernels.generateCompareF32("LE", CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry compare_le_f32"));
        assertTrue(ptx.contains("setp.le.f32"));
        System.out.println("[PASS] Compare LE PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Compare GT generates valid output")
    void testCompareGtPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Compare GT");
        String ptx = CudaKernels.generateCompareF32("GT", CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry compare_gt_f32"));
        assertTrue(ptx.contains("setp.gt.f32"));
        System.out.println("[PASS] Compare GT PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Compare GE generates valid output")
    void testCompareGePtxGeneration() {
        System.out.println("[TEST] PTX Generation: Compare GE");
        String ptx = CudaKernels.generateCompareF32("GE", CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry compare_ge_f32"));
        assertTrue(ptx.contains("setp.ge.f32"));
        System.out.println("[PASS] Compare GE PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Select generates valid output")
    void testSelectPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Select");
        String ptx = CudaKernels.generateSelectF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry select_f32"));
        assertTrue(ptx.contains("pred_ptr"));
        assertTrue(ptx.contains("on_true_ptr"));
        assertTrue(ptx.contains("on_false_ptr"));
        assertTrue(ptx.contains("selp.f32"));
        System.out.println("[PASS] Select PTX generation OK");
    }

    @Test
    @DisplayName("PTX: Clamp generates valid output")
    void testClampPtxGeneration() {
        System.out.println("[TEST] PTX Generation: Clamp");
        String ptx = CudaKernels.generateClampF32(CudaKernels.SALT_NONE);

        assertNotNull(ptx);
        assertTrue(ptx.contains(".visible .entry clamp_f32"));
        assertTrue(ptx.contains("min_ptr"));
        assertTrue(ptx.contains("operand_ptr"));
        assertTrue(ptx.contains("max_ptr"));
        assertTrue(ptx.contains("min.f32"));
        assertTrue(ptx.contains("max.f32"));
        System.out.println("[PASS] Clamp PTX generation OK");
    }

    @Test
    @DisplayName("PTX: All comparison operations support SALT_TIMING")
    void testAllComparisonOperationsSupportTiming() {
        System.out.println("[TEST] PTX Generation: All comparison operations with SALT_TIMING");

        String[] directions = {"EQ", "NE", "LT", "LE", "GT", "GE"};
        for (String dir : directions) {
            String ptx = CudaKernels.generateCompareF32(dir, CudaKernels.SALT_TIMING);
            assertTrue(ptx.contains("timing_ptr"), "Compare " + dir + " should have timing_ptr");
            assertTrue(ptx.contains("%globaltimer"), "Compare " + dir + " should use globaltimer");
            System.out.println("  [OK] Compare " + dir + " supports SALT_TIMING");
        }

        String selectPtx = CudaKernels.generateSelectF32(CudaKernels.SALT_TIMING);
        assertTrue(selectPtx.contains("timing_ptr"), "Select should have timing_ptr");
        System.out.println("  [OK] Select supports SALT_TIMING");

        String clampPtx = CudaKernels.generateClampF32(CudaKernels.SALT_TIMING);
        assertTrue(clampPtx.contains("timing_ptr"), "Clamp should have timing_ptr");
        System.out.println("  [OK] Clamp supports SALT_TIMING");

        System.out.println("[PASS] All comparison operations support SALT_TIMING");
    }

    // ==================== CUDA Hardware Tests ====================

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Compare EQ executes correctly")
    void testCompareEqExecution() {
        System.out.println("[TEST] CUDA Execution: Compare EQ");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 5.0f, 7.0f, 8.0f};
        float[] b = {1.0f, 3.0f, 3.0f, 5.0f, 4.0f, 5.0f, 8.0f, 8.0f};
        float[] expected = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f};

        float[] result = executeCompare(a, b, StableHloAst.ComparisonDirection.EQ);
        assertArrayEquals(expected, result, 1e-5f);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Compare EQ execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Compare LT executes correctly")
    void testCompareLtExecution() {
        System.out.println("[TEST] CUDA Execution: Compare LT");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 5.0f, 7.0f, 8.0f};
        float[] b = {2.0f, 2.0f, 2.0f, 5.0f, 4.0f, 6.0f, 8.0f, 7.0f};
        // a < b: 1<2=T, 2<2=F, 3<2=F, 4<5=T, 5<4=F, 5<6=T, 7<8=T, 8<7=F
        float[] expected = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f};

        float[] result = executeCompare(a, b, StableHloAst.ComparisonDirection.LT);
        assertArrayEquals(expected, result, 1e-5f);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Compare LT execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Compare GT executes correctly")
    void testCompareGtExecution() {
        System.out.println("[TEST] CUDA Execution: Compare GT");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 5.0f, 7.0f, 8.0f};
        float[] b = {0.0f, 2.0f, 4.0f, 3.0f, 5.0f, 4.0f, 6.0f, 9.0f};
        // a > b: 1>0=T, 2>2=F, 3>4=F, 4>3=T, 5>5=F, 5>4=T, 7>6=T, 8>9=F
        float[] expected = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f};

        float[] result = executeCompare(a, b, StableHloAst.ComparisonDirection.GT);
        assertArrayEquals(expected, result, 1e-5f);

        System.out.println("  Input A:   " + Arrays.toString(a));
        System.out.println("  Input B:   " + Arrays.toString(b));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Compare GT execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Select executes correctly")
    void testSelectExecution() {
        System.out.println("[TEST] CUDA Execution: Select");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] pred = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
        float[] onTrue = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
        float[] onFalse = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float[] expected = {10.0f, 2.0f, 30.0f, 4.0f, 50.0f, 6.0f, 70.0f, 8.0f};

        float[] result = executeSelect(pred, onTrue, onFalse);
        assertArrayEquals(expected, result, 1e-5f);

        System.out.println("  Pred:      " + Arrays.toString(pred));
        System.out.println("  On True:   " + Arrays.toString(onTrue));
        System.out.println("  On False:  " + Arrays.toString(onFalse));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Select execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Clamp executes correctly")
    void testClampExecution() {
        System.out.println("[TEST] CUDA Execution: Clamp");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] min = {0.0f, 0.0f, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f, -1.0f};
        float[] operand = {-1.0f, 0.5f, 1.0f, 2.0f, -2.0f, 0.0f, 0.5f, 2.0f};
        float[] max = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        // clamp: max(min, min(operand, max))
        // [-1,0,1] -> [0,0.5,1,1] ; [-2,-1,1] -> [-1,0,0.5,1]
        float[] expected = {0.0f, 0.5f, 1.0f, 1.0f, -1.0f, 0.0f, 0.5f, 1.0f};

        float[] result = executeClamp(min, operand, max);
        assertArrayEquals(expected, result, 1e-5f);

        System.out.println("  Min:       " + Arrays.toString(min));
        System.out.println("  Operand:   " + Arrays.toString(operand));
        System.out.println("  Max:       " + Arrays.toString(max));
        System.out.println("  Result:    " + Arrays.toString(result));
        System.out.println("[PASS] Clamp execution OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: Compare and Select combined (like ReLU)")
    void testCompareSelectReLU() {
        System.out.println("[TEST] CUDA Execution: Compare + Select (ReLU pattern)");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        // ReLU(x) = max(0, x) = select(x > 0, x, 0)
        float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f, 3.0f};
        float[] zeros = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        // First compare: input > 0
        float[] pred = executeCompare(input, zeros, StableHloAst.ComparisonDirection.GT);
        // Then select: pred ? input : 0
        float[] result = executeSelect(pred, input, zeros);

        float[] expected = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.5f, 3.0f};
        assertArrayEquals(expected, result, 1e-5f);

        System.out.println("  Input:     " + Arrays.toString(input));
        System.out.println("  Pred (>0): " + Arrays.toString(pred));
        System.out.println("  ReLU:      " + Arrays.toString(result));
        System.out.println("[PASS] Compare + Select (ReLU) OK");
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA: All comparison directions work")
    void testAllComparisonDirections() {
        System.out.println("[TEST] CUDA Execution: All comparison directions");
        assumeTrue(CudaRuntime.isAvailable(), "CUDA not available");
        assumeTrue(backend.hasCudaContext(), "CUDA context not available");

        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {2.0f, 2.0f, 2.0f, 2.0f};

        // Test EQ
        float[] eq = executeCompare(a, b, StableHloAst.ComparisonDirection.EQ);
        assertArrayEquals(new float[]{0.0f, 1.0f, 0.0f, 0.0f}, eq, 1e-5f);
        System.out.println("  [OK] EQ: " + Arrays.toString(eq));

        // Test NE
        float[] ne = executeCompare(a, b, StableHloAst.ComparisonDirection.NE);
        assertArrayEquals(new float[]{1.0f, 0.0f, 1.0f, 1.0f}, ne, 1e-5f);
        System.out.println("  [OK] NE: " + Arrays.toString(ne));

        // Test LT
        float[] lt = executeCompare(a, b, StableHloAst.ComparisonDirection.LT);
        assertArrayEquals(new float[]{1.0f, 0.0f, 0.0f, 0.0f}, lt, 1e-5f);
        System.out.println("  [OK] LT: " + Arrays.toString(lt));

        // Test LE
        float[] le = executeCompare(a, b, StableHloAst.ComparisonDirection.LE);
        assertArrayEquals(new float[]{1.0f, 1.0f, 0.0f, 0.0f}, le, 1e-5f);
        System.out.println("  [OK] LE: " + Arrays.toString(le));

        // Test GT
        float[] gt = executeCompare(a, b, StableHloAst.ComparisonDirection.GT);
        assertArrayEquals(new float[]{0.0f, 0.0f, 1.0f, 1.0f}, gt, 1e-5f);
        System.out.println("  [OK] GT: " + Arrays.toString(gt));

        // Test GE
        float[] ge = executeCompare(a, b, StableHloAst.ComparisonDirection.GE);
        assertArrayEquals(new float[]{0.0f, 1.0f, 1.0f, 1.0f}, ge, 1e-5f);
        System.out.println("  [OK] GE: " + Arrays.toString(ge));

        System.out.println("[PASS] All comparison directions OK");
    }

    // ==================== Helper Methods ====================

    private float[] executeCompare(float[] a, float[] b, StableHloAst.ComparisonDirection direction) {
        int n = a.length;

        try (Tensor tensorA = Tensor.fromFloatArray(a, n);
             Tensor tensorB = Tensor.fromFloatArray(b, n)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n),
                StableHloAst.ScalarType.F32
            );

            StableHloAst.Value lhs = new StableHloAst.Value("0", resultType);
            StableHloAst.Value rhs = new StableHloAst.Value("1", resultType);
            StableHloAst.Value result = new StableHloAst.Value("2", resultType);

            StableHloAst.CompareOp op = new StableHloAst.CompareOp(
                result, lhs, rhs, direction, resultType
            );

            List<Tensor> results = backend.execute(op, List.of(tensorA, tensorB));
            return results.get(0).toFloatArray();
        }
    }

    private float[] executeSelect(float[] pred, float[] onTrue, float[] onFalse) {
        int n = pred.length;

        try (Tensor tensorPred = Tensor.fromFloatArray(pred, n);
             Tensor tensorOnTrue = Tensor.fromFloatArray(onTrue, n);
             Tensor tensorOnFalse = Tensor.fromFloatArray(onFalse, n)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n),
                StableHloAst.ScalarType.F32
            );

            StableHloAst.Value predVal = new StableHloAst.Value("0", resultType);
            StableHloAst.Value trueVal = new StableHloAst.Value("1", resultType);
            StableHloAst.Value falseVal = new StableHloAst.Value("2", resultType);
            StableHloAst.Value resultVal = new StableHloAst.Value("3", resultType);

            StableHloAst.SelectOp op = new StableHloAst.SelectOp(
                resultVal, predVal, trueVal, falseVal, resultType
            );

            List<Tensor> results = backend.execute(op, List.of(tensorPred, tensorOnTrue, tensorOnFalse));
            return results.get(0).toFloatArray();
        }
    }

    private float[] executeClamp(float[] min, float[] operand, float[] max) {
        int n = min.length;

        try (Tensor tensorMin = Tensor.fromFloatArray(min, n);
             Tensor tensorOperand = Tensor.fromFloatArray(operand, n);
             Tensor tensorMax = Tensor.fromFloatArray(max, n)) {

            StableHloAst.TensorType resultType = new StableHloAst.TensorType(
                List.of(n),
                StableHloAst.ScalarType.F32
            );

            StableHloAst.Value minVal = new StableHloAst.Value("0", resultType);
            StableHloAst.Value operandVal = new StableHloAst.Value("1", resultType);
            StableHloAst.Value maxVal = new StableHloAst.Value("2", resultType);
            StableHloAst.Value resultVal = new StableHloAst.Value("3", resultType);

            StableHloAst.ClampOp op = new StableHloAst.ClampOp(
                resultVal, minVal, operandVal, maxVal, resultType
            );

            List<Tensor> results = backend.execute(op, List.of(tensorMin, tensorOperand, tensorMax));
            return results.get(0).toFloatArray();
        }
    }
}
