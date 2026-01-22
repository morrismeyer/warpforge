package io.surfworks.warpforge.backend.amd;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.amd.ops.HipOpDispatcher;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for HipOpDispatcher.
 *
 * <p>These tests verify the dispatcher correctly routes operations
 * and reports supported operations.
 */
@DisplayName("HIP Op Dispatcher Tests")
class HipOpDispatcherTest {

    private HipOpDispatcher dispatcher;

    @BeforeEach
    void setUp() {
        // Create dispatcher in stub mode (no HIP context)
        dispatcher = new HipOpDispatcher();
    }

    // ==================== Operation Support Tests ====================

    @Test
    @DisplayName("Dispatcher reports supported operations")
    void testSupportedOperations() {
        List<String> ops = dispatcher.supportedOps();

        assertNotNull(ops);
        assertFalse(ops.isEmpty());

        // Check for common operations
        assertTrue(ops.contains("Add"));
        assertTrue(ops.contains("Multiply"));
        assertTrue(ops.contains("Dot"));
        assertTrue(ops.contains("Subtract"));
        assertTrue(ops.contains("Divide"));
        assertTrue(ops.contains("Negate"));
        assertTrue(ops.contains("Exp"));
        assertTrue(ops.contains("Tanh"));
        assertTrue(ops.contains("Reshape"));
        assertTrue(ops.contains("Transpose"));
    }

    @Test
    @DisplayName("Dispatcher supports StableHLO operations")
    void testSupportsOperation() {
        // Create sample operations
        StableHloAst.TensorType type = new StableHloAst.TensorType(
            List.of(4), StableHloAst.ScalarType.F32);
        StableHloAst.Value v1 = new StableHloAst.Value("0", type);
        StableHloAst.Value v2 = new StableHloAst.Value("1", type);
        StableHloAst.Value result = new StableHloAst.Value("2", type);

        StableHloAst.AddOp addOp = new StableHloAst.AddOp(v1, v2, result, type);
        StableHloAst.MultiplyOp mulOp = new StableHloAst.MultiplyOp(v1, v2, result, type);

        assertTrue(dispatcher.supports(addOp));
        assertTrue(dispatcher.supports(mulOp));
    }

    // ==================== Stub Behavior Tests ====================

    @Test
    @DisplayName("Stub operations throw UnsupportedOperationException with helpful message")
    void testStubThrowsWithMessage() {
        StableHloAst.TensorType type = new StableHloAst.TensorType(
            List.of(4), StableHloAst.ScalarType.F32);
        StableHloAst.Value v1 = new StableHloAst.Value("0", type);
        StableHloAst.Value v2 = new StableHloAst.Value("1", type);
        StableHloAst.Value result = new StableHloAst.Value("2", type);

        StableHloAst.AddOp addOp = new StableHloAst.AddOp(v1, v2, result, type);

        UnsupportedOperationException ex = assertThrows(
            UnsupportedOperationException.class,
            () -> dispatcher.dispatch(addOp, List.of())
        );

        // Verify the message is helpful
        String msg = ex.getMessage();
        assertNotNull(msg);
        assertTrue(msg.contains("HIPRTC"), "Message should mention HIPRTC");
        assertTrue(msg.contains("CPU backend") || msg.contains("reference"),
            "Message should suggest alternative");
    }

    @Test
    @DisplayName("DotOp stub mentions rocBLAS")
    void testDotOpStubMentionsRocblas() {
        StableHloAst.TensorType aType = new StableHloAst.TensorType(
            List.of(2, 3), StableHloAst.ScalarType.F32);
        StableHloAst.TensorType bType = new StableHloAst.TensorType(
            List.of(3, 2), StableHloAst.ScalarType.F32);
        StableHloAst.TensorType resultType = new StableHloAst.TensorType(
            List.of(2, 2), StableHloAst.ScalarType.F32);

        StableHloAst.DotOp dotOp = new StableHloAst.DotOp(
            new StableHloAst.Value("lhs", aType),
            new StableHloAst.Value("rhs", bType),
            new StableHloAst.Value("result", resultType),
            resultType
        );

        UnsupportedOperationException ex = assertThrows(
            UnsupportedOperationException.class,
            () -> dispatcher.dispatch(dotOp, List.of())
        );

        String msg = ex.getMessage();
        assertTrue(msg.contains("rocBLAS") || msg.contains("HIPRTC"),
            "DotOp stub should mention rocBLAS or HIPRTC");
    }

    // ==================== Real Implementation Detection ====================

    @Test
    @DisplayName("Stub mode has no real implementations")
    void testStubModeNoRealImplementations() {
        // In stub mode (no HIP context), there should be no real implementations
        assertFalse(dispatcher.hasRealImplementation(StableHloAst.AddOp.class));
        assertFalse(dispatcher.hasRealImplementation(StableHloAst.MultiplyOp.class));
        assertFalse(dispatcher.hasRealImplementation(StableHloAst.DotOp.class));
    }

    // ==================== Operation Coverage ====================

    @Test
    @DisplayName("All major operation categories are registered")
    void testAllCategoriesRegistered() {
        List<String> ops = dispatcher.supportedOps();

        // Binary elementwise
        assertTrue(ops.contains("Add"));
        assertTrue(ops.contains("Subtract"));
        assertTrue(ops.contains("Multiply"));
        assertTrue(ops.contains("Divide"));
        assertTrue(ops.contains("Maximum"));
        assertTrue(ops.contains("Minimum"));
        assertTrue(ops.contains("Power"));

        // Unary elementwise
        assertTrue(ops.contains("Negate"));
        assertTrue(ops.contains("Abs"));
        assertTrue(ops.contains("Exp"));
        assertTrue(ops.contains("Log"));
        assertTrue(ops.contains("Tanh"));
        assertTrue(ops.contains("Sqrt"));

        // Comparison
        assertTrue(ops.contains("Compare"));
        assertTrue(ops.contains("Select"));
        assertTrue(ops.contains("Clamp"));

        // Shape manipulation
        assertTrue(ops.contains("Reshape"));
        assertTrue(ops.contains("Transpose"));
        assertTrue(ops.contains("BroadcastInDim"));
        assertTrue(ops.contains("Concatenate"));
        assertTrue(ops.contains("Slice"));

        // Linear algebra
        assertTrue(ops.contains("Dot"));
        assertTrue(ops.contains("DotGeneral"));

        // Reduction
        assertTrue(ops.contains("Reduce"));

        // Type conversion
        assertTrue(ops.contains("Convert"));
    }

    @Test
    @DisplayName("Bitwise operations are registered")
    void testBitwiseOpsRegistered() {
        List<String> ops = dispatcher.supportedOps();

        assertTrue(ops.contains("And"));
        assertTrue(ops.contains("Or"));
        assertTrue(ops.contains("Xor"));
        assertTrue(ops.contains("Not"));
        assertTrue(ops.contains("ShiftLeft"));
        assertTrue(ops.contains("ShiftRightArithmetic"));
        assertTrue(ops.contains("ShiftRightLogical"));
    }
}
