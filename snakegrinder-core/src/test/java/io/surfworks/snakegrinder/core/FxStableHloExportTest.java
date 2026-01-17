package io.surfworks.snakegrinder.core;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for FxStableHloExport - tests pure Java functionality without GraalPy.
 *
 * These tests run in JVM mode and don't require PyTorch or GraalPy.
 * For integration tests that exercise FX tracing, see FxStableHloIntegrationTest.
 *
 * Run: ./gradlew :snakegrinder-core:test
 */
class FxStableHloExportTest {

    // ========================================================================
    // InputSpec Unit Tests
    // ========================================================================

    @Nested
    class InputSpecTests {

        @Test
        void inputSpecDefaultDtype() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(new int[]{4, 8});
            assertEquals("f32", spec.dtype, "Default dtype should be f32");
        }

        @Test
        void inputSpecWithCustomDtype() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(new int[]{4, 8}, "f64");
            assertEquals("f64", spec.dtype);
        }

        @Test
        void inputSpecNullDtypeDefaultsToF32() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(new int[]{4, 8}, null);
            assertEquals("f32", spec.dtype, "Null dtype should default to f32");
        }

        @Test
        void inputSpecToPythonTuple1D() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(new int[]{8});
            String tuple = spec.toPythonTuple();
            assertEquals("((8,), 'f32')", tuple, "1D shape should have trailing comma");
        }

        @Test
        void inputSpecToPythonTuple2D() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(new int[]{4, 8});
            String tuple = spec.toPythonTuple();
            assertEquals("((4, 8), 'f32')", tuple);
        }

        @Test
        void inputSpecToPythonTuple3D() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(new int[]{2, 4, 8});
            String tuple = spec.toPythonTuple();
            assertEquals("((2, 4, 8), 'f32')", tuple);
        }

        @Test
        void inputSpecToPythonTuple4D() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(new int[]{1, 3, 224, 224});
            String tuple = spec.toPythonTuple();
            assertEquals("((1, 3, 224, 224), 'f32')", tuple);
        }

        @Test
        void inputSpecWithF16Dtype() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(new int[]{4, 8}, "f16");
            String tuple = spec.toPythonTuple();
            assertEquals("((4, 8), 'f16')", tuple);
        }

        @Test
        void inputSpecWithI32Dtype() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(new int[]{4, 8}, "i32");
            String tuple = spec.toPythonTuple();
            assertEquals("((4, 8), 'i32')", tuple);
        }

        @Test
        void inputSpecShapeIsStored() {
            int[] shape = new int[]{1, 2, 3};
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(shape);
            assertArrayEquals(shape, spec.shape);
        }

        @Test
        void inputSpecEmptyShape() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(new int[]{});
            String tuple = spec.toPythonTuple();
            assertEquals("((), 'f32')", tuple, "Empty shape should produce ()");
        }
    }

    // ========================================================================
    // TraceResult Unit Tests
    // ========================================================================

    @Nested
    class TraceResultTests {

        @Test
        void traceResultOkHasSuccessTrue() {
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.ok(
                    "module @main {}", null, Map.of());
            assertTrue(result.success);
        }

        @Test
        void traceResultOkHasMlir() {
            String mlir = "module @main {}";
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.ok(
                    mlir, null, Map.of());
            assertEquals(mlir, result.mlir);
        }

        @Test
        void traceResultOkHasFxGraph() {
            String fxGraph = "graph():...";
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.ok(
                    "mlir", fxGraph, Map.of());
            assertEquals(fxGraph, result.fxGraph);
        }

        @Test
        void traceResultOkHasMetadata() {
            Map<String, Object> metadata = Map.of("key", "value");
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.ok(
                    "mlir", null, metadata);
            assertEquals("value", result.metadata.get("key"));
        }

        @Test
        void traceResultOkHasNoError() {
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.ok(
                    "mlir", null, Map.of());
            assertNull(result.error);
            assertNull(result.traceback);
        }

        @Test
        void traceResultOkHasEmptyWarnings() {
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.ok(
                    "mlir", null, Map.of());
            assertNotNull(result.warnings);
            assertTrue(result.warnings.isEmpty());
        }

        @Test
        void traceResultFailHasSuccessFalse() {
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.fail(
                    "error message", "traceback");
            assertFalse(result.success);
        }

        @Test
        void traceResultFailHasError() {
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.fail(
                    "error message", "traceback");
            assertEquals("error message", result.error);
            assertEquals("traceback", result.traceback);
        }

        @Test
        void traceResultFailHasNoMlir() {
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.fail(
                    "error", null);
            assertNull(result.mlir);
            assertNull(result.fxGraph);
        }

        @Test
        void traceResultFailHasEmptyMetadata() {
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.fail(
                    "error", null);
            assertNotNull(result.metadata);
            assertTrue(result.metadata.isEmpty());
        }

        @Test
        void traceResultOkWithNullMetadataDefaultsToEmpty() {
            FxStableHloExport.TraceResult result = FxStableHloExport.TraceResult.ok(
                    "mlir", null, null);
            assertNotNull(result.metadata);
            assertTrue(result.metadata.isEmpty());
        }
    }

    // ========================================================================
    // Resource Loading Tests
    // ========================================================================

    @Nested
    class ResourceLoadingTests {

        @Test
        void fxToStableHloPyResourceExists() {
            var stream = FxStableHloExport.class.getResourceAsStream("/snakegrinder/fx_to_stablehlo.py");
            assertNotNull(stream, "fx_to_stablehlo.py should be on classpath");
        }

        @Test
        void bootstrapPyResourceExists() {
            var stream = FxStableHloExport.class.getResourceAsStream("/snakegrinder/bootstrap.py");
            assertNotNull(stream, "bootstrap.py should be on classpath");
        }
    }

    // ========================================================================
    // Input Validation Tests
    // ========================================================================

    @Nested
    class InputValidationTests {

        @Test
        void inputSpecListToPythonFormat() {
            List<FxStableHloExport.InputSpec> inputs = List.of(
                    new FxStableHloExport.InputSpec(new int[]{1, 8}),
                    new FxStableHloExport.InputSpec(new int[]{8, 16})
            );

            // Verify each spec can generate Python tuple
            assertEquals("((1, 8), 'f32')", inputs.get(0).toPythonTuple());
            assertEquals("((8, 16), 'f32')", inputs.get(1).toPythonTuple());
        }

        @Test
        void inputSpecWithLargeDimensions() {
            FxStableHloExport.InputSpec spec = new FxStableHloExport.InputSpec(
                    new int[]{1024, 768, 512});
            String tuple = spec.toPythonTuple();
            assertTrue(tuple.contains("1024"));
            assertTrue(tuple.contains("768"));
            assertTrue(tuple.contains("512"));
        }
    }
}
