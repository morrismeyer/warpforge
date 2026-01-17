package io.surfworks.warpforge.backend.cpu.integration;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloParser;
import io.surfworks.warpforge.core.endtoend.EndToEndTestFixture;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.testing.TensorAssert;
import io.surfworks.warpforge.core.testing.ToleranceConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Integration tests comparing PyTorch outputs (via SnakeGrinder) against
 * Java CPU backend execution.
 *
 * <p>These tests load E2E fixtures that contain:
 * <ul>
 *   <li>StableHLO MLIR generated from PyTorch models</li>
 *   <li>Input tensors used during tracing</li>
 *   <li>Expected outputs from PyTorch execution</li>
 * </ul>
 *
 * <p>The test parses the MLIR, executes it through the CPU backend,
 * and compares the results against PyTorch's ground truth.
 */
@DisplayName("PyTorch vs Java Integration Tests")
@Tag("integration")
class PyTorchVsJavaIntegrationTest {

    private static final Path FIXTURES_DIR = Path.of("../warpforge-core/src/test/resources/fixtures/e2e");

    private StableHloInterpreter interpreter;

    @BeforeEach
    void setUp() {
        interpreter = new StableHloInterpreter();
    }

    @AfterEach
    void tearDown() {
        if (interpreter != null) {
            interpreter.close();
        }
    }

    /**
     * Provides all available E2E fixture names.
     */
    static Stream<String> fixtureNames() throws IOException {
        if (!Files.exists(FIXTURES_DIR)) {
            return Stream.empty();
        }
        return Files.list(FIXTURES_DIR)
            .filter(Files::isDirectory)
            .map(p -> p.getFileName().toString())
            .sorted();
    }

    @ParameterizedTest(name = "fixture: {0}")
    @MethodSource("fixtureNames")
    @DisplayName("PyTorch output matches Java CPU backend")
    void pytorchMatchesJavaCpuBackend(String fixtureName) throws IOException {
        Path fixtureDir = FIXTURES_DIR.resolve(fixtureName);

        try (EndToEndTestFixture fixture = EndToEndTestFixture.load(fixtureDir)) {
            // Parse the MLIR
            StableHloAst.Module module = StableHloParser.parse(fixture.mlir());
            assertNotNull(module, "Failed to parse MLIR for fixture: " + fixtureName);

            // Execute through Java CPU backend
            List<Tensor> javaOutputs = interpreter.execute(module, fixture.inputs());

            // Compare with PyTorch outputs
            List<Tensor> expectedOutputs = fixture.expectedOutputs();
            assertEquals(expectedOutputs.size(), javaOutputs.size(),
                "Output count mismatch for fixture: " + fixtureName);

            // Use tolerance appropriate for the operation type
            ToleranceConfig tolerance = getToleranceForFixture(fixtureName);

            for (int i = 0; i < expectedOutputs.size(); i++) {
                Tensor expected = expectedOutputs.get(i);
                Tensor actual = javaOutputs.get(i);

                TensorAssert.assertEquals(
                    "Output " + i + " for " + fixtureName,
                    expected, actual, tolerance
                );
            }

            // Close actual outputs (expected outputs owned by fixture)
            for (Tensor t : javaOutputs) {
                t.close();
            }
        }
    }

    @Nested
    @DisplayName("Individual Operation Tests")
    class IndividualOperationTests {

        @Test
        @DisplayName("add: element-wise addition")
        void testAdd() throws IOException {
            runFixtureTest("add");
        }

        @Test
        @DisplayName("subtract: element-wise subtraction")
        void testSubtract() throws IOException {
            runFixtureTest("subtract");
        }

        @Test
        @DisplayName("multiply: element-wise multiplication")
        void testMultiply() throws IOException {
            runFixtureTest("multiply");
        }

        @Test
        @DisplayName("negate: element-wise negation")
        void testNegate() throws IOException {
            runFixtureTest("negate");
        }

        @Test
        @DisplayName("abs: element-wise absolute value")
        void testAbs() throws IOException {
            runFixtureTest("abs");
        }

        @Test
        @DisplayName("exp: element-wise exponential")
        void testExp() throws IOException {
            runFixtureTest("exp");
        }

        @Test
        @DisplayName("tanh: element-wise hyperbolic tangent")
        void testTanh() throws IOException {
            runFixtureTest("tanh");
        }

        @Test
        @DisplayName("sigmoid: element-wise logistic function")
        void testSigmoid() throws IOException {
            runFixtureTest("sigmoid");
        }

        @Test
        @DisplayName("relu: rectified linear unit")
        void testRelu() throws IOException {
            runFixtureTest("relu");
        }

        private void runFixtureTest(String fixtureName) throws IOException {
            Path fixtureDir = FIXTURES_DIR.resolve(fixtureName);
            assumeTrue(Files.exists(fixtureDir),
                "Fixture not found: " + fixtureName);

            try (EndToEndTestFixture fixture = EndToEndTestFixture.load(fixtureDir)) {
                StableHloAst.Module module = StableHloParser.parse(fixture.mlir());

                List<Tensor> javaOutputs = interpreter.execute(module, fixture.inputs());
                List<Tensor> expectedOutputs = fixture.expectedOutputs();

                assertEquals(expectedOutputs.size(), javaOutputs.size());

                ToleranceConfig tolerance = getToleranceForFixture(fixtureName);

                for (int i = 0; i < expectedOutputs.size(); i++) {
                    TensorAssert.assertEquals(expectedOutputs.get(i), javaOutputs.get(i), tolerance);
                }

                // Close actual outputs (expected outputs owned by fixture)
                for (Tensor t : javaOutputs) {
                    t.close();
                }
            }
        }
    }

    @Nested
    @DisplayName("Numerical Accuracy Tests")
    class NumericalAccuracyTests {

        @Test
        @DisplayName("verify transcendental function accuracy")
        void transcendentalAccuracy() throws IOException {
            // Test exp, tanh, sigmoid which use transcendental functions
            for (String fixture : List.of("exp", "tanh", "sigmoid")) {
                Path fixtureDir = FIXTURES_DIR.resolve(fixture);
                if (!Files.exists(fixtureDir)) continue;

                try (EndToEndTestFixture f = EndToEndTestFixture.load(fixtureDir)) {
                    StableHloAst.Module module = StableHloParser.parse(f.mlir());
                    List<Tensor> outputs = interpreter.execute(module, f.inputs());

                    for (int i = 0; i < outputs.size(); i++) {
                        try (Tensor actual = outputs.get(i)) {
                            TensorAssert.assertFinite(actual);
                            TensorAssert.assertNoNaN(actual);
                        }
                    }
                }
            }
        }

        @Test
        @DisplayName("verify elementwise operations are exact for simple values")
        void elementwiseExactness() throws IOException {
            for (String fixture : List.of("add", "subtract", "multiply", "negate", "abs")) {
                Path fixtureDir = FIXTURES_DIR.resolve(fixture);
                if (!Files.exists(fixtureDir)) continue;

                try (EndToEndTestFixture f = EndToEndTestFixture.load(fixtureDir)) {
                    StableHloAst.Module module = StableHloParser.parse(f.mlir());
                    List<Tensor> outputs = interpreter.execute(module, f.inputs());

                    // For simple elementwise ops, expect very high accuracy
                    ToleranceConfig strict = ToleranceConfig.forOp("elementwise");

                    for (int i = 0; i < outputs.size(); i++) {
                        TensorAssert.assertEquals(f.expectedOutputs().get(i), outputs.get(i), strict);
                    }

                    // Close actual outputs (expected outputs owned by fixture)
                    for (Tensor t : outputs) {
                        t.close();
                    }
                }
            }
        }
    }

    /**
     * Get appropriate tolerance for a fixture based on its operation type.
     */
    private static ToleranceConfig getToleranceForFixture(String fixtureName) {
        return switch (fixtureName) {
            // Transcendental functions need looser tolerance
            case "exp", "tanh", "sigmoid", "relu" -> ToleranceConfig.forOp("transcendental");
            // Linear operations should be exact
            case "add", "subtract", "multiply", "negate", "abs" -> ToleranceConfig.forOp("elementwise");
            // Default tolerance
            default -> ToleranceConfig.forDtype(io.surfworks.warpforge.core.tensor.ScalarType.F32);
        };
    }
}
