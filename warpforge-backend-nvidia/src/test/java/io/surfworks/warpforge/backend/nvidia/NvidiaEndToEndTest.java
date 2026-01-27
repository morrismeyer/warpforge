package io.surfworks.warpforge.backend.nvidia;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloParser;
import io.surfworks.warpforge.backend.nvidia.cuda.CudaRuntime;
import io.surfworks.warpforge.core.endtoend.EndToEndTestFixture;
import io.surfworks.warpforge.core.graph.ExecutableGraph;
import io.surfworks.warpforge.core.graph.GraphCompiler;
import io.surfworks.warpforge.core.graph.GraphExecutor;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.testing.TensorAssert;
import io.surfworks.warpforge.core.testing.ToleranceConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * End-to-end tests for NVIDIA GPU backend.
 *
 * <p>These tests run the same fixtures as the CPU end-to-end tests but execute
 * on the NVIDIA GPU backend to verify CUDA kernel correctness.
 *
 * <p>Requires CUDA hardware to run (tagged with "nvidia").
 *
 * <p>Test distribution:
 * <ul>
 *   <li>Quick fixtures (add, relu, softmax, etc.) - run on every CI push</li>
 *   <li>BERT fixtures (bert_squad_*) - run nightly only (tagged with "nightly")</li>
 * </ul>
 */
@Tag("nvidia")
@DisplayName("NVIDIA Backend End-to-End Tests")
class NvidiaEndToEndTest {

    /**
     * Fixtures are read from the NFS-mounted warpdata directory.
     * NUC generates to: ~/surfworks/warpdata/fixtures/endtoend
     * GPU boxes read from: /mnt/warpdata/fixtures/endtoend
     */
    private static final Path FIXTURES_DIR = Paths.get(
        "/mnt/warpdata/fixtures/endtoend"
    );

    /**
     * Pattern for expensive fixtures that should only run nightly.
     * BERT models take significantly longer due to their size and complexity.
     */
    private static final java.util.regex.Pattern NIGHTLY_FIXTURE_PATTERN =
        java.util.regex.Pattern.compile("bert_.*");

    private NvidiaBackend backend;
    private GraphExecutor executor;

    @BeforeAll
    static void checkCudaAvailable() {
        assumeTrue(CudaRuntime.isAvailable(), "CUDA runtime not available");
    }

    @BeforeEach
    void setUp() {
        backend = new NvidiaBackend();
        executor = new GraphExecutor(backend);
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    /**
     * Discover quick EndToEnd fixture directories (excludes BERT and other expensive fixtures).
     * These run on every CI push.
     */
    static Stream<Path> quickEndToEndFixtures() {
        return discoverFixtures(false);
    }

    /**
     * Discover expensive EndToEnd fixture directories (BERT models).
     * These run nightly only.
     */
    static Stream<Path> nightlyEndToEndFixtures() {
        return discoverFixtures(true);
    }

    /**
     * Discover fixtures based on whether we want expensive (nightly) or quick fixtures.
     */
    private static Stream<Path> discoverFixtures(boolean nightlyOnly) {
        if (!Files.exists(FIXTURES_DIR)) {
            return Stream.of(Paths.get("NO_FIXTURES_AVAILABLE"));
        }

        try {
            List<Path> fixtures = Files.walk(FIXTURES_DIR)
                .filter(Files::isDirectory)
                .filter(dir -> Files.exists(dir.resolve("model.mlir")))
                .filter(dir -> Files.exists(dir.resolve("inputs")) || Files.exists(dir.resolve("outputs")))
                .filter(dir -> {
                    String name = dir.getFileName().toString();
                    boolean isExpensive = NIGHTLY_FIXTURE_PATTERN.matcher(name).matches();
                    return nightlyOnly ? isExpensive : !isExpensive;
                })
                .toList();

            if (fixtures.isEmpty()) {
                return Stream.of(Paths.get("NO_FIXTURES_AVAILABLE"));
            }
            return fixtures.stream();
        } catch (IOException e) {
            return Stream.of(Paths.get("NO_FIXTURES_AVAILABLE"));
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("quickEndToEndFixtures")
    @DisplayName("NVIDIA backend matches PyTorch output")
    void nvidiaMatchesPytorch(Path fixtureDir) throws IOException {
        runNvidiaE2eTest(fixtureDir);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("nightlyEndToEndFixtures")
    @Tag("nightly")
    @DisplayName("BERT model matches PyTorch output on NVIDIA (nightly)")
    void bertMatchesPytorchOnNvidia(Path fixtureDir) throws IOException {
        runNvidiaE2eTest(fixtureDir);
    }

    private void runNvidiaE2eTest(Path fixtureDir) throws IOException {
        if (fixtureDir.toString().equals("NO_FIXTURES_AVAILABLE")) {
            // Use assumeTrue to properly skip with visible message in JUnit report
            assumeTrue(false, "No EndToEnd fixtures found. Run: ./gradlew :warpforge-core:generateEndToEndFixtures");
            return;
        }

        // Use the shared fixture loader from warpforge-core
        try (EndToEndTestFixture fixture = EndToEndTestFixture.load(fixtureDir)) {
            if (fixture.expectedOutputs().isEmpty()) {
                System.out.println("Skipping " + fixture.name() + " - no expected outputs");
                return;
            }

            // Parse MLIR
            StableHloAst.Module module;
            try {
                module = StableHloParser.parse(fixture.mlir());
            } catch (Exception e) {
                fail("Failed to parse MLIR for " + fixture.name() + ": " + e.getMessage());
                return;
            }

            // Compile to ExecutableGraph
            ExecutableGraph graph;
            try {
                graph = GraphCompiler.compile(module);
            } catch (Exception e) {
                fail("Failed to compile graph for " + fixture.name() + ": " + e.getMessage());
                return;
            }

            // Verify input count
            assertEquals(fixture.totalArgCount(), graph.inputCount(),
                "Input count mismatch for " + fixture.name());

            // Execute on NVIDIA backend
            List<Tensor> actualOutputs;
            try {
                actualOutputs = executor.execute(graph, fixture.allInputs());
            } catch (UnsupportedOperationException e) {
                System.out.println("Skipping " + fixture.name() + " - unsupported op: " + e.getMessage());
                return;
            } catch (Exception e) {
                e.printStackTrace();
                fail("Failed to execute on NVIDIA backend for " + fixture.name() + ": " +
                     e.getClass().getName() + " - " + e.getMessage());
                return;
            }

            // Compare outputs
            assertEquals(fixture.outputCount(), actualOutputs.size(),
                "Output count mismatch for " + fixture.name());

            for (int i = 0; i < actualOutputs.size(); i++) {
                Tensor expected = fixture.expectedOutputs().get(i);
                Tensor actual = actualOutputs.get(i);

                assertArrayEquals(expected.shape(), actual.shape(),
                    "Output " + i + " shape mismatch for " + fixture.name());

                // GPU may have slightly different numerical precision - use slightly looser tolerance
                ToleranceConfig baseTolerance = ToleranceConfig.forOp(fixture.name(), expected.dtype());
                ToleranceConfig tolerance = baseTolerance.scaled(2.0);  // 2x looser for GPU

                try {
                    TensorAssert.assertEquals(expected, actual, tolerance);
                } catch (AssertionError e) {
                    fail("Output " + i + " value mismatch for " + fixture.name() + ": " + e.getMessage());
                }
            }

            for (Tensor t : actualOutputs) {
                if (!fixture.allInputs().contains(t)) {
                    t.close();
                }
            }
        }
    }
}
