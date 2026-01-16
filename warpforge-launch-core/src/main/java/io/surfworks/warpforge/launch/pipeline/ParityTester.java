package io.surfworks.warpforge.launch.pipeline;

import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.launch.config.LaunchConfig;
import io.surfworks.warpforge.launch.config.LaunchConfigLoader;
import io.surfworks.warpforge.launch.job.JobDefinition;
import io.surfworks.warpforge.launch.job.TensorResult;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Tests parity between WarpForge and native PyTorch execution.
 *
 * <p>Runs the same model through both paths and compares outputs:
 * <ul>
 *   <li><b>PyTorch path</b>: Direct Python/PyTorch execution</li>
 *   <li><b>WarpForge path</b>: SnakeGrinder → SnakeBurger → WarpForge</li>
 * </ul>
 *
 * <p>Results are compared element-by-element within a configurable tolerance
 * to account for floating-point precision differences.
 */
public final class ParityTester {

    /** Default tolerance for floating-point comparison */
    public static final double DEFAULT_TOLERANCE = 1e-5;

    private final LaunchConfig config;
    private final PipelineExecutor executor;
    private final double tolerance;
    private final Path workDir;

    /**
     * Creates a parity tester with default settings.
     */
    public ParityTester() {
        this(LaunchConfigLoader.load(), DEFAULT_TOLERANCE);
    }

    /**
     * Creates a parity tester with custom configuration.
     *
     * @param config    launch configuration
     * @param tolerance tolerance for floating-point comparison
     */
    public ParityTester(LaunchConfig config, double tolerance) {
        this.config = config;
        this.executor = new PipelineExecutor(config);
        this.tolerance = tolerance;
        this.workDir = Path.of(System.getProperty("java.io.tmpdir"), "warpforge-parity");
    }

    /**
     * Tests parity for a job definition.
     *
     * @param definition the model to test
     * @return parity test result
     */
    public ParityResult test(JobDefinition definition) {
        try {
            Files.createDirectories(workDir);

            // Run PyTorch
            Instant pytorchStart = Instant.now();
            List<double[]> pytorchOutputs = runPyTorch(definition);
            Duration pytorchTime = Duration.between(pytorchStart, Instant.now());

            // Run WarpForge
            Instant warpforgeStart = Instant.now();
            List<double[]> warpforgeOutputs = runWarpForge(definition);
            Duration warpforgeTime = Duration.between(warpforgeStart, Instant.now());

            // Compare outputs
            return compareOutputs(pytorchOutputs, warpforgeOutputs, pytorchTime, warpforgeTime);

        } catch (Exception e) {
            return ParityResult.error("Parity test failed: " + e.getMessage(), tolerance);
        }
    }

    /**
     * Tests parity using pre-computed PyTorch reference outputs.
     *
     * <p>Useful for regression testing against known-good outputs.
     *
     * @param definition        the model to test
     * @param referenceOutputs  pre-computed PyTorch outputs
     * @return parity test result
     */
    public ParityResult testAgainstReference(JobDefinition definition, List<double[]> referenceOutputs) {
        try {
            Files.createDirectories(workDir);

            // Run WarpForge
            Instant warpforgeStart = Instant.now();
            List<double[]> warpforgeOutputs = runWarpForge(definition);
            Duration warpforgeTime = Duration.between(warpforgeStart, Instant.now());

            // Compare against reference
            return compareOutputs(referenceOutputs, warpforgeOutputs, Duration.ZERO, warpforgeTime);

        } catch (Exception e) {
            return ParityResult.error("Parity test failed: " + e.getMessage(), tolerance);
        }
    }

    private List<double[]> runPyTorch(JobDefinition definition) throws PipelineException {
        Path snakegrinderBin = config.snakegrinderPath();
        if (snakegrinderBin == null) {
            throw new PipelineException("pytorch",
                    "snakegrinder binary not found (needed for PyTorch reference execution)");
        }

        String jobId = "parity-pytorch-" + definition.name().replaceAll("[^a-zA-Z0-9]", "-");
        Path outputDir = workDir.resolve(jobId);

        try {
            Files.createDirectories(outputDir);

            // Use snakegrinder with --pytorch-only flag to get reference outputs
            List<String> command = new ArrayList<>();
            command.add(snakegrinderBin.toString());
            command.add("--pytorch-only");
            command.add("--source");
            command.add(definition.modelSource().toString());
            command.add("--class");
            command.add(definition.modelClass());
            command.add("--inputs");
            command.add(definition.formatInputSpecs());
            command.add("--seed");
            command.add(String.valueOf(definition.seed()));
            command.add("--out");
            command.add(outputDir.toString());

            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            pb.directory(workDir.toFile());

            Process process = pb.start();

            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }

            boolean completed = process.waitFor(300, TimeUnit.SECONDS);
            if (!completed) {
                process.destroyForcibly();
                throw new PipelineException("pytorch", "Execution timed out");
            }

            int exitCode = process.exitValue();
            if (exitCode != 0) {
                throw new PipelineException("pytorch", "Exit code " + exitCode + ": " + output);
            }

            // Parse output values from snakegrinder JSON output
            return parseOutputs(outputDir);

        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new PipelineException("pytorch", "Execution failed: " + e.getMessage(), e);
        }
    }

    private List<double[]> runWarpForge(JobDefinition definition) throws PipelineException {
        PipelineResult result = executor.execute(definition);

        if (!result.success()) {
            throw new PipelineException("warpforge", result.errorMessage());
        }

        // Convert TensorResult outputs to double arrays
        List<double[]> outputs = new ArrayList<>();
        for (TensorResult tensor : result.outputs()) {
            if (tensor instanceof TensorResult.Inline inline) {
                // Extract values from the Tensor
                double[] values = extractTensorValues(inline.tensor());
                outputs.add(values);
            } else if (tensor instanceof TensorResult.FileRef fileRef) {
                // TODO: Load tensor data from file
                outputs.add(new double[0]);
            }
        }
        return outputs;
    }

    private ParityResult compareOutputs(
            List<double[]> pytorchOutputs,
            List<double[]> warpforgeOutputs,
            Duration pytorchTime,
            Duration warpforgeTime) {

        if (pytorchOutputs.size() != warpforgeOutputs.size()) {
            return ParityResult.error(
                    "Output count mismatch: PyTorch=" + pytorchOutputs.size() +
                            ", WarpForge=" + warpforgeOutputs.size(),
                    tolerance);
        }

        List<ParityResult.OutputDifference> differences = new ArrayList<>();
        boolean allMatch = true;

        for (int i = 0; i < pytorchOutputs.size(); i++) {
            double[] pytorch = pytorchOutputs.get(i);
            double[] warpforge = warpforgeOutputs.get(i);

            if (pytorch.length != warpforge.length) {
                return ParityResult.error(
                        "Output " + i + " size mismatch: PyTorch=" + pytorch.length +
                                ", WarpForge=" + warpforge.length,
                        tolerance);
            }

            double maxDiff = 0.0;
            double sumDiff = 0.0;
            int mismatches = 0;

            for (int j = 0; j < pytorch.length; j++) {
                double diff = Math.abs(pytorch[j] - warpforge[j]);
                maxDiff = Math.max(maxDiff, diff);
                sumDiff += diff;
                if (diff > tolerance) {
                    mismatches++;
                }
            }

            double meanDiff = pytorch.length > 0 ? sumDiff / pytorch.length : 0.0;

            if (mismatches > 0) {
                allMatch = false;
                differences.add(new ParityResult.OutputDifference(
                        i, maxDiff, meanDiff, mismatches, pytorch.length));
            }
        }

        if (allMatch) {
            return ParityResult.matching(
                    pytorchOutputs, warpforgeOutputs, pytorchTime, warpforgeTime, tolerance);
        } else {
            return ParityResult.different(
                    pytorchOutputs, warpforgeOutputs, differences, pytorchTime, warpforgeTime, tolerance);
        }
    }

    private List<double[]> parseOutputs(Path outputDir) throws IOException {
        // TODO: Parse actual output files from snakegrinder
        // For now, return empty list as placeholder
        return List.of();
    }

    private double[] toDoubleArray(float[] floats) {
        double[] doubles = new double[floats.length];
        for (int i = 0; i < floats.length; i++) {
            doubles[i] = floats[i];
        }
        return doubles;
    }

    private double[] extractTensorValues(Tensor tensor) {
        // Extract tensor values to double array
        // This depends on the Tensor API - using a generic approach
        int[] shape = tensor.shape();
        int totalElements = 1;
        for (int dim : shape) {
            totalElements *= dim;
        }

        double[] values = new double[totalElements];
        // TODO: Implement proper tensor value extraction based on warpforge-core Tensor API
        // For now, return zeros as placeholder
        return values;
    }
}
