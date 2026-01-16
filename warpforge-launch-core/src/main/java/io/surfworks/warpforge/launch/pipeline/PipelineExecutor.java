package io.surfworks.warpforge.launch.pipeline;

import io.surfworks.warpforge.launch.config.LaunchConfig;
import io.surfworks.warpforge.launch.config.LaunchConfigLoader;
import io.surfworks.warpforge.launch.job.InputSpec;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Executes the full WarpForge pipeline.
 *
 * <p>Pipeline stages:
 * <ol>
 *   <li><b>snakegrinder</b>: PyTorch model → StableHLO MLIR</li>
 *   <li><b>snakeburger</b>: Parse MLIR → Babylon Code Reflection IR</li>
 *   <li><b>warpforge</b>: Compile IR → GPU kernel execution</li>
 * </ol>
 *
 * <p>For distributed execution, use a {@code Scheduler} to submit
 * {@code JobSubmission} objects instead of invoking this class directly.
 */
public final class PipelineExecutor {

    private static final String STAGE_SNAKEGRINDER = "snakegrinder";
    private static final String STAGE_SNAKEBURGER = "snakeburger";
    private static final String STAGE_WARPFORGE = "warpforge";

    private final LaunchConfig config;
    private final Path workDir;

    /**
     * Creates a pipeline executor with default configuration.
     */
    public PipelineExecutor() {
        this(LaunchConfigLoader.load());
    }

    /**
     * Creates a pipeline executor with custom configuration.
     *
     * @param config launch configuration
     */
    public PipelineExecutor(LaunchConfig config) {
        this.config = config;
        this.workDir = Path.of(System.getProperty("java.io.tmpdir"), "warpforge-pipeline");
    }

    /**
     * Creates a pipeline executor with custom configuration and work directory.
     *
     * @param config  launch configuration
     * @param workDir directory for intermediate files
     */
    public PipelineExecutor(LaunchConfig config, Path workDir) {
        this.config = config;
        this.workDir = workDir;
    }

    /**
     * Executes the full pipeline for a job definition.
     *
     * @param definition the job to execute
     * @return pipeline execution result
     * @throws PipelineException if execution fails
     */
    public PipelineResult execute(JobDefinition definition) throws PipelineException {
        Instant startTime = Instant.now();
        Map<String, Duration> stageTimes = new HashMap<>();

        try {
            Files.createDirectories(workDir);

            // Stage 1: SnakeGrinder (PyTorch → StableHLO)
            Instant stageStart = Instant.now();
            Path mlirPath = runSnakegrinder(definition);
            stageTimes.put(STAGE_SNAKEGRINDER, Duration.between(stageStart, Instant.now()));

            // Stage 2: SnakeBurger (Parse MLIR)
            stageStart = Instant.now();
            // Note: In full implementation, this would invoke SnakeBurger parser
            // For now, we just verify the MLIR file was generated
            if (!Files.exists(mlirPath)) {
                throw new PipelineException(STAGE_SNAKEBURGER, "MLIR file not found: " + mlirPath);
            }
            stageTimes.put(STAGE_SNAKEBURGER, Duration.between(stageStart, Instant.now()));

            // Stage 3: WarpForge (Compile and execute)
            stageStart = Instant.now();
            List<TensorResult> outputs = runWarpforge(mlirPath, definition);
            stageTimes.put(STAGE_WARPFORGE, Duration.between(stageStart, Instant.now()));

            Duration totalTime = Duration.between(startTime, Instant.now());
            return PipelineResult.success(outputs, mlirPath, totalTime, stageTimes);

        } catch (IOException e) {
            Duration totalTime = Duration.between(startTime, Instant.now());
            return PipelineResult.failure("I/O error: " + e.getMessage(), totalTime, stageTimes);
        }
    }

    /**
     * Executes only the snakegrinder stage (PyTorch → StableHLO).
     *
     * @param definition the job definition
     * @return path to the generated MLIR file
     * @throws PipelineException if execution fails
     */
    public Path executeSnakegrinderOnly(JobDefinition definition) throws PipelineException {
        try {
            Files.createDirectories(workDir);
            return runSnakegrinder(definition);
        } catch (IOException e) {
            throw new PipelineException(STAGE_SNAKEGRINDER, "I/O error: " + e.getMessage(), e);
        }
    }

    private Path runSnakegrinder(JobDefinition definition) throws PipelineException {
        Path snakegrinderBin = config.snakegrinderPath();
        if (snakegrinderBin == null) {
            throw new PipelineException(STAGE_SNAKEGRINDER,
                    "snakegrinder binary not found. Configure 'snakegrinderPath' in launch.json or install snakegrinder in PATH.");
        }

        if (!Files.isExecutable(snakegrinderBin)) {
            throw new PipelineException(STAGE_SNAKEGRINDER,
                    "snakegrinder binary is not executable: " + snakegrinderBin);
        }

        String jobId = definition.name().replaceAll("[^a-zA-Z0-9]", "-");
        Path outputDir = workDir.resolve(jobId);

        try {
            Files.createDirectories(outputDir);

            List<String> command = new ArrayList<>();
            command.add(snakegrinderBin.toString());
            command.add("--trace-with-values");
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

            // Capture output
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }

            long timeoutSeconds = definition.timeout() != null ?
                    definition.timeout().toSeconds() : 300; // Default 5 minutes

            boolean completed = process.waitFor(timeoutSeconds, TimeUnit.SECONDS);
            if (!completed) {
                process.destroyForcibly();
                throw new PipelineException(STAGE_SNAKEGRINDER,
                        "Execution timed out after " + timeoutSeconds + " seconds");
            }

            int exitCode = process.exitValue();
            if (exitCode != 0) {
                throw new PipelineException(STAGE_SNAKEGRINDER,
                        "Exit code " + exitCode + ": " + output);
            }

            // Find the generated MLIR file
            Path mlirPath = outputDir.resolve(definition.modelClass() + ".mlir");
            if (!Files.exists(mlirPath)) {
                // Try alternative naming
                mlirPath = outputDir.resolve("model.mlir");
            }

            return mlirPath;

        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new PipelineException(STAGE_SNAKEGRINDER, "Execution failed: " + e.getMessage(), e);
        }
    }

    private List<TensorResult> runWarpforge(Path mlirPath, JobDefinition definition) throws PipelineException {
        // TODO: Implement full warpforge-core integration
        // This would:
        // 1. Use StableHloParser to parse the MLIR
        // 2. Use GraphCompiler to compile to GPU kernels
        // 3. Execute on the selected Backend
        // 4. Return TensorResult outputs

        // For now, return empty list as placeholder
        return List.of();
    }
}
