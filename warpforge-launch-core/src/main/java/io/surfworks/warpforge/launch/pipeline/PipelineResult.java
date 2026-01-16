package io.surfworks.warpforge.launch.pipeline;

import io.surfworks.warpforge.launch.job.TensorResult;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * Result of executing the WarpForge pipeline.
 *
 * @param success      true if pipeline completed successfully
 * @param outputs      output tensors from execution
 * @param mlirPath     path to generated StableHLO MLIR file
 * @param totalTime    total pipeline execution time
 * @param stageTimes   execution time for each stage
 * @param errorMessage error message if failed (null if success)
 * @param metadata     additional metadata from execution
 */
public record PipelineResult(
        boolean success,
        List<TensorResult> outputs,
        Path mlirPath,
        Duration totalTime,
        Map<String, Duration> stageTimes,
        String errorMessage,
        Map<String, String> metadata
) {

    /**
     * Creates a successful result.
     */
    public static PipelineResult success(
            List<TensorResult> outputs,
            Path mlirPath,
            Duration totalTime,
            Map<String, Duration> stageTimes) {
        return new PipelineResult(true, outputs, mlirPath, totalTime, stageTimes, null, Map.of());
    }

    /**
     * Creates a failure result.
     */
    public static PipelineResult failure(String errorMessage, Duration totalTime) {
        return new PipelineResult(false, List.of(), null, totalTime, Map.of(), errorMessage, Map.of());
    }

    /**
     * Creates a failure result with stage times.
     */
    public static PipelineResult failure(
            String errorMessage,
            Duration totalTime,
            Map<String, Duration> stageTimes) {
        return new PipelineResult(false, List.of(), null, totalTime, stageTimes, errorMessage, Map.of());
    }
}
