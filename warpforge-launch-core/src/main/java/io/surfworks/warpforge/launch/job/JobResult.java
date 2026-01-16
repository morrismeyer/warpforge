package io.surfworks.warpforge.launch.job;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Result of a completed job.
 *
 * @param jobId          Scheduler-assigned job ID
 * @param correlationId  Original correlation ID from submission
 * @param success        Whether the job completed successfully
 * @param outputs        Output tensors (or file references)
 * @param mlirOutput     Path to generated StableHLO MLIR (for FULL_PIPELINE jobs)
 * @param executionTime  Total execution time
 * @param errorMessage   Error message (if failed)
 * @param stackTrace     Stack trace (if failed)
 * @param metrics        Performance metrics and timing breakdown
 */
public record JobResult(
        String jobId,
        String correlationId,
        boolean success,
        List<TensorResult> outputs,
        Path mlirOutput,
        Duration executionTime,
        String errorMessage,
        String stackTrace,
        Map<String, Object> metrics
) {

    public JobResult {
        Objects.requireNonNull(jobId, "jobId cannot be null");
        Objects.requireNonNull(executionTime, "executionTime cannot be null");

        outputs = outputs == null ? List.of() : List.copyOf(outputs);
        metrics = metrics == null ? Map.of() : Map.copyOf(metrics);
    }

    /**
     * Creates a successful result with outputs.
     */
    public static JobResult success(
            String jobId,
            String correlationId,
            List<TensorResult> outputs,
            Duration executionTime
    ) {
        return new JobResult(
                jobId, correlationId, true, outputs, null,
                executionTime, null, null, Map.of()
        );
    }

    /**
     * Creates a successful result with outputs and MLIR file.
     */
    public static JobResult success(
            String jobId,
            String correlationId,
            List<TensorResult> outputs,
            Path mlirOutput,
            Duration executionTime
    ) {
        return new JobResult(
                jobId, correlationId, true, outputs, mlirOutput,
                executionTime, null, null, Map.of()
        );
    }

    /**
     * Creates a failed result.
     */
    public static JobResult failure(
            String jobId,
            String correlationId,
            String errorMessage,
            Duration executionTime
    ) {
        return new JobResult(
                jobId, correlationId, false, List.of(), null,
                executionTime, errorMessage, null, Map.of()
        );
    }

    /**
     * Creates a failed result with stack trace.
     */
    public static JobResult failure(
            String jobId,
            String correlationId,
            String errorMessage,
            String stackTrace,
            Duration executionTime
    ) {
        return new JobResult(
                jobId, correlationId, false, List.of(), null,
                executionTime, errorMessage, stackTrace, Map.of()
        );
    }

    /**
     * Returns a new result with additional metrics.
     */
    public JobResult withMetric(String key, Object value) {
        var newMetrics = new java.util.HashMap<>(metrics);
        newMetrics.put(key, value);
        return new JobResult(
                jobId, correlationId, success, outputs, mlirOutput,
                executionTime, errorMessage, stackTrace, newMetrics
        );
    }

    /**
     * Returns a new result with all specified metrics.
     */
    public JobResult withMetrics(Map<String, Object> additionalMetrics) {
        var newMetrics = new java.util.HashMap<>(metrics);
        newMetrics.putAll(additionalMetrics);
        return new JobResult(
                jobId, correlationId, success, outputs, mlirOutput,
                executionTime, errorMessage, stackTrace, newMetrics
        );
    }

    /**
     * Returns true if the job failed.
     */
    public boolean isFailed() {
        return !success;
    }
}
