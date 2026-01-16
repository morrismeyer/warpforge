package io.surfworks.warpforge.launch.job;

/**
 * Types of jobs that can be submitted to a scheduler.
 */
public enum JobType {
    /**
     * Full pipeline: SnakeGrinder traces PyTorch model -> StableHLO MLIR,
     * then SnakeBurger parses -> Babylon IR, then warpforge-core executes.
     */
    FULL_PIPELINE,

    /**
     * Skip tracing: Start from pre-traced StableHLO MLIR file,
     * parse with SnakeBurger and execute with warpforge-core.
     */
    STABLEHLO_ONLY,

    /**
     * Native PyTorch execution (for parity comparison).
     * Runs the model directly with PyTorch, no WarpForge processing.
     */
    PYTORCH_NATIVE
}
