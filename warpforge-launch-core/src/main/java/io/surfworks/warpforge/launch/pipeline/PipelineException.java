package io.surfworks.warpforge.launch.pipeline;

/**
 * Checked exception for pipeline execution errors.
 */
public class PipelineException extends Exception {

    private final String stage;

    public PipelineException(String message) {
        super(message);
        this.stage = null;
    }

    public PipelineException(String message, Throwable cause) {
        super(message, cause);
        this.stage = null;
    }

    public PipelineException(String stage, String message) {
        super(stage + ": " + message);
        this.stage = stage;
    }

    public PipelineException(String stage, String message, Throwable cause) {
        super(stage + ": " + message, cause);
        this.stage = stage;
    }

    /**
     * Returns the pipeline stage where the error occurred (may be null).
     */
    public String stage() {
        return stage;
    }
}
