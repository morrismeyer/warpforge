package io.surfworks.warpforge.launch.artifact;

/**
 * Checked exception for artifact store operations.
 */
public class ArtifactException extends Exception {

    public ArtifactException(String message) {
        super(message);
    }

    public ArtifactException(String message, Throwable cause) {
        super(message, cause);
    }
}
