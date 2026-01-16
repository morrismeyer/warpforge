package io.surfworks.warpforge.launch.artifact;

import java.net.URI;
import java.nio.file.Path;

/**
 * Interface for storing and retrieving job artifacts.
 *
 * <p>Artifacts include model files, input data, and job outputs that need
 * to be transferred between the orchestrator and compute nodes.
 */
public interface ArtifactStore extends AutoCloseable {

    /**
     * Returns the name of this artifact store.
     */
    String name();

    /**
     * Stores a local file as an artifact.
     *
     * @param localFile  the local file to store
     * @param artifactId unique identifier for this artifact
     * @return URI that can be used to retrieve the artifact
     * @throws ArtifactException if storage fails
     */
    URI store(Path localFile, String artifactId) throws ArtifactException;

    /**
     * Retrieves an artifact to a local directory.
     *
     * @param artifactUri the URI returned from store()
     * @param localDir    directory to store the retrieved file
     * @return path to the retrieved file
     * @throws ArtifactException if retrieval fails
     */
    Path retrieve(URI artifactUri, Path localDir) throws ArtifactException;

    /**
     * Deletes an artifact.
     *
     * @param artifactUri the URI of the artifact to delete
     * @throws ArtifactException if deletion fails
     */
    void delete(URI artifactUri) throws ArtifactException;

    /**
     * Checks if an artifact exists.
     *
     * @param artifactUri the URI to check
     * @return true if the artifact exists
     */
    boolean exists(URI artifactUri);

    /**
     * Returns the base URI for this store.
     *
     * <p>Used for generating artifact URIs and debugging.
     */
    URI baseUri();

    @Override
    default void close() {
        // Default no-op implementation
    }
}
