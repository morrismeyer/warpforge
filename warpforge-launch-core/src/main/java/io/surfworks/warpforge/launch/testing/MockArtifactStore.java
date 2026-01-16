package io.surfworks.warpforge.launch.testing;

import io.surfworks.warpforge.launch.artifact.ArtifactException;
import io.surfworks.warpforge.launch.artifact.ArtifactStore;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-memory artifact store for unit testing.
 *
 * <p>Stores artifacts in memory and tracks all operations for assertions.
 * No actual filesystem I/O is performed (unless copying from local files).
 */
public final class MockArtifactStore implements ArtifactStore {

    private static final String SCHEME = "mock";

    private final Map<String, byte[]> artifacts = new ConcurrentHashMap<>();
    private final List<StoreOperation> storeOperations = new ArrayList<>();
    private final List<RetrieveOperation> retrieveOperations = new ArrayList<>();
    private final List<String> deleteOperations = new ArrayList<>();

    private ArtifactException storeException;
    private ArtifactException retrieveException;

    @Override
    public String name() {
        return "mock";
    }

    @Override
    public URI store(Path localFile, String artifactId) throws ArtifactException {
        if (storeException != null) {
            throw storeException;
        }

        try {
            byte[] content = Files.readAllBytes(localFile);
            artifacts.put(artifactId, content);
            storeOperations.add(new StoreOperation(localFile, artifactId, content.length));

            return URI.create(SCHEME + "://artifacts/" + artifactId);

        } catch (IOException e) {
            throw new ArtifactException("Failed to read local file: " + localFile, e);
        }
    }

    @Override
    public Path retrieve(URI artifactUri, Path localDir) throws ArtifactException {
        if (retrieveException != null) {
            throw retrieveException;
        }

        String artifactId = extractArtifactId(artifactUri);
        byte[] content = artifacts.get(artifactId);

        if (content == null) {
            throw new ArtifactException("Artifact not found: " + artifactUri);
        }

        try {
            Files.createDirectories(localDir);
            Path targetPath = localDir.resolve(artifactId);
            Files.write(targetPath, content);

            retrieveOperations.add(new RetrieveOperation(artifactUri, localDir, artifactId));
            return targetPath;

        } catch (IOException e) {
            throw new ArtifactException("Failed to write to local dir: " + localDir, e);
        }
    }

    @Override
    public void delete(URI artifactUri) throws ArtifactException {
        String artifactId = extractArtifactId(artifactUri);
        artifacts.remove(artifactId);
        deleteOperations.add(artifactId);
    }

    @Override
    public boolean exists(URI artifactUri) {
        String artifactId = extractArtifactId(artifactUri);
        return artifacts.containsKey(artifactId);
    }

    @Override
    public URI baseUri() {
        return URI.create(SCHEME + "://artifacts/");
    }

    // ===== Test configuration methods =====

    /**
     * Pre-populates an artifact for testing retrieval.
     */
    public MockArtifactStore addArtifact(String artifactId, byte[] content) {
        artifacts.put(artifactId, content);
        return this;
    }

    /**
     * Pre-populates an artifact with string content.
     */
    public MockArtifactStore addArtifact(String artifactId, String content) {
        return addArtifact(artifactId, content.getBytes());
    }

    /**
     * Configures store() to throw an exception.
     */
    public MockArtifactStore setStoreException(ArtifactException exception) {
        this.storeException = exception;
        return this;
    }

    /**
     * Configures retrieve() to throw an exception.
     */
    public MockArtifactStore setRetrieveException(ArtifactException exception) {
        this.retrieveException = exception;
        return this;
    }

    // ===== Test assertion methods =====

    /**
     * Returns all stored artifacts.
     */
    public Map<String, byte[]> getArtifacts() {
        return Map.copyOf(artifacts);
    }

    /**
     * Returns the content of a specific artifact.
     */
    public byte[] getArtifact(String artifactId) {
        return artifacts.get(artifactId);
    }

    /**
     * Returns all store operations.
     */
    public List<StoreOperation> getStoreOperations() {
        return List.copyOf(storeOperations);
    }

    /**
     * Returns all retrieve operations.
     */
    public List<RetrieveOperation> getRetrieveOperations() {
        return List.copyOf(retrieveOperations);
    }

    /**
     * Returns all deleted artifact IDs.
     */
    public List<String> getDeleteOperations() {
        return List.copyOf(deleteOperations);
    }

    /**
     * Returns the total number of store operations.
     */
    public int getStoreCount() {
        return storeOperations.size();
    }

    /**
     * Returns the total number of retrieve operations.
     */
    public int getRetrieveCount() {
        return retrieveOperations.size();
    }

    /**
     * Clears all stored artifacts and operation records.
     */
    public void clear() {
        artifacts.clear();
        storeOperations.clear();
        retrieveOperations.clear();
        deleteOperations.clear();
    }

    private String extractArtifactId(URI uri) {
        String path = uri.getPath();
        if (path.startsWith("/")) {
            path = path.substring(1);
        }
        return path;
    }

    /**
     * Record of a store operation.
     */
    public record StoreOperation(Path localFile, String artifactId, long size) {}

    /**
     * Record of a retrieve operation.
     */
    public record RetrieveOperation(URI artifactUri, Path localDir, String artifactId) {}
}
