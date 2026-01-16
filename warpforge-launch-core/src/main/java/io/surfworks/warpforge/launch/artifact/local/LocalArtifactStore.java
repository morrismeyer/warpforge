package io.surfworks.warpforge.launch.artifact.local;

import io.surfworks.warpforge.launch.artifact.ArtifactException;
import io.surfworks.warpforge.launch.artifact.ArtifactStore;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Artifact store using the local filesystem.
 *
 * <p>Stores artifacts in a local directory with URI scheme {@code file://}.
 * Suitable for local scheduler execution or single-node testing.
 */
public final class LocalArtifactStore implements ArtifactStore {

    private static final String SCHEME = "file";

    private final Path storageDir;

    /**
     * Creates a local artifact store with a custom storage directory.
     *
     * @param storageDir directory for storing artifacts
     */
    public LocalArtifactStore(Path storageDir) {
        this.storageDir = storageDir;
    }

    /**
     * Creates a local artifact store with default storage in system temp.
     */
    public LocalArtifactStore() {
        this(Path.of(System.getProperty("java.io.tmpdir"), "warpforge-artifacts"));
    }

    @Override
    public String name() {
        return "local";
    }

    @Override
    public URI store(Path localFile, String artifactId) throws ArtifactException {
        try {
            Files.createDirectories(storageDir);
            Path targetPath = storageDir.resolve(artifactId);
            Files.copy(localFile, targetPath, StandardCopyOption.REPLACE_EXISTING);
            return targetPath.toUri();

        } catch (IOException e) {
            throw new ArtifactException("Failed to store artifact: " + artifactId, e);
        }
    }

    @Override
    public Path retrieve(URI artifactUri, Path localDir) throws ArtifactException {
        validateScheme(artifactUri);

        try {
            Path sourcePath = Path.of(artifactUri);
            String fileName = sourcePath.getFileName().toString();
            Path targetPath = localDir.resolve(fileName);

            Files.createDirectories(localDir);
            Files.copy(sourcePath, targetPath, StandardCopyOption.REPLACE_EXISTING);
            return targetPath;

        } catch (IOException e) {
            throw new ArtifactException("Failed to retrieve artifact: " + artifactUri, e);
        }
    }

    @Override
    public void delete(URI artifactUri) throws ArtifactException {
        validateScheme(artifactUri);

        try {
            Path path = Path.of(artifactUri);
            Files.deleteIfExists(path);

        } catch (IOException e) {
            throw new ArtifactException("Failed to delete artifact: " + artifactUri, e);
        }
    }

    @Override
    public boolean exists(URI artifactUri) {
        if (!SCHEME.equals(artifactUri.getScheme())) {
            return false;
        }
        return Files.exists(Path.of(artifactUri));
    }

    @Override
    public URI baseUri() {
        return storageDir.toUri();
    }

    private void validateScheme(URI uri) throws ArtifactException {
        if (!SCHEME.equals(uri.getScheme())) {
            throw new ArtifactException("Invalid URI scheme: " + uri.getScheme() +
                    " (expected: " + SCHEME + ")");
        }
    }
}
