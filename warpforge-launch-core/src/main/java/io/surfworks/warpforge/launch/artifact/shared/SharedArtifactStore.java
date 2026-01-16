package io.surfworks.warpforge.launch.artifact.shared;

import io.surfworks.warpforge.launch.artifact.ArtifactException;
import io.surfworks.warpforge.launch.artifact.ArtifactStore;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;

/**
 * Artifact store using a shared filesystem (NFS, GPFS, etc.).
 *
 * <p>Stores artifacts on a filesystem accessible to all cluster nodes.
 * Used with Ray, Kubernetes, and Slurm schedulers where workers need
 * direct access to model files and outputs.
 *
 * <p>The shared directory should be mounted at the same path on all
 * nodes. This is typically configured via:
 * <ul>
 *   <li>Config file: {@code ~/.config/warpforge/launch.json} with {@code "sharedDir": "/path"}</li>
 *   <li>Auto-detection: Looks for common mount points like {@code /shared/warpforge}</li>
 * </ul>
 */
public final class SharedArtifactStore implements ArtifactStore {

    private static final String SCHEME = "file";

    private final Path sharedDir;

    /**
     * Creates a shared artifact store with the given shared directory.
     *
     * @param sharedDir path to the shared filesystem directory
     * @throws IllegalArgumentException if sharedDir does not exist
     */
    public SharedArtifactStore(Path sharedDir) {
        this.sharedDir = sharedDir;
        if (!Files.isDirectory(sharedDir)) {
            throw new IllegalArgumentException("Shared directory does not exist: " + sharedDir);
        }
    }

    /**
     * Auto-detects the shared directory from common locations.
     *
     * @return a SharedArtifactStore if a valid location is found
     * @throws IllegalStateException if no shared directory is found
     */
    public static SharedArtifactStore autoDetect() {
        // Check common shared filesystem mount points
        Path[] candidates = {
                Path.of("/shared/warpforge/artifacts"),
                Path.of("/mnt/shared/warpforge/artifacts"),
                Path.of("/nfs/warpforge/artifacts"),
                Path.of(System.getProperty("user.home"), "shared", "warpforge", "artifacts")
        };

        for (Path candidate : candidates) {
            if (Files.isDirectory(candidate)) {
                return new SharedArtifactStore(candidate);
            }
            // Try parent directory
            Path parent = candidate.getParent();
            if (parent != null && Files.isDirectory(parent)) {
                try {
                    Files.createDirectories(candidate);
                    return new SharedArtifactStore(candidate);
                } catch (IOException e) {
                    // Continue to next candidate
                }
            }
        }

        throw new IllegalStateException(
                "No shared directory found. Configure 'sharedDir' in ~/.config/warpforge/launch.json");
    }

    @Override
    public String name() {
        return "shared";
    }

    @Override
    public URI store(Path localFile, String artifactId) throws ArtifactException {
        try {
            // Create subdirectory based on date for organization
            String datePath = java.time.LocalDate.now().toString();
            Path targetDir = sharedDir.resolve(datePath);
            Files.createDirectories(targetDir);

            Path targetPath = targetDir.resolve(artifactId);
            Files.copy(localFile, targetPath, StandardCopyOption.REPLACE_EXISTING);
            return targetPath.toUri();

        } catch (IOException e) {
            throw new ArtifactException("Failed to store artifact: " + artifactId, e);
        }
    }

    @Override
    public Path retrieve(URI artifactUri, Path localDir) throws ArtifactException {
        validateScheme(artifactUri);

        // For shared storage, the file is already accessible on all nodes
        // Just return the path directly
        Path sharedPath = Path.of(artifactUri);

        if (!Files.exists(sharedPath)) {
            throw new ArtifactException("Artifact not found: " + artifactUri);
        }

        // If localDir is within the shared directory, just return the shared path
        if (sharedPath.startsWith(sharedDir)) {
            return sharedPath;
        }

        // Otherwise, copy to local dir (for cases where local cache is needed)
        try {
            String fileName = sharedPath.getFileName().toString();
            Path targetPath = localDir.resolve(fileName);
            Files.createDirectories(localDir);
            Files.copy(sharedPath, targetPath, StandardCopyOption.REPLACE_EXISTING);
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
        return sharedDir.toUri();
    }

    /**
     * Returns the shared directory path.
     */
    public Path sharedDir() {
        return sharedDir;
    }

    private void validateScheme(URI uri) throws ArtifactException {
        if (!SCHEME.equals(uri.getScheme())) {
            throw new ArtifactException("Invalid URI scheme: " + uri.getScheme() +
                    " (expected: " + SCHEME + ")");
        }
    }
}
