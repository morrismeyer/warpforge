package io.surfworks.warpforge.data.golden;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * Interface for storing and retrieving golden outputs.
 *
 * <p>Implementations may store golden outputs in:
 * <ul>
 *   <li>Local files (FileGoldenStore)</li>
 *   <li>Remote storage (S3, GCS, etc.)</li>
 *   <li>Database</li>
 * </ul>
 */
public interface GoldenStore {

    /**
     * Save a golden output.
     *
     * @param output The golden output to save
     * @throws IOException if saving fails
     */
    void save(GoldenOutput output) throws IOException;

    /**
     * Load a golden output by ID.
     *
     * @param id The golden output ID
     * @return The golden output, or empty if not found
     * @throws IOException if loading fails
     */
    Optional<GoldenOutput> load(String id) throws IOException;

    /**
     * Check if a golden output exists.
     *
     * @param id The golden output ID
     * @return true if the golden output exists
     */
    boolean exists(String id);

    /**
     * Delete a golden output.
     *
     * @param id The golden output ID
     * @return true if deleted, false if not found
     * @throws IOException if deletion fails
     */
    boolean delete(String id) throws IOException;

    /**
     * List all golden output IDs.
     *
     * @return List of golden output IDs
     * @throws IOException if listing fails
     */
    List<String> list() throws IOException;

    /**
     * List golden output IDs matching a prefix.
     *
     * @param prefix Prefix to match (e.g., "bert-base/")
     * @return List of matching golden output IDs
     * @throws IOException if listing fails
     */
    List<String> listByPrefix(String prefix) throws IOException;

    /**
     * Create a file-based golden store.
     *
     * @param directory Directory to store golden outputs
     * @return A new FileGoldenStore
     */
    static GoldenStore file(Path directory) {
        return new FileGoldenStore(directory);
    }
}
