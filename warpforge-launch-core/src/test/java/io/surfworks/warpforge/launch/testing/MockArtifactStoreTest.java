package io.surfworks.warpforge.launch.testing;

import io.surfworks.warpforge.launch.artifact.ArtifactException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for MockArtifactStore.
 */
class MockArtifactStoreTest {

    private MockArtifactStore store;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() {
        store = new MockArtifactStore();
    }

    @Test
    void storeAndRetrieve() throws IOException, ArtifactException {
        // Create a local file
        Path localFile = tempDir.resolve("test.txt");
        Files.writeString(localFile, "Hello, World!");

        // Store it
        URI artifactUri = store.store(localFile, "test-artifact");

        // Retrieve it
        Path retrieveDir = tempDir.resolve("retrieved");
        Path retrievedFile = store.retrieve(artifactUri, retrieveDir);

        assertTrue(Files.exists(retrievedFile));
        assertEquals("Hello, World!", Files.readString(retrievedFile));
    }

    @Test
    void existsReturnsTrueForStoredArtifact() throws IOException, ArtifactException {
        Path localFile = tempDir.resolve("test.txt");
        Files.writeString(localFile, "content");

        URI artifactUri = store.store(localFile, "my-artifact");

        assertTrue(store.exists(artifactUri));
    }

    @Test
    void existsReturnsFalseForMissingArtifact() {
        URI fakeUri = URI.create("mock://artifacts/nonexistent");
        assertFalse(store.exists(fakeUri));
    }

    @Test
    void deleteRemovesArtifact() throws IOException, ArtifactException {
        Path localFile = tempDir.resolve("test.txt");
        Files.writeString(localFile, "content");

        URI artifactUri = store.store(localFile, "to-delete");
        assertTrue(store.exists(artifactUri));

        store.delete(artifactUri);
        assertFalse(store.exists(artifactUri));
    }

    @Test
    void retrieveNonexistentThrows() {
        URI fakeUri = URI.create("mock://artifacts/nonexistent");
        assertThrows(ArtifactException.class, () ->
                store.retrieve(fakeUri, tempDir));
    }

    @Test
    void addArtifactPrePopulates() throws ArtifactException, IOException {
        store.addArtifact("pre-added", "Pre-added content");

        URI uri = URI.create("mock://artifacts/pre-added");
        assertTrue(store.exists(uri));

        Path retrieved = store.retrieve(uri, tempDir);
        assertEquals("Pre-added content", Files.readString(retrieved));
    }

    @Test
    void storeExceptionConfiguration() throws IOException {
        store.setStoreException(new ArtifactException("Store disabled"));

        Path localFile = tempDir.resolve("test.txt");
        Files.writeString(localFile, "content");

        assertThrows(ArtifactException.class, () ->
                store.store(localFile, "artifact"));
    }

    @Test
    void retrieveExceptionConfiguration() throws IOException, ArtifactException {
        Path localFile = tempDir.resolve("test.txt");
        Files.writeString(localFile, "content");
        URI uri = store.store(localFile, "artifact");

        store.setRetrieveException(new ArtifactException("Retrieve disabled"));

        assertThrows(ArtifactException.class, () ->
                store.retrieve(uri, tempDir.resolve("output")));
    }

    @Test
    void trackStoreOperations() throws IOException, ArtifactException {
        Path file1 = tempDir.resolve("file1.txt");
        Files.writeString(file1, "content1");
        Path file2 = tempDir.resolve("file2.txt");
        Files.writeString(file2, "content2");

        store.store(file1, "artifact1");
        store.store(file2, "artifact2");

        assertEquals(2, store.getStoreCount());
        var ops = store.getStoreOperations();
        assertEquals("artifact1", ops.get(0).artifactId());
        assertEquals("artifact2", ops.get(1).artifactId());
    }

    @Test
    void trackRetrieveOperations() throws IOException, ArtifactException {
        Path localFile = tempDir.resolve("test.txt");
        Files.writeString(localFile, "content");
        URI uri = store.store(localFile, "artifact");

        store.retrieve(uri, tempDir.resolve("out1"));
        store.retrieve(uri, tempDir.resolve("out2"));

        assertEquals(2, store.getRetrieveCount());
    }

    @Test
    void trackDeleteOperations() throws IOException, ArtifactException {
        Path localFile = tempDir.resolve("test.txt");
        Files.writeString(localFile, "content");
        URI uri = store.store(localFile, "artifact");

        store.delete(uri);

        assertEquals(1, store.getDeleteOperations().size());
        assertEquals("artifact", store.getDeleteOperations().get(0));
    }

    @Test
    void clearResetsAllState() throws IOException, ArtifactException {
        Path localFile = tempDir.resolve("test.txt");
        Files.writeString(localFile, "content");
        store.store(localFile, "artifact");

        store.clear();

        assertEquals(0, store.getStoreCount());
        assertEquals(0, store.getRetrieveCount());
        assertTrue(store.getArtifacts().isEmpty());
    }

    @Test
    void getArtifactReturnsContent() throws IOException, ArtifactException {
        Path localFile = tempDir.resolve("test.txt");
        byte[] content = "test content".getBytes();
        Files.write(localFile, content);

        store.store(localFile, "artifact");

        byte[] retrieved = store.getArtifact("artifact");
        assertArrayEquals(content, retrieved);
    }

    @Test
    void baseUri() {
        assertEquals("mock://artifacts/", store.baseUri().toString());
    }
}
