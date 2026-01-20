package io.surfworks.warpforge.codegen.api;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelMetadataTest {

    @Test
    void testRecordCreation() {
        long timestamp = 1704067200000L; // 2024-01-01 00:00:00 UTC
        var metadata = new ModelMetadata("test_model", "abc123", timestamp, "1.0.0");

        assertEquals("test_model", metadata.name());
        assertEquals("abc123", metadata.sourceHash());
        assertEquals(timestamp, metadata.generatedAt());
        assertEquals("1.0.0", metadata.generatorVersion());
    }

    @Test
    void testCreateFactoryMethod() {
        long before = System.currentTimeMillis();
        var metadata = ModelMetadata.create("my_model", "sha256hash", "2.0.0");
        long after = System.currentTimeMillis();

        assertEquals("my_model", metadata.name());
        assertEquals("sha256hash", metadata.sourceHash());
        assertEquals("2.0.0", metadata.generatorVersion());
        assertTrue(metadata.generatedAt() >= before);
        assertTrue(metadata.generatedAt() <= after);
    }

    @Test
    void testEquality() {
        var m1 = new ModelMetadata("model", "hash", 1000L, "1.0");
        var m2 = new ModelMetadata("model", "hash", 1000L, "1.0");

        assertEquals(m1, m2);
        assertEquals(m1.hashCode(), m2.hashCode());
    }

    @Test
    void testToString() {
        var metadata = new ModelMetadata("test", "h", 0L, "v");
        assertNotNull(metadata.toString());
        assertTrue(metadata.toString().contains("test"));
    }
}
