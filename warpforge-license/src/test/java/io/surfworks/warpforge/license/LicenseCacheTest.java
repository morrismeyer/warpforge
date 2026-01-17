package io.surfworks.warpforge.license;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link LicenseCache}.
 */
class LicenseCacheTest {

    @TempDir
    Path tempDir;

    private LicenseCache cache;

    @BeforeEach
    void setUp() {
        cache = new LicenseCache(tempDir);
    }

    @Test
    @DisplayName("load returns null for empty cache")
    void load_emptyCache_returnsNull() {
        assertNull(cache.load());
    }

    @Test
    @DisplayName("exists returns false for empty cache")
    void exists_emptyCache_returnsFalse() {
        assertFalse(cache.exists());
    }

    @Test
    @DisplayName("save and load round-trip works")
    void saveAndLoad_roundTrip() {
        var original = new LicenseInfo(
            "ABCD-1234-5678-WXYZ",
            "instance-123",
            WarpForgeProduct.WARPFORGE_PRO,
            Instant.now().plus(30, ChronoUnit.DAYS).truncatedTo(ChronoUnit.SECONDS),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            Instant.now().truncatedTo(ChronoUnit.SECONDS),
            "fingerprint-abc123",
            "test@example.com",
            Map.of("custom", "value")
        );

        cache.save(original);

        assertTrue(cache.exists());

        var loaded = cache.load();
        assertNotNull(loaded);
        assertEquals(original.key(), loaded.key());
        assertEquals(original.instanceId(), loaded.instanceId());
        assertEquals(original.product(), loaded.product());
        assertEquals(original.validUntil(), loaded.validUntil());
        assertEquals(original.activatedAt(), loaded.activatedAt());
        assertEquals(original.lastValidated(), loaded.lastValidated());
        assertEquals(original.machineFingerprint(), loaded.machineFingerprint());
        assertEquals(original.customerEmail(), loaded.customerEmail());
    }

    @Test
    @DisplayName("save with null fields works")
    void save_nullFields_works() {
        var original = new LicenseInfo(
            "key",
            "instance",
            WarpForgeProduct.WARPFORGE_PRO,
            null, // perpetual
            null,
            null,
            null,
            null,
            Map.of()
        );

        cache.save(original);

        var loaded = cache.load();
        assertNotNull(loaded);
        assertEquals("key", loaded.key());
        assertNull(loaded.validUntil());
        assertNotNull(loaded.metadata());
    }

    @Test
    @DisplayName("clear removes the license file")
    void clear_removesFile() {
        cache.save(LicenseInfo.devBypass());
        assertTrue(cache.exists());

        cache.clear();

        assertFalse(cache.exists());
        assertNull(cache.load());
    }

    @Test
    @DisplayName("clear on empty cache is safe")
    void clear_emptyCache_isSafe() {
        assertFalse(cache.exists());

        // Should not throw
        cache.clear();

        assertFalse(cache.exists());
    }

    @Test
    @DisplayName("Corrupted cache file returns null")
    void corruptedFile_returnsNull() throws IOException {
        Path licenseFile = tempDir.resolve("license.json");
        Files.writeString(licenseFile, "not valid json{{{");

        assertNull(cache.load());
    }

    @Test
    @DisplayName("Empty cache file returns null")
    void emptyFile_returnsNull() throws IOException {
        Path licenseFile = tempDir.resolve("license.json");
        Files.writeString(licenseFile, "");

        assertNull(cache.load());
    }

    @Test
    @DisplayName("Null JSON value returns null")
    void nullJsonValue_returnsNull() throws IOException {
        Path licenseFile = tempDir.resolve("license.json");
        Files.writeString(licenseFile, "null");

        assertNull(cache.load());
    }

    @Test
    @DisplayName("Invalid product name in JSON returns null")
    void invalidProductName_returnsNull() throws IOException {
        Path licenseFile = tempDir.resolve("license.json");
        Files.writeString(licenseFile, """
            {
              "key": "test",
              "instanceId": "instance",
              "product": "INVALID_PRODUCT"
            }
            """);

        // Should return null due to enum parsing failure
        assertNull(cache.load());
    }

    @Test
    @DisplayName("License file is created in config directory")
    void licenseFile_isCreated() {
        cache.save(LicenseInfo.devBypass());

        Path licenseFile = tempDir.resolve("license.json");
        assertTrue(Files.exists(licenseFile));
    }

    @Test
    @DisplayName("License file is valid JSON")
    void licenseFile_isValidJson() throws IOException {
        cache.save(LicenseInfo.devBypass());

        Path licenseFile = tempDir.resolve("license.json");
        String content = Files.readString(licenseFile);

        assertTrue(content.contains("\"key\""));
        assertTrue(content.contains("\"product\""));
    }

    @Test
    @DisplayName("Multiple save calls overwrite previous")
    void multipleSaves_overwrite() {
        var license1 = new LicenseInfo(
            "key1", "instance1", WarpForgeProduct.WARPFORGE_PRO,
            null, null, null, null, null, Map.of()
        );
        var license2 = new LicenseInfo(
            "key2", "instance2", WarpForgeProduct.WARPFORGE_TEAM,
            null, null, null, null, null, Map.of()
        );

        cache.save(license1);
        assertEquals("key1", cache.load().key());

        cache.save(license2);
        assertEquals("key2", cache.load().key());
        assertEquals(WarpForgeProduct.WARPFORGE_TEAM, cache.load().product());
    }

    @Test
    @DisplayName("Metadata with null value is handled")
    void metadataWithNullValue_handled() throws IOException {
        Path licenseFile = tempDir.resolve("license.json");
        Files.writeString(licenseFile, """
            {
              "key": "test",
              "instanceId": "instance",
              "product": "WARPFORGE_PRO",
              "metadata": null
            }
            """);

        var loaded = cache.load();
        assertNotNull(loaded);
        assertNotNull(loaded.metadata());
    }
}
