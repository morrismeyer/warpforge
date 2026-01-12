package io.surfworks.warpforge.license;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;

/**
 * Caches license information locally for offline use.
 *
 * <p>Stores license in ~/.config/warpforge/license.json
 */
public class LicenseCache {

    private static final String LICENSE_FILE = "license.json";
    private static final Gson GSON = new GsonBuilder()
        .setPrettyPrinting()
        .create();

    private final Path configDir;
    private final Path licenseFile;

    public LicenseCache(Path configDir) {
        this.configDir = configDir;
        this.licenseFile = configDir.resolve(LICENSE_FILE);
    }

    /**
     * Load cached license info.
     *
     * @return license info or null if not cached
     */
    public LicenseInfo load() {
        if (!Files.exists(licenseFile)) {
            return null;
        }

        try {
            String json = Files.readString(licenseFile);
            CachedLicense cached = GSON.fromJson(json, CachedLicense.class);
            if (cached == null) {
                return null;
            }

            return new LicenseInfo(
                cached.key,
                cached.instanceId,
                WarpForgeProduct.valueOf(cached.product),
                cached.validUntil != null ? Instant.parse(cached.validUntil) : null,
                cached.activatedAt != null ? Instant.parse(cached.activatedAt) : null,
                cached.lastValidated != null ? Instant.parse(cached.lastValidated) : null,
                cached.machineFingerprint,
                cached.customerEmail,
                cached.metadata != null ? cached.metadata : Map.of()
            );
        } catch (Exception e) {
            // Corrupted cache - return null
            return null;
        }
    }

    /**
     * Save license info to cache.
     */
    public void save(LicenseInfo license) {
        CachedLicense cached = new CachedLicense();
        cached.key = license.key();
        cached.instanceId = license.instanceId();
        cached.product = license.product().name();
        cached.validUntil = license.validUntil() != null ? license.validUntil().toString() : null;
        cached.activatedAt = license.activatedAt() != null ? license.activatedAt().toString() : null;
        cached.lastValidated = license.lastValidated() != null ? license.lastValidated().toString() : null;
        cached.machineFingerprint = license.machineFingerprint();
        cached.customerEmail = license.customerEmail();
        cached.metadata = license.metadata();

        try {
            Files.createDirectories(configDir);
            Files.writeString(licenseFile, GSON.toJson(cached));
        } catch (IOException e) {
            // Best effort
        }
    }

    /**
     * Clear the cached license.
     */
    public void clear() {
        try {
            Files.deleteIfExists(licenseFile);
        } catch (IOException e) {
            // Best effort
        }
    }

    /**
     * Check if a license is cached.
     */
    public boolean exists() {
        return Files.exists(licenseFile);
    }

    /**
     * Internal structure for JSON serialization.
     */
    private static class CachedLicense {
        String key;
        String instanceId;
        String product;
        String validUntil;
        String activatedAt;
        String lastValidated;
        String machineFingerprint;
        String customerEmail;
        Map<String, Object> metadata;
    }
}
