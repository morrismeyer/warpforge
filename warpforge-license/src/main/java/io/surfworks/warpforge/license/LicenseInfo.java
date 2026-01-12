package io.surfworks.warpforge.license;

import java.time.Instant;
import java.util.Map;

/**
 * License information cached locally and returned from Lemon Squeezy API.
 *
 * @param key the license key (masked for display)
 * @param instanceId the activation instance ID from Lemon Squeezy
 * @param product the WarpForge product tier
 * @param validUntil when the license expires (null = perpetual)
 * @param activatedAt when this instance was activated
 * @param lastValidated last successful online validation
 * @param machineFingerprint hardware fingerprint for this machine
 * @param customerEmail customer email (for support)
 * @param metadata additional metadata from Lemon Squeezy
 */
public record LicenseInfo(
    String key,
    String instanceId,
    WarpForgeProduct product,
    Instant validUntil,
    Instant activatedAt,
    Instant lastValidated,
    String machineFingerprint,
    String customerEmail,
    Map<String, Object> metadata
) {

    /**
     * Grace period after expiration before hard lockout (7 days).
     */
    private static final long GRACE_PERIOD_SECONDS = 7 * 24 * 3600;

    /**
     * Revalidation interval - check online every 7 days.
     */
    private static final long REVALIDATION_INTERVAL_SECONDS = 7 * 24 * 3600;

    public boolean isExpired() {
        return validUntil != null && Instant.now().isAfter(validUntil);
    }

    public boolean isWithinGracePeriod() {
        if (validUntil == null) return true; // Perpetual license
        return Instant.now().isBefore(validUntil.plusSeconds(GRACE_PERIOD_SECONDS));
    }

    public boolean needsRevalidation() {
        if (lastValidated == null) return true;
        return lastValidated.plusSeconds(REVALIDATION_INTERVAL_SECONDS).isBefore(Instant.now());
    }

    /**
     * Get masked key for display (e.g., "XXXX...YYYY").
     */
    public String getMaskedKey() {
        if (key == null || key.length() < 8) return "****";
        return key.substring(0, 4) + "..." + key.substring(key.length() - 4);
    }

    /**
     * Create a license info for the free tier.
     */
    public static LicenseInfo freeTier() {
        return new LicenseInfo(
            null, null, WarpForgeProduct.FREE,
            null, null, null, null, null, Map.of()
        );
    }

    /**
     * Create a license info for developer/CI bypass.
     */
    public static LicenseInfo devBypass() {
        return new LicenseInfo(
            "DEV-INTERNAL", "dev",
            WarpForgeProduct.DEV,
            Instant.now().plusSeconds(365 * 24 * 3600),
            Instant.now(), Instant.now(),
            null, null, Map.of("dev", true)
        );
    }
}
