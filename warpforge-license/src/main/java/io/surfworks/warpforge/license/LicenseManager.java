package io.surfworks.warpforge.license;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;

/**
 * Main entry point for WarpForge license management.
 *
 * <p>Usage:
 * <pre>{@code
 * LicenseManager license = LicenseManager.getInstance();
 *
 * // Check before a licensed operation
 * LicenseCheckResult result = license.checkLicense();
 * if (!result.allowed()) {
 *     System.err.println(result.message());
 *     System.exit(1);
 * }
 *
 * // Activate a new license
 * ActivationResult activation = license.activate("XXXX-XXXX-XXXX-XXXX");
 * if (activation.success()) {
 *     System.out.println("License activated: " + activation.license().product());
 * }
 * }</pre>
 */
public class LicenseManager {

    private static volatile LicenseManager instance;
    private static volatile LicenseProvider configuredProvider;

    private final LicenseCache cache;
    private final LicenseProvider provider;
    private final UsageTracker usage;
    private final Path configDir;

    /**
     * Create a LicenseManager with the specified provider.
     *
     * @param provider the license provider to use
     */
    public LicenseManager(LicenseProvider provider) {
        this.configDir = LicenseConfig.getConfigDir();
        this.cache = new LicenseCache(configDir);
        this.provider = provider;
        this.usage = new UsageTracker(configDir);
    }

    /**
     * Configure the default provider for the singleton instance.
     *
     * <p>Must be called before {@link #getInstance()} to take effect.
     * If not called, no provider is configured and license operations
     * will fail until a provider is set.
     *
     * @param provider the license provider to use
     */
    public static void configureProvider(LicenseProvider provider) {
        configuredProvider = provider;
        // Reset instance so next getInstance() uses new provider
        instance = null;
    }

    /**
     * Get the singleton instance.
     *
     * <p>If no provider has been configured via {@link #configureProvider(LicenseProvider)},
     * license operations requiring a provider will fail gracefully.
     */
    public static LicenseManager getInstance() {
        if (instance == null) {
            synchronized (LicenseManager.class) {
                if (instance == null) {
                    LicenseProvider provider = configuredProvider;
                    if (provider == null) {
                        // No provider configured - use a no-op provider
                        provider = new NoOpProvider();
                    }
                    instance = new LicenseManager(provider);
                }
            }
        }
        return instance;
    }

    /**
     * Get the current license provider.
     *
     * @return the configured provider, or NoOpProvider if none configured
     */
    public LicenseProvider getProvider() {
        return provider;
    }

    /**
     * Check if the current operation is allowed.
     *
     * <p>Call this before any licensed operation (e.g., tracing a model).
     *
     * @return result indicating if operation is allowed
     */
    public LicenseCheckResult checkLicense() {
        // If licensing is disabled, always allow
        if (!LicenseConfig.isEnabled()) {
            return LicenseCheckResult.allowed(WarpForgeProduct.DEV);
        }

        // Check for dev license file (developer machines)
        if (hasValidDevLicense()) {
            return LicenseCheckResult.allowed(WarpForgeProduct.DEV);
        }

        // Check for license key in environment (CI)
        String envKey = LicenseConfig.getLicenseKeyFromEnv();
        if (envKey != null && !envKey.isBlank()) {
            if (isValidDevKey(envKey)) {
                return LicenseCheckResult.allowed(WarpForgeProduct.DEV);
            }
            // Try to activate/validate the env key
            ActivationResult result = validateOrActivate(envKey);
            if (result.success()) {
                return LicenseCheckResult.allowed(result.license().product());
            }
        }

        // Check for cached valid license
        LicenseInfo cached = cache.load();
        if (cached != null) {
            if (!cached.isExpired()) {
                // Valid license - revalidate periodically in background
                if (cached.needsRevalidation()) {
                    revalidateInBackground(cached);
                }
                return LicenseCheckResult.allowed(cached.product());
            }
            if (cached.isWithinGracePeriod()) {
                return LicenseCheckResult.allowedWithWarning(
                    cached.product(),
                    "License expired. Please renew at " + LicenseConfig.UPGRADE_URL
                );
            }
            // License fully expired
            cache.clear();
        }

        // Fall back to free tier with usage limits
        int todayUsage = usage.getTodayCount();
        int freeLimit = WarpForgeProduct.FREE.getDailyTraceLimit();

        if (todayUsage < freeLimit) {
            usage.increment();
            return LicenseCheckResult.allowed(
                WarpForgeProduct.FREE,
                String.format("Free tier: %d/%d traces today", todayUsage + 1, freeLimit)
            );
        }

        // Over free limit - deny with upgrade prompt
        return LicenseCheckResult.denied(String.format("""
            Free tier limit reached (%d traces/day).

            To continue, activate a license:
              warpforge --activate YOUR_LICENSE_KEY

            Get a license at: %s
              WarpForge Pro: $29/month - Unlimited traces
            """, freeLimit, LicenseConfig.UPGRADE_URL));
    }

    /**
     * Activate a license key.
     *
     * @param licenseKey the license key from the configured provider
     * @return activation result
     */
    public ActivationResult activate(String licenseKey) {
        ActivationResult result = provider.activate(licenseKey);
        if (result.success()) {
            cache.save(result.license());
        }
        return result;
    }

    /**
     * Deactivate the current license.
     *
     * <p>This frees up an activation slot for use on another machine.
     */
    public void deactivate() {
        LicenseInfo cached = cache.load();
        if (cached != null && cached.instanceId() != null && cached.key() != null) {
            provider.deactivate(cached.key(), cached.instanceId());
        }
        cache.clear();
    }

    /**
     * Get the current license info.
     *
     * @return license info or null if no license
     */
    public LicenseInfo getCurrentLicense() {
        return cache.load();
    }

    /**
     * Get the machine fingerprint.
     */
    public String getMachineFingerprint() {
        return MachineFingerprint.generate();
    }

    /**
     * Get the machine name for display.
     */
    public String getMachineName() {
        return MachineFingerprint.getMachineName();
    }

    /**
     * Generate a dev key from a secret.
     */
    public static String generateDevKey(String secret) {
        String input = "warpforge-dev-" + secret;
        String hash = sha256(input);
        return "DEV-" + hash.substring(0, 16);
    }

    // ========== Private Methods ==========

    private boolean hasValidDevLicense() {
        Path devFile = LicenseConfig.getDevLicenseFile();
        if (!Files.exists(devFile)) {
            return false;
        }

        try {
            String devKey = Files.readString(devFile).trim();
            return isValidDevKey(devKey);
        } catch (IOException e) {
            return false;
        }
    }

    private boolean isValidDevKey(String key) {
        if (key == null || !key.startsWith("DEV-")) {
            return false;
        }

        // Check against dev secret from environment
        String secret = LicenseConfig.getDevSecretFromEnv();
        if (secret != null && !secret.isBlank()) {
            String expectedKey = generateDevKey(secret);
            if (key.equals(expectedKey)) {
                return true;
            }
        }

        // Allow DEV-UNSAFE-SKIP only when no secret is configured (development only)
        if (secret == null && key.equals("DEV-UNSAFE-SKIP")) {
            return true;
        }

        return false;
    }

    private ActivationResult validateOrActivate(String licenseKey) {
        LicenseInfo cached = cache.load();
        if (cached != null && licenseKey.equals(cached.key()) && cached.instanceId() != null) {
            // Already activated - validate
            return provider.validate(licenseKey, cached.instanceId());
        }
        // New key - activate
        return activate(licenseKey);
    }

    private void revalidateInBackground(LicenseInfo license) {
        // Simple background revalidation - update lastValidated on success
        Thread.ofVirtual().start(() -> {
            if (license.key() != null && license.instanceId() != null) {
                ActivationResult result = provider.validate(license.key(), license.instanceId());
                if (result.success()) {
                    cache.save(result.license());
                }
            }
        });
    }

    private static String sha256(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (Exception e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }

    /**
     * Reset singleton and configured provider (for testing).
     */
    static void resetInstance() {
        instance = null;
        configuredProvider = null;
    }
}
