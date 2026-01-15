package io.surfworks.warpforge.license;

import java.nio.file.Path;

/**
 * Configuration for WarpForge licensing.
 *
 * <p>Licensing is DISABLED by default. Set {@code WARPFORGE_LICENSE_ENABLED=true}
 * or use {@link #enable()} to turn on license checking.
 */
public final class LicenseConfig {

    /**
     * Environment variable to enable/disable licensing.
     */
    public static final String ENV_LICENSE_ENABLED = "WARPFORGE_LICENSE_ENABLED";

    /**
     * Environment variable for license key (CI/automation).
     */
    public static final String ENV_LICENSE_KEY = "WARPFORGE_LICENSE_KEY";

    /**
     * Environment variable for dev secret (internal use).
     */
    public static final String ENV_DEV_SECRET = "WARPFORGE_DEV_SECRET";

    /**
     * Environment variable for license provider selection.
     *
     * <p>Valid values: "keygen", "none"
     */
    public static final String ENV_LICENSE_PROVIDER = "WARPFORGE_LICENSE_PROVIDER";

    /**
     * Default config directory.
     */
    public static final Path DEFAULT_CONFIG_DIR = Path.of(
        System.getProperty("user.home"), ".config", "warpforge"
    );

    /**
     * Dev license file name.
     */
    public static final String DEV_LICENSE_FILE = "dev.license";

    /**
     * Upgrade URL for license purchase.
     */
    public static final String UPGRADE_URL = "https://surfworks.energy/pricing";

    /**
     * Support email.
     */
    public static final String SUPPORT_EMAIL = "support@surfworks.energy";

    // Singleton state
    private static volatile boolean enabled = false;
    private static volatile boolean initialized = false;

    private LicenseConfig() {}

    /**
     * Check if licensing is enabled.
     *
     * <p>Licensing is disabled by default. It can be enabled by:
     * <ul>
     *   <li>Setting environment variable WARPFORGE_LICENSE_ENABLED=true</li>
     *   <li>Calling {@link #enable()} programmatically</li>
     * </ul>
     */
    public static boolean isEnabled() {
        if (!initialized) {
            initialize();
        }
        return enabled;
    }

    /**
     * Enable licensing programmatically.
     */
    public static void enable() {
        enabled = true;
        initialized = true;
    }

    /**
     * Disable licensing programmatically.
     */
    public static void disable() {
        enabled = false;
        initialized = true;
    }

    /**
     * Get the license key from environment variable.
     */
    public static String getLicenseKeyFromEnv() {
        return System.getenv(ENV_LICENSE_KEY);
    }

    /**
     * Get the dev secret from environment variable.
     */
    public static String getDevSecretFromEnv() {
        return System.getenv(ENV_DEV_SECRET);
    }

    /**
     * Get the configured license provider name from environment variable.
     *
     * @return provider name ("keygen", etc.) or null if not set
     */
    public static String getProviderFromEnv() {
        return System.getenv(ENV_LICENSE_PROVIDER);
    }

    /**
     * Get the config directory.
     */
    public static Path getConfigDir() {
        String configHome = System.getenv("XDG_CONFIG_HOME");
        if (configHome != null && !configHome.isBlank()) {
            return Path.of(configHome, "warpforge");
        }
        return DEFAULT_CONFIG_DIR;
    }

    /**
     * Get the path to the dev license file.
     */
    public static Path getDevLicenseFile() {
        return getConfigDir().resolve(DEV_LICENSE_FILE);
    }

    private static synchronized void initialize() {
        if (initialized) return;

        String envValue = System.getenv(ENV_LICENSE_ENABLED);
        enabled = "true".equalsIgnoreCase(envValue) || "1".equals(envValue);
        initialized = true;
    }

    /**
     * Reset state (for testing).
     */
    static void reset() {
        initialized = false;
        enabled = false;
    }
}
