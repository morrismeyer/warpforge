package io.surfworks.warpforge.license;

/**
 * Result of a license check operation.
 *
 * @param allowed whether the operation is allowed
 * @param product the product tier (FREE if no license)
 * @param message informational message (usage count, etc.)
 * @param warning optional warning (e.g., "License expires soon")
 * @param upgradeUrl URL for upgrading (if denied)
 */
public record LicenseCheckResult(
    boolean allowed,
    WarpForgeProduct product,
    String message,
    String warning,
    String upgradeUrl
) {

    private static final String DEFAULT_UPGRADE_URL = "https://surfworks.energy/pricing";

    /**
     * Operation allowed with no message.
     */
    public static LicenseCheckResult allowed(WarpForgeProduct product) {
        return new LicenseCheckResult(true, product, null, null, null);
    }

    /**
     * Operation allowed with informational message.
     */
    public static LicenseCheckResult allowed(WarpForgeProduct product, String message) {
        return new LicenseCheckResult(true, product, message, null, null);
    }

    /**
     * Operation allowed but with a warning.
     */
    public static LicenseCheckResult allowedWithWarning(WarpForgeProduct product, String warning) {
        return new LicenseCheckResult(true, product, null, warning, null);
    }

    /**
     * Operation denied.
     */
    public static LicenseCheckResult denied(String message) {
        return new LicenseCheckResult(false, WarpForgeProduct.FREE, message, null, DEFAULT_UPGRADE_URL);
    }

    /**
     * Operation denied with custom upgrade URL.
     */
    public static LicenseCheckResult denied(String message, String upgradeUrl) {
        return new LicenseCheckResult(false, WarpForgeProduct.FREE, message, null, upgradeUrl);
    }
}
