package io.surfworks.warpforge.license;

/**
 * Provider-agnostic interface for license operations.
 *
 * <p>Implementations handle the specifics of different license providers
 * (Keygen, Stripe, etc.) while exposing a common interface to the
 * {@link LicenseManager}.
 *
 * <p>Example providers:
 * <ul>
 *   <li>{@link KeygenProvider} - Keygen.sh with offline cryptographic validation</li>
 *   <li>Custom implementations for other providers</li>
 * </ul>
 */
public interface LicenseProvider {

    /**
     * Activate a license key, binding it to this machine.
     *
     * @param licenseKey the license key from the provider
     * @return activation result with license info on success
     */
    ActivationResult activate(String licenseKey);

    /**
     * Validate an existing license key.
     *
     * @param licenseKey the license key
     * @param instanceId the instance ID from activation (provider-specific)
     * @return activation result with updated license info
     */
    ActivationResult validate(String licenseKey, String instanceId);

    /**
     * Deactivate a license instance.
     *
     * <p>This frees up an activation slot for use on another machine.
     *
     * @param licenseKey the license key
     * @param instanceId the instance ID to deactivate
     * @return true if deactivation succeeded
     */
    boolean deactivate(String licenseKey, String instanceId);

    /**
     * Get the provider name for display/logging.
     *
     * @return human-readable provider name (e.g., "Keygen", "Stripe")
     */
    String getProviderName();

    /**
     * Check if this provider supports offline validation.
     *
     * <p>Providers like Keygen support cryptographic offline validation
     * where the license can be verified without network access using
     * embedded public keys.
     *
     * @return true if offline validation is supported
     */
    default boolean supportsOfflineValidation() {
        return false;
    }

    /**
     * Validate a license offline using cryptographic verification.
     *
     * <p>Only applicable for providers that support offline validation
     * (e.g., Keygen with Ed25519 signed license files).
     *
     * @param licenseKey the license key or signed license data
     * @return activation result, or failure if offline validation not supported
     */
    default ActivationResult validateOffline(String licenseKey) {
        return ActivationResult.failure(
            "Offline validation not supported by " + getProviderName(),
            ActivationResult.ErrorCode.UNKNOWN
        );
    }
}
