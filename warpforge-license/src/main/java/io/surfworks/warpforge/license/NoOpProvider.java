package io.surfworks.warpforge.license;

/**
 * A no-op license provider used when no provider is configured.
 *
 * <p>All operations fail gracefully with informative error messages,
 * guiding the user to configure a proper license provider.
 */
public class NoOpProvider implements LicenseProvider {

    @Override
    public ActivationResult activate(String licenseKey) {
        return ActivationResult.failure(
            "No license provider configured. Please configure a provider before activating.",
            ActivationResult.ErrorCode.UNKNOWN
        );
    }

    @Override
    public ActivationResult validate(String licenseKey, String instanceId) {
        return ActivationResult.failure(
            "No license provider configured.",
            ActivationResult.ErrorCode.UNKNOWN
        );
    }

    @Override
    public boolean deactivate(String licenseKey, String instanceId) {
        // No-op, but return true to not block deactivation flow
        return true;
    }

    @Override
    public String getProviderName() {
        return "None (not configured)";
    }
}
