package io.surfworks.warpforge.license;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link NoOpProvider}.
 */
class NoOpProviderTest {

    private NoOpProvider provider;

    @BeforeEach
    void setUp() {
        provider = new NoOpProvider();
    }

    @Test
    @DisplayName("activate returns failure with informative message")
    void activate_returnsFailure() {
        var result = provider.activate("any-key");

        assertFalse(result.success());
        assertNotNull(result.error());
        assertTrue(result.error().contains("No license provider configured"));
        assertEquals(ActivationResult.ErrorCode.UNKNOWN, result.errorCode());
    }

    @Test
    @DisplayName("validate returns failure")
    void validate_returnsFailure() {
        var result = provider.validate("key", "instance");

        assertFalse(result.success());
        assertNotNull(result.error());
        assertTrue(result.error().contains("No license provider configured"));
        assertEquals(ActivationResult.ErrorCode.UNKNOWN, result.errorCode());
    }

    @Test
    @DisplayName("deactivate returns true to not block flow")
    void deactivate_returnsTrue() {
        assertTrue(provider.deactivate("key", "instance"));
    }

    @Test
    @DisplayName("getProviderName returns descriptive name")
    void getProviderName_returnsDescriptiveName() {
        String name = provider.getProviderName();

        assertNotNull(name);
        assertTrue(name.contains("None") || name.contains("not configured"));
    }

    @Test
    @DisplayName("supportsOfflineValidation returns false (default)")
    void supportsOfflineValidation_returnsFalse() {
        assertFalse(provider.supportsOfflineValidation());
    }

    @Test
    @DisplayName("validateOffline returns failure (default)")
    void validateOffline_returnsFailure() {
        var result = provider.validateOffline("license-data");

        assertFalse(result.success());
        assertTrue(result.error().contains("Offline validation not supported"));
    }

    @Test
    @DisplayName("Implements LicenseProvider interface")
    void implementsInterface() {
        assertTrue(provider instanceof LicenseProvider);
    }
}
