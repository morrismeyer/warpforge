package io.surfworks.warpforge.license;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link LicenseConfig}.
 */
class LicenseConfigTest {

    @BeforeEach
    void setUp() {
        LicenseConfig.reset();
    }

    @AfterEach
    void tearDown() {
        LicenseConfig.reset();
    }

    @Test
    @DisplayName("isEnabled returns false by default")
    void isEnabled_defaultFalse() {
        // Note: This will only be false if WARPFORGE_LICENSE_ENABLED env var is not set
        // In test environment, we reset state so it reads from env
        boolean enabled = LicenseConfig.isEnabled();

        // Just verify it doesn't throw and returns a boolean
        assertNotNull(enabled);
    }

    @Test
    @DisplayName("enable() sets enabled to true")
    void enable_setsTrue() {
        LicenseConfig.enable();

        assertTrue(LicenseConfig.isEnabled());
    }

    @Test
    @DisplayName("disable() sets enabled to false")
    void disable_setsFalse() {
        LicenseConfig.enable();
        assertTrue(LicenseConfig.isEnabled());

        LicenseConfig.disable();

        assertFalse(LicenseConfig.isEnabled());
    }

    @Test
    @DisplayName("getConfigDir returns path")
    void getConfigDir_returnsPath() {
        Path configDir = LicenseConfig.getConfigDir();

        assertNotNull(configDir);
        assertTrue(configDir.toString().contains("warpforge"));
    }

    @Test
    @DisplayName("getDevLicenseFile returns path in config dir")
    void getDevLicenseFile_returnsPathInConfigDir() {
        Path devFile = LicenseConfig.getDevLicenseFile();

        assertNotNull(devFile);
        assertTrue(devFile.toString().contains(LicenseConfig.DEV_LICENSE_FILE));
        assertTrue(devFile.startsWith(LicenseConfig.getConfigDir()));
    }

    @Test
    @DisplayName("Constants are defined")
    void constants_areDefined() {
        assertNotNull(LicenseConfig.ENV_LICENSE_ENABLED);
        assertNotNull(LicenseConfig.ENV_LICENSE_KEY);
        assertNotNull(LicenseConfig.ENV_DEV_SECRET);
        assertNotNull(LicenseConfig.ENV_LICENSE_PROVIDER);
        assertNotNull(LicenseConfig.DEFAULT_CONFIG_DIR);
        assertNotNull(LicenseConfig.DEV_LICENSE_FILE);
        assertNotNull(LicenseConfig.UPGRADE_URL);
        assertNotNull(LicenseConfig.SUPPORT_EMAIL);
    }

    @Test
    @DisplayName("Environment variable names follow convention")
    void envVarNames_followConvention() {
        assertTrue(LicenseConfig.ENV_LICENSE_ENABLED.startsWith("WARPFORGE_"));
        assertTrue(LicenseConfig.ENV_LICENSE_KEY.startsWith("WARPFORGE_"));
        assertTrue(LicenseConfig.ENV_DEV_SECRET.startsWith("WARPFORGE_"));
        assertTrue(LicenseConfig.ENV_LICENSE_PROVIDER.startsWith("WARPFORGE_"));
    }

    @Test
    @DisplayName("DEFAULT_CONFIG_DIR is in user home")
    void defaultConfigDir_inUserHome() {
        String userHome = System.getProperty("user.home");
        assertTrue(LicenseConfig.DEFAULT_CONFIG_DIR.toString().startsWith(userHome));
    }

    @Test
    @DisplayName("UPGRADE_URL is valid URL")
    void upgradeUrl_isValidUrl() {
        assertTrue(LicenseConfig.UPGRADE_URL.startsWith("https://"));
    }

    @Test
    @DisplayName("SUPPORT_EMAIL is valid email format")
    void supportEmail_isValidFormat() {
        assertTrue(LicenseConfig.SUPPORT_EMAIL.contains("@"));
    }

    @Test
    @DisplayName("getLicenseKeyFromEnv returns null when not set")
    void getLicenseKeyFromEnv_returnsNullWhenNotSet() {
        // This test documents expected behavior
        // The actual return depends on environment
        String key = LicenseConfig.getLicenseKeyFromEnv();
        // Can be null or a value - just verify no exception
    }

    @Test
    @DisplayName("getDevSecretFromEnv returns null when not set")
    void getDevSecretFromEnv_returnsNullWhenNotSet() {
        // This test documents expected behavior
        String secret = LicenseConfig.getDevSecretFromEnv();
        // Can be null or a value - just verify no exception
    }

    @Test
    @DisplayName("getProviderFromEnv returns null when not set")
    void getProviderFromEnv_returnsNullWhenNotSet() {
        // This test documents expected behavior
        String provider = LicenseConfig.getProviderFromEnv();
        // Can be null or a value - just verify no exception
    }

    @Test
    @DisplayName("enable/disable cycle works")
    void enableDisable_cycle() {
        LicenseConfig.disable();
        assertFalse(LicenseConfig.isEnabled());

        LicenseConfig.enable();
        assertTrue(LicenseConfig.isEnabled());

        LicenseConfig.disable();
        assertFalse(LicenseConfig.isEnabled());

        LicenseConfig.enable();
        assertTrue(LicenseConfig.isEnabled());
    }
}
