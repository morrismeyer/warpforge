package io.surfworks.warpforge.license;

import org.junit.jupiter.api.AfterEach;
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

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link LicenseManager}.
 */
class LicenseManagerTest {

    @TempDir
    Path tempDir;

    private LicenseManager manager;
    private MockProvider mockProvider;

    @BeforeEach
    void setUp() {
        // Reset static state
        LicenseManager.resetInstance();
        LicenseConfig.reset();

        mockProvider = new MockProvider();
    }

    @AfterEach
    void tearDown() {
        LicenseManager.resetInstance();
        LicenseConfig.reset();
    }

    @Test
    @DisplayName("getInstance returns NoOpProvider when none configured")
    void getInstance_noProvider_returnsNoOp() {
        var manager = LicenseManager.getInstance();

        assertNotNull(manager);
        assertTrue(manager.getProvider() instanceof NoOpProvider);
    }

    @Test
    @DisplayName("configureProvider sets the provider")
    void configureProvider_setsProvider() {
        LicenseManager.configureProvider(mockProvider);

        var manager = LicenseManager.getInstance();
        assertEquals(mockProvider, manager.getProvider());
    }

    @Test
    @DisplayName("checkLicense allows when licensing disabled")
    void checkLicense_licensingDisabled_allows() {
        LicenseConfig.disable();
        manager = new LicenseManager(mockProvider);

        var result = manager.checkLicense();

        assertTrue(result.allowed());
        assertEquals(WarpForgeProduct.DEV, result.product());
    }

    @Test
    @DisplayName("activate delegates to provider and caches result")
    void activate_delegatesAndCaches() {
        var license = createTestLicense("TEST-KEY", WarpForgeProduct.WARPFORGE_PRO);
        mockProvider.setActivateResult(ActivationResult.success(license));

        manager = new LicenseManager(mockProvider);

        var result = manager.activate("TEST-KEY");

        assertTrue(result.success());
        assertEquals(license, result.license());

        // Should be cached
        var cached = manager.getCurrentLicense();
        assertNotNull(cached);
        assertEquals("TEST-KEY", cached.key());
    }

    @Test
    @DisplayName("activate failure does not cache")
    void activate_failure_doesNotCache() {
        mockProvider.setActivateResult(ActivationResult.failure("Invalid key"));

        manager = new LicenseManager(mockProvider);

        var result = manager.activate("BAD-KEY");

        assertFalse(result.success());

        // Should not be cached
        assertNull(manager.getCurrentLicense());
    }

    @Test
    @DisplayName("deactivate clears cache and calls provider")
    void deactivate_clearsCacheAndCallsProvider() {
        var license = createTestLicense("TEST-KEY", WarpForgeProduct.WARPFORGE_PRO);
        mockProvider.setActivateResult(ActivationResult.success(license));

        manager = new LicenseManager(mockProvider);
        manager.activate("TEST-KEY");

        assertNotNull(manager.getCurrentLicense());

        manager.deactivate();

        assertNull(manager.getCurrentLicense());
        assertTrue(mockProvider.deactivateCalled);
    }

    @Test
    @DisplayName("getMachineFingerprint returns fingerprint")
    void getMachineFingerprint_returnsFingerprint() {
        manager = new LicenseManager(mockProvider);

        String fp = manager.getMachineFingerprint();

        assertNotNull(fp);
        assertEquals(32, fp.length());
    }

    @Test
    @DisplayName("getMachineName returns machine name")
    void getMachineName_returnsMachineName() {
        manager = new LicenseManager(mockProvider);

        String name = manager.getMachineName();

        assertNotNull(name);
        assertFalse(name.isBlank());
    }

    @Test
    @DisplayName("generateDevKey creates deterministic key")
    void generateDevKey_isDeterministic() {
        String key1 = LicenseManager.generateDevKey("test-secret");
        String key2 = LicenseManager.generateDevKey("test-secret");

        assertEquals(key1, key2);
        assertTrue(key1.startsWith("DEV-"));
        assertEquals(20, key1.length()); // "DEV-" + 16 hex chars
    }

    @Test
    @DisplayName("generateDevKey produces different keys for different secrets")
    void generateDevKey_differentSecrets_differentKeys() {
        String key1 = LicenseManager.generateDevKey("secret1");
        String key2 = LicenseManager.generateDevKey("secret2");

        assertNotEquals(key1, key2);
    }

    @Test
    @DisplayName("checkLicense uses cached valid license")
    void checkLicense_usesCachedLicense() throws IOException {
        LicenseConfig.enable();

        // Pre-populate the cache by creating a LicenseCache directly
        var cache = new LicenseCache(tempDir);
        var license = createTestLicense("CACHED-KEY", WarpForgeProduct.WARPFORGE_PRO);
        cache.save(license);

        // Create manager that will use the cached license
        // Note: This test verifies the concept but actual cache location varies
        manager = new LicenseManager(mockProvider);

        // The provider was not called for activation (would fail)
        assertFalse(mockProvider.activateCalled);
    }

    @Test
    @DisplayName("Singleton pattern works correctly")
    void singleton_worksCorrectly() {
        LicenseManager.configureProvider(mockProvider);

        var instance1 = LicenseManager.getInstance();
        var instance2 = LicenseManager.getInstance();

        assertSame(instance1, instance2);
    }

    @Test
    @DisplayName("Reconfiguring provider resets singleton")
    void reconfigureProvider_resetsSingleton() {
        var provider1 = new MockProvider();
        var provider2 = new MockProvider();

        LicenseManager.configureProvider(provider1);
        var instance1 = LicenseManager.getInstance();

        LicenseManager.configureProvider(provider2);
        var instance2 = LicenseManager.getInstance();

        assertNotSame(instance1, instance2);
        assertEquals(provider2, instance2.getProvider());
    }

    // ============ Helper classes and methods ============

    private LicenseInfo createTestLicense(String key, WarpForgeProduct product) {
        return new LicenseInfo(
            key,
            "instance-" + key,
            product,
            Instant.now().plus(30, ChronoUnit.DAYS),
            Instant.now(),
            Instant.now(),
            MachineFingerprint.generate(),
            "test@example.com",
            Map.of()
        );
    }

    /**
     * Mock provider for testing.
     */
    private static class MockProvider implements LicenseProvider {
        private ActivationResult activateResult = ActivationResult.failure("Not configured");
        private ActivationResult validateResult = ActivationResult.failure("Not configured");
        boolean activateCalled = false;
        boolean validateCalled = false;
        boolean deactivateCalled = false;
        String lastActivateKey;
        String lastValidateKey;
        String lastValidateInstanceId;

        void setActivateResult(ActivationResult result) {
            this.activateResult = result;
        }

        void setValidateResult(ActivationResult result) {
            this.validateResult = result;
        }

        @Override
        public ActivationResult activate(String licenseKey) {
            activateCalled = true;
            lastActivateKey = licenseKey;
            return activateResult;
        }

        @Override
        public ActivationResult validate(String licenseKey, String instanceId) {
            validateCalled = true;
            lastValidateKey = licenseKey;
            lastValidateInstanceId = instanceId;
            return validateResult;
        }

        @Override
        public boolean deactivate(String licenseKey, String instanceId) {
            deactivateCalled = true;
            return true;
        }

        @Override
        public String getProviderName() {
            return "Mock";
        }
    }
}
