package io.surfworks.warpforge.license;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.time.Instant;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ActivationResult} record.
 */
class ActivationResultTest {

    @Test
    @DisplayName("success() creates successful result with license")
    void success_createsSuccessfulResult() {
        var license = new LicenseInfo(
            "key", "instance", WarpForgeProduct.WARPFORGE_PRO,
            Instant.now().plusSeconds(86400),
            Instant.now(), Instant.now(),
            "fingerprint", "test@example.com", Map.of()
        );

        var result = ActivationResult.success(license);

        assertTrue(result.success());
        assertEquals(license, result.license());
        assertNull(result.error());
        assertEquals(ActivationResult.ErrorCode.NONE, result.errorCode());
    }

    @Test
    @DisplayName("failure() with message creates failed result")
    void failure_withMessage_createsFailedResult() {
        var result = ActivationResult.failure("Invalid license key");

        assertFalse(result.success());
        assertNull(result.license());
        assertEquals("Invalid license key", result.error());
        assertEquals(ActivationResult.ErrorCode.UNKNOWN, result.errorCode());
    }

    @Test
    @DisplayName("failure() with message and code creates failed result")
    void failure_withMessageAndCode_createsFailedResult() {
        var result = ActivationResult.failure(
            "License key is invalid",
            ActivationResult.ErrorCode.INVALID_KEY
        );

        assertFalse(result.success());
        assertNull(result.license());
        assertEquals("License key is invalid", result.error());
        assertEquals(ActivationResult.ErrorCode.INVALID_KEY, result.errorCode());
    }

    @Test
    @DisplayName("All error codes are defined")
    void errorCodes_allDefined() {
        var codes = ActivationResult.ErrorCode.values();

        assertTrue(codes.length >= 7);
        assertNotNull(ActivationResult.ErrorCode.NONE);
        assertNotNull(ActivationResult.ErrorCode.INVALID_KEY);
        assertNotNull(ActivationResult.ErrorCode.ACTIVATION_LIMIT_REACHED);
        assertNotNull(ActivationResult.ErrorCode.KEY_EXPIRED);
        assertNotNull(ActivationResult.ErrorCode.KEY_DISABLED);
        assertNotNull(ActivationResult.ErrorCode.NETWORK_ERROR);
        assertNotNull(ActivationResult.ErrorCode.UNKNOWN);
    }

    @Test
    @DisplayName("Record components are accessible")
    void recordComponents_areAccessible() {
        var license = LicenseInfo.devBypass();
        var result = new ActivationResult(true, license, null, ActivationResult.ErrorCode.NONE);

        assertTrue(result.success());
        assertEquals(license, result.license());
        assertNull(result.error());
        assertEquals(ActivationResult.ErrorCode.NONE, result.errorCode());
    }

    @Test
    @DisplayName("failure() for activation limit")
    void failure_activationLimit() {
        var result = ActivationResult.failure(
            "Maximum activations reached",
            ActivationResult.ErrorCode.ACTIVATION_LIMIT_REACHED
        );

        assertFalse(result.success());
        assertEquals(ActivationResult.ErrorCode.ACTIVATION_LIMIT_REACHED, result.errorCode());
    }

    @Test
    @DisplayName("failure() for expired key")
    void failure_expiredKey() {
        var result = ActivationResult.failure(
            "License key has expired",
            ActivationResult.ErrorCode.KEY_EXPIRED
        );

        assertFalse(result.success());
        assertEquals(ActivationResult.ErrorCode.KEY_EXPIRED, result.errorCode());
    }

    @Test
    @DisplayName("failure() for network error")
    void failure_networkError() {
        var result = ActivationResult.failure(
            "Unable to connect to license server",
            ActivationResult.ErrorCode.NETWORK_ERROR
        );

        assertFalse(result.success());
        assertEquals(ActivationResult.ErrorCode.NETWORK_ERROR, result.errorCode());
    }
}
