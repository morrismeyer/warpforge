package io.surfworks.warpforge.license;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link LicenseCheckResult} record.
 */
class LicenseCheckResultTest {

    @Test
    @DisplayName("allowed() creates allowed result with product")
    void allowed_createsAllowedResult() {
        var result = LicenseCheckResult.allowed(WarpForgeProduct.WARPFORGE_PRO);

        assertTrue(result.allowed());
        assertEquals(WarpForgeProduct.WARPFORGE_PRO, result.product());
        assertNull(result.message());
        assertNull(result.warning());
        assertNull(result.upgradeUrl());
    }

    @Test
    @DisplayName("allowed() with message creates result with message")
    void allowed_withMessage_createsResultWithMessage() {
        var result = LicenseCheckResult.allowed(WarpForgeProduct.FREE, "1/3 traces today");

        assertTrue(result.allowed());
        assertEquals(WarpForgeProduct.FREE, result.product());
        assertEquals("1/3 traces today", result.message());
        assertNull(result.warning());
        assertNull(result.upgradeUrl());
    }

    @Test
    @DisplayName("allowedWithWarning() creates result with warning")
    void allowedWithWarning_createsResultWithWarning() {
        var result = LicenseCheckResult.allowedWithWarning(
            WarpForgeProduct.WARPFORGE_PRO,
            "License expires in 3 days"
        );

        assertTrue(result.allowed());
        assertEquals(WarpForgeProduct.WARPFORGE_PRO, result.product());
        assertNull(result.message());
        assertEquals("License expires in 3 days", result.warning());
        assertNull(result.upgradeUrl());
    }

    @Test
    @DisplayName("denied() creates denied result with default upgrade URL")
    void denied_createsDeniedResult() {
        var result = LicenseCheckResult.denied("Free tier limit reached");

        assertFalse(result.allowed());
        assertEquals(WarpForgeProduct.FREE, result.product());
        assertEquals("Free tier limit reached", result.message());
        assertNull(result.warning());
        assertEquals("https://surfworks.energy/pricing", result.upgradeUrl());
    }

    @Test
    @DisplayName("denied() with custom URL creates result with custom URL")
    void denied_withCustomUrl_createsResultWithCustomUrl() {
        var result = LicenseCheckResult.denied("Custom error", "https://custom.url/upgrade");

        assertFalse(result.allowed());
        assertEquals(WarpForgeProduct.FREE, result.product());
        assertEquals("Custom error", result.message());
        assertEquals("https://custom.url/upgrade", result.upgradeUrl());
    }

    @Test
    @DisplayName("Result record components are accessible")
    void recordComponents_areAccessible() {
        var result = new LicenseCheckResult(
            true,
            WarpForgeProduct.WARPFORGE_TEAM,
            "message",
            "warning",
            "url"
        );

        assertTrue(result.allowed());
        assertEquals(WarpForgeProduct.WARPFORGE_TEAM, result.product());
        assertEquals("message", result.message());
        assertEquals("warning", result.warning());
        assertEquals("url", result.upgradeUrl());
    }
}
