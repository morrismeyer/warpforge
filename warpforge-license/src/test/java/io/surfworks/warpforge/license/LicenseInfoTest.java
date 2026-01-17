package io.surfworks.warpforge.license;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link LicenseInfo} record.
 */
class LicenseInfoTest {

    @Test
    @DisplayName("isExpired returns false for perpetual license (null validUntil)")
    void isExpired_perpetualLicense_returnsFalse() {
        var license = new LicenseInfo(
            "key", "instance", WarpForgeProduct.WARPFORGE_PRO,
            null, // perpetual
            Instant.now(), Instant.now(),
            "fingerprint", "test@example.com", Map.of()
        );

        assertFalse(license.isExpired());
    }

    @Test
    @DisplayName("isExpired returns false for license expiring in the future")
    void isExpired_futureExpiration_returnsFalse() {
        var license = new LicenseInfo(
            "key", "instance", WarpForgeProduct.WARPFORGE_PRO,
            Instant.now().plus(30, ChronoUnit.DAYS),
            Instant.now(), Instant.now(),
            "fingerprint", "test@example.com", Map.of()
        );

        assertFalse(license.isExpired());
    }

    @Test
    @DisplayName("isExpired returns true for license expired in the past")
    void isExpired_pastExpiration_returnsTrue() {
        var license = new LicenseInfo(
            "key", "instance", WarpForgeProduct.WARPFORGE_PRO,
            Instant.now().minus(1, ChronoUnit.DAYS),
            Instant.now().minus(30, ChronoUnit.DAYS), Instant.now(),
            "fingerprint", "test@example.com", Map.of()
        );

        assertTrue(license.isExpired());
    }

    @Test
    @DisplayName("isWithinGracePeriod returns true for perpetual license")
    void isWithinGracePeriod_perpetualLicense_returnsTrue() {
        var license = new LicenseInfo(
            "key", "instance", WarpForgeProduct.WARPFORGE_PRO,
            null, // perpetual
            Instant.now(), Instant.now(),
            "fingerprint", null, Map.of()
        );

        assertTrue(license.isWithinGracePeriod());
    }

    @Test
    @DisplayName("isWithinGracePeriod returns true within 7-day grace period")
    void isWithinGracePeriod_withinGracePeriod_returnsTrue() {
        // Expired 3 days ago (within 7-day grace period)
        var license = new LicenseInfo(
            "key", "instance", WarpForgeProduct.WARPFORGE_PRO,
            Instant.now().minus(3, ChronoUnit.DAYS),
            Instant.now().minus(30, ChronoUnit.DAYS), Instant.now(),
            "fingerprint", null, Map.of()
        );

        assertTrue(license.isWithinGracePeriod());
    }

    @Test
    @DisplayName("isWithinGracePeriod returns false after 7-day grace period")
    void isWithinGracePeriod_afterGracePeriod_returnsFalse() {
        // Expired 10 days ago (beyond 7-day grace period)
        var license = new LicenseInfo(
            "key", "instance", WarpForgeProduct.WARPFORGE_PRO,
            Instant.now().minus(10, ChronoUnit.DAYS),
            Instant.now().minus(40, ChronoUnit.DAYS), Instant.now(),
            "fingerprint", null, Map.of()
        );

        assertFalse(license.isWithinGracePeriod());
    }

    @Test
    @DisplayName("needsRevalidation returns true when lastValidated is null")
    void needsRevalidation_nullLastValidated_returnsTrue() {
        var license = new LicenseInfo(
            "key", "instance", WarpForgeProduct.WARPFORGE_PRO,
            Instant.now().plus(30, ChronoUnit.DAYS),
            Instant.now(), null, // never validated
            "fingerprint", null, Map.of()
        );

        assertTrue(license.needsRevalidation());
    }

    @Test
    @DisplayName("needsRevalidation returns true after 7 days")
    void needsRevalidation_olderThan7Days_returnsTrue() {
        var license = new LicenseInfo(
            "key", "instance", WarpForgeProduct.WARPFORGE_PRO,
            Instant.now().plus(30, ChronoUnit.DAYS),
            Instant.now().minus(30, ChronoUnit.DAYS),
            Instant.now().minus(8, ChronoUnit.DAYS), // validated 8 days ago
            "fingerprint", null, Map.of()
        );

        assertTrue(license.needsRevalidation());
    }

    @Test
    @DisplayName("needsRevalidation returns false when recently validated")
    void needsRevalidation_recentlyValidated_returnsFalse() {
        var license = new LicenseInfo(
            "key", "instance", WarpForgeProduct.WARPFORGE_PRO,
            Instant.now().plus(30, ChronoUnit.DAYS),
            Instant.now().minus(30, ChronoUnit.DAYS),
            Instant.now().minus(1, ChronoUnit.DAYS), // validated 1 day ago
            "fingerprint", null, Map.of()
        );

        assertFalse(license.needsRevalidation());
    }

    @Test
    @DisplayName("getMaskedKey masks middle of long key")
    void getMaskedKey_longKey_masksProperly() {
        var license = new LicenseInfo(
            "ABCD-1234-5678-WXYZ", "instance", WarpForgeProduct.WARPFORGE_PRO,
            null, null, null, null, null, Map.of()
        );

        assertEquals("ABCD...WXYZ", license.getMaskedKey());
    }

    @Test
    @DisplayName("getMaskedKey returns stars for short key")
    void getMaskedKey_shortKey_returnsStars() {
        var license = new LicenseInfo(
            "ABC", "instance", WarpForgeProduct.WARPFORGE_PRO,
            null, null, null, null, null, Map.of()
        );

        assertEquals("****", license.getMaskedKey());
    }

    @Test
    @DisplayName("getMaskedKey returns stars for null key")
    void getMaskedKey_nullKey_returnsStars() {
        var license = new LicenseInfo(
            null, "instance", WarpForgeProduct.WARPFORGE_PRO,
            null, null, null, null, null, Map.of()
        );

        assertEquals("****", license.getMaskedKey());
    }

    @Test
    @DisplayName("freeTier creates correct license info")
    void freeTier_createsCorrectInfo() {
        var license = LicenseInfo.freeTier();

        assertNull(license.key());
        assertNull(license.instanceId());
        assertEquals(WarpForgeProduct.FREE, license.product());
        assertNull(license.validUntil());
        assertNotNull(license.metadata());
    }

    @Test
    @DisplayName("devBypass creates correct license info")
    void devBypass_createsCorrectInfo() {
        var license = LicenseInfo.devBypass();

        assertEquals("DEV-INTERNAL", license.key());
        assertEquals("dev", license.instanceId());
        assertEquals(WarpForgeProduct.DEV, license.product());
        assertNotNull(license.validUntil());
        assertFalse(license.isExpired());
        assertEquals(true, license.metadata().get("dev"));
    }
}
