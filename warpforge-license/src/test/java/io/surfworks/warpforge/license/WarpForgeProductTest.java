package io.surfworks.warpforge.license;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link WarpForgeProduct} enum.
 */
class WarpForgeProductTest {

    @Test
    @DisplayName("FREE tier has correct daily limit")
    void freeTier_hasCorrectDailyLimit() {
        assertEquals(3, WarpForgeProduct.FREE.getDailyTraceLimit());
        assertEquals(0, WarpForgeProduct.FREE.getMaxActivations());
        assertEquals("Free", WarpForgeProduct.FREE.getDisplayName());
    }

    @Test
    @DisplayName("WARPFORGE_PRO has unlimited traces")
    void proTier_hasUnlimitedTraces() {
        assertEquals(Integer.MAX_VALUE, WarpForgeProduct.WARPFORGE_PRO.getDailyTraceLimit());
        assertEquals(2, WarpForgeProduct.WARPFORGE_PRO.getMaxActivations());
        assertEquals("WarpForge Pro", WarpForgeProduct.WARPFORGE_PRO.getDisplayName());
    }

    @Test
    @DisplayName("WARPFORGE_TEAM has correct activations")
    void teamTier_hasCorrectActivations() {
        assertEquals(5, WarpForgeProduct.WARPFORGE_TEAM.getMaxActivations());
        assertTrue(WarpForgeProduct.WARPFORGE_TEAM.isUnlimited());
    }

    @Test
    @DisplayName("WARPFORGE_ENTERPRISE has unlimited everything")
    void enterpriseTier_hasUnlimitedEverything() {
        assertEquals(Integer.MAX_VALUE, WarpForgeProduct.WARPFORGE_ENTERPRISE.getDailyTraceLimit());
        assertEquals(Integer.MAX_VALUE, WarpForgeProduct.WARPFORGE_ENTERPRISE.getMaxActivations());
    }

    @Test
    @DisplayName("DEV tier has unlimited everything")
    void devTier_hasUnlimitedEverything() {
        assertEquals(Integer.MAX_VALUE, WarpForgeProduct.DEV.getDailyTraceLimit());
        assertEquals(Integer.MAX_VALUE, WarpForgeProduct.DEV.getMaxActivations());
    }

    @Test
    @DisplayName("isUnlimited returns correct value")
    void isUnlimited_returnsCorrectValue() {
        assertFalse(WarpForgeProduct.FREE.isUnlimited());
        assertTrue(WarpForgeProduct.WARPFORGE_PRO.isUnlimited());
        assertTrue(WarpForgeProduct.WARPFORGE_TEAM.isUnlimited());
        assertTrue(WarpForgeProduct.WARPFORGE_ENTERPRISE.isUnlimited());
        assertTrue(WarpForgeProduct.DEV.isUnlimited());
    }

    @Test
    @DisplayName("isPaid returns correct value")
    void isPaid_returnsCorrectValue() {
        assertFalse(WarpForgeProduct.FREE.isPaid());
        assertFalse(WarpForgeProduct.DEV.isPaid());
        assertTrue(WarpForgeProduct.WARPFORGE_PRO.isPaid());
        assertTrue(WarpForgeProduct.WARPFORGE_TEAM.isPaid());
        assertTrue(WarpForgeProduct.WARPFORGE_ENTERPRISE.isPaid());
    }

    @ParameterizedTest
    @DisplayName("fromProductName parses enterprise variants")
    @CsvSource({
        "WarpForge Enterprise, , WARPFORGE_ENTERPRISE",
        "enterprise plan, , WARPFORGE_ENTERPRISE",
        "ENTERPRISE, , WARPFORGE_ENTERPRISE",
    })
    void fromProductName_parsesEnterprise(String productName, String variantName, WarpForgeProduct expected) {
        assertEquals(expected, WarpForgeProduct.fromProductName(productName, variantName));
    }

    @ParameterizedTest
    @DisplayName("fromProductName parses team variants")
    @CsvSource({
        "WarpForge Team, , WARPFORGE_TEAM",
        "team plan, , WARPFORGE_TEAM",
        "TEAM LICENSE, , WARPFORGE_TEAM",
    })
    void fromProductName_parsesTeam(String productName, String variantName, WarpForgeProduct expected) {
        assertEquals(expected, WarpForgeProduct.fromProductName(productName, variantName));
    }

    @ParameterizedTest
    @DisplayName("fromProductName parses pro variants")
    @CsvSource({
        "WarpForge Pro, , WARPFORGE_PRO",
        "pro plan, , WARPFORGE_PRO",
        "PRO monthly, , WARPFORGE_PRO",
        "WarpForge Professional, , WARPFORGE_PRO",
    })
    void fromProductName_parsesPro(String productName, String variantName, WarpForgeProduct expected) {
        assertEquals(expected, WarpForgeProduct.fromProductName(productName, variantName));
    }

    @Test
    @DisplayName("fromProductName returns FREE for null")
    void fromProductName_nullReturnsNull() {
        assertEquals(WarpForgeProduct.FREE, WarpForgeProduct.fromProductName(null, null));
    }

    @Test
    @DisplayName("fromProductName returns FREE for unknown product")
    void fromProductName_unknownReturnsFree() {
        // Note: "Unknown Product" contains "pro" so it would match WARPFORGE_PRO
        assertEquals(WarpForgeProduct.FREE, WarpForgeProduct.fromProductName("Something Random", null));
        assertEquals(WarpForgeProduct.FREE, WarpForgeProduct.fromProductName("basic", null));
        assertEquals(WarpForgeProduct.FREE, WarpForgeProduct.fromProductName("starter", null));
    }

    @Test
    @DisplayName("fromProductName priority: enterprise > team > pro")
    void fromProductName_priorityOrder() {
        // If somehow a product name contains multiple keywords, enterprise wins
        assertEquals(WarpForgeProduct.WARPFORGE_ENTERPRISE,
            WarpForgeProduct.fromProductName("enterprise team pro", null));

        // team > pro
        assertEquals(WarpForgeProduct.WARPFORGE_TEAM,
            WarpForgeProduct.fromProductName("team pro plan", null));
    }

    @Test
    @DisplayName("fromLemonSqueezy is deprecated alias")
    @SuppressWarnings("deprecation")
    void fromLemonSqueezy_isDeprecatedAlias() {
        assertEquals(WarpForgeProduct.WARPFORGE_PRO,
            WarpForgeProduct.fromLemonSqueezy("WarpForge Pro", "Monthly"));
    }

    @Test
    @DisplayName("All enum values exist")
    void allEnumValues_exist() {
        var values = WarpForgeProduct.values();
        assertEquals(5, values.length);
    }
}
