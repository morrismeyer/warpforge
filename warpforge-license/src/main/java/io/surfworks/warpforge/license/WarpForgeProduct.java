package io.surfworks.warpforge.license;

/**
 * WarpForge product tiers.
 *
 * <p>Products:
 * <ul>
 *   <li>WarpForge Pro - $29/month or $290/year (individual developers)</li>
 *   <li>WarpForge Team - $49/user/month (future)</li>
 *   <li>WarpForge Enterprise - custom pricing (future)</li>
 * </ul>
 */
public enum WarpForgeProduct {

    /**
     * Free tier - limited usage, no license key required.
     */
    FREE("Free", 3, 0),

    /**
     * WarpForge Pro - unlimited usage for individual developers.
     * Includes both monthly ($29/mo) and annual ($290/yr) variants.
     */
    WARPFORGE_PRO("WarpForge Pro", Integer.MAX_VALUE, 2),

    /**
     * WarpForge Team - unlimited usage for teams.
     */
    WARPFORGE_TEAM("WarpForge Team", Integer.MAX_VALUE, 5),

    /**
     * WarpForge Enterprise - custom terms, unlimited everything.
     */
    WARPFORGE_ENTERPRISE("WarpForge Enterprise", Integer.MAX_VALUE, Integer.MAX_VALUE),

    /**
     * Developer/CI license - internal use only.
     */
    DEV("Developer", Integer.MAX_VALUE, Integer.MAX_VALUE);

    private final String displayName;
    private final int dailyTraceLimit;
    private final int maxActivations;

    WarpForgeProduct(String displayName, int dailyTraceLimit, int maxActivations) {
        this.displayName = displayName;
        this.dailyTraceLimit = dailyTraceLimit;
        this.maxActivations = maxActivations;
    }

    public String getDisplayName() {
        return displayName;
    }

    /**
     * Maximum traces per day. Integer.MAX_VALUE means unlimited.
     */
    public int getDailyTraceLimit() {
        return dailyTraceLimit;
    }

    /**
     * Maximum simultaneous machine activations.
     */
    public int getMaxActivations() {
        return maxActivations;
    }

    public boolean isUnlimited() {
        return dailyTraceLimit == Integer.MAX_VALUE;
    }

    public boolean isPaid() {
        return this != FREE && this != DEV;
    }

    /**
     * Parse product from provider-specific names using keyword matching.
     *
     * <p>This is a simple default implementation that looks for keywords
     * like "enterprise", "team", "pro" in the product name. For more
     * sophisticated mapping, use {@link ProductMapper}.
     *
     * @param productName the product name from the provider API
     * @param variantName the variant name (currently unused, for future use)
     * @return the matching WarpForgeProduct, or FREE if no match
     */
    public static WarpForgeProduct fromProductName(String productName, String variantName) {
        if (productName == null) {
            return FREE;
        }

        String name = productName.toLowerCase();

        if (name.contains("enterprise")) {
            return WARPFORGE_ENTERPRISE;
        }
        if (name.contains("team")) {
            return WARPFORGE_TEAM;
        }
        if (name.contains("pro")) {
            return WARPFORGE_PRO;
        }

        return FREE;
    }

    /**
     * Parse product from Lemon Squeezy variant name.
     *
     * @deprecated Use {@link #fromProductName(String, String)} or {@link ProductMapper} instead.
     */
    @Deprecated
    public static WarpForgeProduct fromLemonSqueezy(String productName, String variantName) {
        return fromProductName(productName, variantName);
    }
}
