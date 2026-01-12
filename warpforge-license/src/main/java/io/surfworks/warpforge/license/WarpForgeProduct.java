package io.surfworks.warpforge.license;

/**
 * WarpForge product tiers matching Lemon Squeezy product configuration.
 *
 * <p>Products:
 * <ul>
 *   <li>WarpForge Pro - $29/month or $290/year (variant)</li>
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
     * Parse product from Lemon Squeezy variant name.
     *
     * @param variantName the variant name from Lemon Squeezy API (e.g., "WarpForge Pro", "Monthly", "Annual")
     * @param productName the product name from Lemon Squeezy API
     * @return the matching WarpForgeProduct
     */
    public static WarpForgeProduct fromLemonSqueezy(String productName, String variantName) {
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
}
