package io.surfworks.warpforge.license;

import java.util.Map;

/**
 * Maps provider-specific product/variant names to WarpForge products.
 *
 * <p>Each license provider has its own way of naming products and variants.
 * This interface allows provider implementations to define their own mapping
 * logic while the rest of the license system works with {@link WarpForgeProduct}.
 */
public interface ProductMapper {

    /**
     * Map provider-specific product information to a WarpForge product.
     *
     * @param productName the product name from the provider (may be null)
     * @param variantName the variant name from the provider (may be null)
     * @param metadata additional metadata that might help with mapping
     * @return the corresponding WarpForgeProduct, defaults to FREE if unmappable
     */
    WarpForgeProduct mapProduct(String productName, String variantName, Map<String, Object> metadata);

    /**
     * Default mapper that uses simple name matching.
     *
     * <p>Looks for keywords like "enterprise", "team", "pro" in the product name.
     */
    ProductMapper DEFAULT = (productName, variantName, metadata) -> {
        if (productName == null) {
            return WarpForgeProduct.FREE;
        }

        String name = productName.toLowerCase();

        if (name.contains("enterprise")) {
            return WarpForgeProduct.WARPFORGE_ENTERPRISE;
        }
        if (name.contains("team")) {
            return WarpForgeProduct.WARPFORGE_TEAM;
        }
        if (name.contains("pro")) {
            return WarpForgeProduct.WARPFORGE_PRO;
        }

        return WarpForgeProduct.FREE;
    };
}
