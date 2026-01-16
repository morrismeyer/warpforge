package io.surfworks.warpforge.launch.artifact;

import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Registry of artifact stores.
 *
 * <p>Follows the BackendRegistry pattern. Stores are registered by name
 * and can be retrieved lazily when needed.
 */
public final class ArtifactRegistry {

    private static final ArtifactRegistry INSTANCE = new ArtifactRegistry();

    private final Map<String, Supplier<ArtifactStore>> stores = new ConcurrentHashMap<>();
    private volatile String defaultStore = "local";

    private ArtifactRegistry() {
    }

    /**
     * Returns the singleton registry instance.
     */
    public static ArtifactRegistry instance() {
        return INSTANCE;
    }

    /**
     * Registers an artifact store supplier.
     *
     * @param name     unique name for the store
     * @param supplier factory that creates the store when needed
     */
    public void register(String name, Supplier<ArtifactStore> supplier) {
        stores.put(name, supplier);
    }

    /**
     * Gets an artifact store by name.
     *
     * @param name the store name
     * @return the store if registered
     */
    public Optional<ArtifactStore> get(String name) {
        Supplier<ArtifactStore> supplier = stores.get(name);
        if (supplier == null) {
            return Optional.empty();
        }
        return Optional.of(supplier.get());
    }

    /**
     * Gets the default artifact store.
     *
     * @return the default store
     * @throws IllegalStateException if no default is registered
     */
    public ArtifactStore getDefault() {
        return get(defaultStore)
                .orElseThrow(() -> new IllegalStateException(
                        "Default artifact store '" + defaultStore + "' is not registered"));
    }

    /**
     * Sets the default store name.
     *
     * @param name the store to use as default
     */
    public void setDefault(String name) {
        this.defaultStore = name;
    }

    /**
     * Returns the names of all registered stores.
     */
    public Set<String> available() {
        return Set.copyOf(stores.keySet());
    }

    /**
     * Returns the current default store name.
     */
    public String defaultName() {
        return defaultStore;
    }

    /**
     * Clears all registered stores. For testing only.
     */
    public void clear() {
        stores.clear();
        defaultStore = "local";
    }
}
