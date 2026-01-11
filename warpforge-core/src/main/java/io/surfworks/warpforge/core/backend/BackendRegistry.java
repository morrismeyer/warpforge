package io.surfworks.warpforge.core.backend;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Registry for backend implementations.
 * Allows registration and discovery of execution backends.
 */
public final class BackendRegistry {

    private static final Map<String, Supplier<Backend>> FACTORIES = new ConcurrentHashMap<>();
    private static String defaultBackendName = "cpu";

    private BackendRegistry() {} // Utility class

    /**
     * Register a backend factory.
     *
     * @param name    Backend name (e.g., "cpu", "nvidia")
     * @param factory Factory function that creates backend instances
     */
    public static void register(String name, Supplier<Backend> factory) {
        FACTORIES.put(name.toLowerCase(), factory);
    }

    /**
     * Unregister a backend.
     *
     * @param name Backend name to remove
     */
    public static void unregister(String name) {
        FACTORIES.remove(name.toLowerCase());
    }

    /**
     * Check if a backend is registered.
     *
     * @param name Backend name
     * @return true if the backend is registered
     */
    public static boolean isRegistered(String name) {
        return FACTORIES.containsKey(name.toLowerCase());
    }

    /**
     * Get a new instance of a registered backend.
     *
     * @param name Backend name
     * @return A new backend instance
     * @throws IllegalArgumentException if the backend is not registered
     */
    public static Backend get(String name) {
        Supplier<Backend> factory = FACTORIES.get(name.toLowerCase());
        if (factory == null) {
            throw new IllegalArgumentException(
                "Backend '" + name + "' not registered. Available: " + available());
        }
        return factory.get();
    }

    /**
     * Get a new instance of the default backend.
     *
     * @return A new instance of the default backend
     * @throws IllegalStateException if no backends are registered
     */
    public static Backend getDefault() {
        if (FACTORIES.isEmpty()) {
            throw new IllegalStateException("No backends registered");
        }
        if (FACTORIES.containsKey(defaultBackendName)) {
            return get(defaultBackendName);
        }
        // Fall back to first available
        return FACTORIES.values().iterator().next().get();
    }

    /**
     * Set the default backend name.
     *
     * @param name Backend name to use as default
     */
    public static void setDefault(String name) {
        if (!isRegistered(name)) {
            throw new IllegalArgumentException("Backend '" + name + "' not registered");
        }
        defaultBackendName = name.toLowerCase();
    }

    /**
     * Get the name of the default backend.
     */
    public static String getDefaultName() {
        return defaultBackendName;
    }

    /**
     * Get list of available backend names.
     *
     * @return List of registered backend names
     */
    public static List<String> available() {
        return List.copyOf(FACTORIES.keySet());
    }

    /**
     * Clear all registered backends (mainly for testing).
     */
    public static void clear() {
        FACTORIES.clear();
        defaultBackendName = "cpu";
    }
}
