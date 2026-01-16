package io.surfworks.warpforge.launch.scheduler;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Registry for scheduler implementations.
 * Allows registration and discovery of execution schedulers.
 *
 * <p>Thread-safe. Schedulers are created on demand via factory functions.
 */
public final class SchedulerRegistry {

    private static final Map<String, Supplier<Scheduler>> FACTORIES = new ConcurrentHashMap<>();
    private static volatile String defaultSchedulerName = "local";

    private SchedulerRegistry() {} // Utility class

    /**
     * Registers a scheduler factory.
     *
     * @param name    Scheduler name (e.g., "local", "ray")
     * @param factory Factory function that creates scheduler instances
     */
    public static void register(String name, Supplier<Scheduler> factory) {
        FACTORIES.put(name.toLowerCase(), factory);
    }

    /**
     * Unregisters a scheduler.
     *
     * @param name Scheduler name to remove
     */
    public static void unregister(String name) {
        FACTORIES.remove(name.toLowerCase());
    }

    /**
     * Checks if a scheduler is registered.
     *
     * @param name Scheduler name
     * @return true if the scheduler is registered
     */
    public static boolean isRegistered(String name) {
        return FACTORIES.containsKey(name.toLowerCase());
    }

    /**
     * Gets a new instance of a registered scheduler.
     *
     * @param name Scheduler name
     * @return A new scheduler instance
     * @throws IllegalArgumentException if the scheduler is not registered
     */
    public static Scheduler get(String name) {
        Supplier<Scheduler> factory = FACTORIES.get(name.toLowerCase());
        if (factory == null) {
            throw new IllegalArgumentException(
                    "Scheduler '" + name + "' not registered. Available: " + available());
        }
        return factory.get();
    }

    /**
     * Gets a new instance of the default scheduler.
     *
     * @return A new instance of the default scheduler
     * @throws IllegalStateException if no schedulers are registered
     */
    public static Scheduler getDefault() {
        if (FACTORIES.isEmpty()) {
            throw new IllegalStateException("No schedulers registered");
        }
        if (FACTORIES.containsKey(defaultSchedulerName)) {
            return get(defaultSchedulerName);
        }
        // Fall back to first available
        return FACTORIES.values().iterator().next().get();
    }

    /**
     * Sets the default scheduler name.
     *
     * @param name Scheduler name to use as default
     * @throws IllegalArgumentException if the scheduler is not registered
     */
    public static void setDefault(String name) {
        if (!isRegistered(name)) {
            throw new IllegalArgumentException("Scheduler '" + name + "' not registered");
        }
        defaultSchedulerName = name.toLowerCase();
    }

    /**
     * Gets the name of the default scheduler.
     *
     * @return Default scheduler name
     */
    public static String getDefaultName() {
        return defaultSchedulerName;
    }

    /**
     * Gets list of available scheduler names.
     *
     * @return List of registered scheduler names
     */
    public static List<String> available() {
        return List.copyOf(FACTORIES.keySet());
    }

    /**
     * Clears all registered schedulers (mainly for testing).
     */
    public static void clear() {
        FACTORIES.clear();
        defaultSchedulerName = "local";
    }
}
