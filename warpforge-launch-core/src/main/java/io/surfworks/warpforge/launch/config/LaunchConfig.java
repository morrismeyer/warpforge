package io.surfworks.warpforge.launch.config;

import io.surfworks.warpforge.launch.scheduler.kubernetes.KubernetesConfig;
import io.surfworks.warpforge.launch.scheduler.ray.RayConfig;
import io.surfworks.warpforge.launch.scheduler.slurm.SlurmConfig;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Configuration for warpforge-launch.
 *
 * <p>Configuration is loaded in order of precedence:
 * <ol>
 *   <li>CLI arguments (highest priority)</li>
 *   <li>Config file ({@code ~/.config/warpforge/launch.json})</li>
 *   <li>Auto-detection and defaults (lowest priority)</li>
 * </ol>
 *
 * <p>Environment variables are avoided for configuration per project philosophy.
 * The config file path can be overridden via CLI argument if needed.
 *
 * @param defaultScheduler  default scheduler name ("local", "ray", "kubernetes", "slurm")
 * @param defaultArtifactStore default artifact store name ("local", "shared")
 * @param rayConfig         Ray scheduler configuration (may be null)
 * @param kubernetesConfig  Kubernetes scheduler configuration (may be null)
 * @param slurmConfig       Slurm scheduler configuration (may be null)
 * @param sharedDir         shared filesystem directory for artifacts (may be null)
 * @param snakegrinderPath  path to snakegrinder binary (auto-detected if null)
 */
public record LaunchConfig(
        String defaultScheduler,
        String defaultArtifactStore,
        RayConfig rayConfig,
        KubernetesConfig kubernetesConfig,
        SlurmConfig slurmConfig,
        Path sharedDir,
        Path snakegrinderPath
) {

    /** Default scheduler when none is configured */
    public static final String DEFAULT_SCHEDULER = "local";

    /** Default artifact store when none is configured */
    public static final String DEFAULT_ARTIFACT_STORE = "local";

    /** Config directory */
    public static final Path CONFIG_DIR = Path.of(
            System.getProperty("user.home"), ".config", "warpforge"
    );

    /** Config file name */
    public static final String CONFIG_FILE = "launch.json";

    public LaunchConfig {
        Objects.requireNonNull(defaultScheduler, "defaultScheduler cannot be null");
        Objects.requireNonNull(defaultArtifactStore, "defaultArtifactStore cannot be null");

        if (defaultScheduler.isBlank()) {
            throw new IllegalArgumentException("defaultScheduler cannot be blank");
        }
        if (defaultArtifactStore.isBlank()) {
            throw new IllegalArgumentException("defaultArtifactStore cannot be blank");
        }
    }

    /**
     * Returns the default configuration with local scheduler.
     */
    public static LaunchConfig defaults() {
        return new LaunchConfig(
                DEFAULT_SCHEDULER,
                DEFAULT_ARTIFACT_STORE,
                null,
                null,
                null,
                null,
                null
        );
    }

    /**
     * Returns the config file path.
     */
    public static Path configFile() {
        return CONFIG_DIR.resolve(CONFIG_FILE);
    }

    /**
     * Returns a new config with the specified scheduler.
     */
    public LaunchConfig withScheduler(String scheduler) {
        return new LaunchConfig(scheduler, defaultArtifactStore, rayConfig,
                kubernetesConfig, slurmConfig, sharedDir, snakegrinderPath);
    }

    /**
     * Returns a new config with the specified artifact store.
     */
    public LaunchConfig withArtifactStore(String store) {
        return new LaunchConfig(defaultScheduler, store, rayConfig,
                kubernetesConfig, slurmConfig, sharedDir, snakegrinderPath);
    }

    /**
     * Returns a new config with Ray configuration.
     */
    public LaunchConfig withRay(RayConfig ray) {
        return new LaunchConfig(defaultScheduler, defaultArtifactStore, ray,
                kubernetesConfig, slurmConfig, sharedDir, snakegrinderPath);
    }

    /**
     * Returns a new config with Kubernetes configuration.
     */
    public LaunchConfig withKubernetes(KubernetesConfig kubernetes) {
        return new LaunchConfig(defaultScheduler, defaultArtifactStore, rayConfig,
                kubernetes, slurmConfig, sharedDir, snakegrinderPath);
    }

    /**
     * Returns a new config with Slurm configuration.
     */
    public LaunchConfig withSlurm(SlurmConfig slurm) {
        return new LaunchConfig(defaultScheduler, defaultArtifactStore, rayConfig,
                kubernetesConfig, slurm, sharedDir, snakegrinderPath);
    }

    /**
     * Returns a new config with shared directory.
     */
    public LaunchConfig withSharedDir(Path shared) {
        return new LaunchConfig(defaultScheduler, defaultArtifactStore, rayConfig,
                kubernetesConfig, slurmConfig, shared, snakegrinderPath);
    }

    /**
     * Returns a new config with snakegrinder path.
     */
    public LaunchConfig withSnakegrinder(Path snakegrinder) {
        return new LaunchConfig(defaultScheduler, defaultArtifactStore, rayConfig,
                kubernetesConfig, slurmConfig, sharedDir, snakegrinder);
    }
}
