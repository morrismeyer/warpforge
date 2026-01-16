package io.surfworks.warpforge.launch.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.surfworks.warpforge.launch.scheduler.kubernetes.KubernetesConfig;
import io.surfworks.warpforge.launch.scheduler.ray.RayConfig;
import io.surfworks.warpforge.launch.scheduler.slurm.SlurmConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

/**
 * Loads and saves LaunchConfig.
 *
 * <p>Configuration sources (in order of precedence):
 * <ol>
 *   <li>Config file ({@code ~/.config/warpforge/launch.json})</li>
 *   <li>Auto-detection (snakegrinder path, shared directory)</li>
 *   <li>Defaults</li>
 * </ol>
 *
 * <p>CLI arguments are handled by the caller and merged into the config.
 */
public final class LaunchConfigLoader {

    private static final ObjectMapper JSON = new ObjectMapper();

    private LaunchConfigLoader() {
    }

    /**
     * Loads configuration from the default config file.
     *
     * <p>If the config file doesn't exist, returns defaults with auto-detection.
     *
     * @return the loaded configuration
     */
    public static LaunchConfig load() {
        return load(LaunchConfig.configFile());
    }

    /**
     * Loads configuration from a specific file.
     *
     * @param configFile path to the config file
     * @return the loaded configuration
     */
    public static LaunchConfig load(Path configFile) {
        LaunchConfig config = LaunchConfig.defaults();

        // Load from config file if it exists
        if (Files.exists(configFile)) {
            config = loadFromFile(configFile, config);
        }

        // Apply auto-detection for missing values
        config = applyAutoDetection(config);

        return config;
    }

    /**
     * Saves configuration to the default config file.
     *
     * @param config the configuration to save
     * @throws IOException if saving fails
     */
    public static void save(LaunchConfig config) throws IOException {
        save(config, LaunchConfig.configFile());
    }

    /**
     * Saves configuration to a specific file.
     *
     * @param config     the configuration to save
     * @param configFile path to write the config
     * @throws IOException if saving fails
     */
    public static void save(LaunchConfig config, Path configFile) throws IOException {
        Files.createDirectories(configFile.getParent());

        ObjectNode root = JSON.createObjectNode();
        root.put("defaultScheduler", config.defaultScheduler());
        root.put("defaultArtifactStore", config.defaultArtifactStore());

        if (config.rayConfig() != null) {
            ObjectNode ray = root.putObject("ray");
            ray.put("dashboardUrl", config.rayConfig().dashboardUrl());
        }

        if (config.kubernetesConfig() != null) {
            ObjectNode k8s = root.putObject("kubernetes");
            k8s.put("namespace", config.kubernetesConfig().namespace());
            k8s.put("image", config.kubernetesConfig().warpforgeImage());
            if (config.kubernetesConfig().kubeConfigPath() != null) {
                k8s.put("kubeConfig", config.kubernetesConfig().kubeConfigPath().toString());
            }
        }

        if (config.slurmConfig() != null) {
            ObjectNode slurm = root.putObject("slurm");
            slurm.put("host", config.slurmConfig().sshHost());
            slurm.put("user", config.slurmConfig().sshUser());
            if (config.slurmConfig().sshKeyPath() != null) {
                slurm.put("keyPath", config.slurmConfig().sshKeyPath().toString());
            }
            if (config.slurmConfig().partition() != null) {
                slurm.put("partition", config.slurmConfig().partition());
            }
            slurm.put("workDir", config.slurmConfig().remoteWorkDir().toString());
        }

        if (config.sharedDir() != null) {
            root.put("sharedDir", config.sharedDir().toString());
        }

        if (config.snakegrinderPath() != null) {
            root.put("snakegrinderPath", config.snakegrinderPath().toString());
        }

        JSON.writerWithDefaultPrettyPrinter().writeValue(configFile.toFile(), root);
    }

    private static LaunchConfig loadFromFile(Path configFile, LaunchConfig base) {
        try {
            JsonNode root = JSON.readTree(configFile.toFile());

            String scheduler = getStringOrDefault(root, "defaultScheduler", base.defaultScheduler());
            String artifactStore = getStringOrDefault(root, "defaultArtifactStore", base.defaultArtifactStore());

            LaunchConfig config = base
                    .withScheduler(scheduler)
                    .withArtifactStore(artifactStore);

            // Parse Ray config
            if (root.has("ray")) {
                JsonNode rayNode = root.get("ray");
                String dashboardUrl = getStringOrDefault(rayNode, "dashboardUrl", RayConfig.DEFAULT_DASHBOARD_URL);
                config = config.withRay(RayConfig.of(dashboardUrl));
            }

            // Parse Kubernetes config
            if (root.has("kubernetes")) {
                JsonNode k8sNode = root.get("kubernetes");
                String namespace = getStringOrDefault(k8sNode, "namespace", KubernetesConfig.DEFAULT_NAMESPACE);
                String image = getStringOrDefault(k8sNode, "image", KubernetesConfig.DEFAULT_IMAGE);
                Path kubeConfig = k8sNode.has("kubeConfig") ? Path.of(k8sNode.get("kubeConfig").asText()) : null;

                config = config.withKubernetes(new KubernetesConfig(namespace, image, kubeConfig));
            }

            // Parse Slurm config
            if (root.has("slurm")) {
                JsonNode slurmNode = root.get("slurm");
                if (slurmNode.has("host") && slurmNode.has("user")) {
                    String host = slurmNode.get("host").asText();
                    String user = slurmNode.get("user").asText();
                    Path keyPath = slurmNode.has("keyPath") ? Path.of(slurmNode.get("keyPath").asText()) : null;
                    String partition = slurmNode.has("partition") ? slurmNode.get("partition").asText() : null;
                    Path workDir = slurmNode.has("workDir") ?
                            Path.of(slurmNode.get("workDir").asText()) : SlurmConfig.DEFAULT_WORK_DIR;

                    SlurmConfig slurmConfig = new SlurmConfig(host, user, keyPath, partition, workDir,
                            SlurmConfig.DEFAULT_SSH_TIMEOUT);
                    config = config.withSlurm(slurmConfig);
                }
            }

            // Parse shared directory
            if (root.has("sharedDir")) {
                config = config.withSharedDir(Path.of(root.get("sharedDir").asText()));
            }

            // Parse snakegrinder path
            if (root.has("snakegrinderPath")) {
                config = config.withSnakegrinder(Path.of(root.get("snakegrinderPath").asText()));
            }

            return config;

        } catch (IOException e) {
            // Return base config if parsing fails
            return base;
        }
    }

    private static LaunchConfig applyAutoDetection(LaunchConfig config) {
        // Auto-detect snakegrinder if not configured
        if (config.snakegrinderPath() == null) {
            Path detected = detectSnakegrinder();
            if (detected != null) {
                config = config.withSnakegrinder(detected);
            }
        }

        // Auto-detect shared directory if not configured
        if (config.sharedDir() == null) {
            Path detected = detectSharedDir();
            if (detected != null) {
                config = config.withSharedDir(detected);
            }
        }

        return config;
    }

    private static Path detectSnakegrinder() {
        // Check sibling directories first (following "It Just Works" philosophy)
        Path siblingPaths[] = {
                Path.of(System.getProperty("user.dir"), "..", "snakegrinder-dist", "build", "dist", "bin", "snakegrinder"),
                Path.of(System.getProperty("user.dir"), "snakegrinder-dist", "build", "dist", "bin", "snakegrinder"),
                Path.of(System.getProperty("user.home"), "surfworks", "warpforge", "snakegrinder-dist", "build", "dist", "bin", "snakegrinder")
        };

        for (Path path : siblingPaths) {
            Path normalized = path.normalize();
            if (Files.isExecutable(normalized)) {
                return normalized;
            }
        }

        // Check PATH via which command
        try {
            ProcessBuilder pb = new ProcessBuilder("which", "snakegrinder");
            pb.redirectErrorStream(true);
            Process p = pb.start();
            String output = new String(p.getInputStream().readAllBytes()).trim();
            int exitCode = p.waitFor();
            if (exitCode == 0 && !output.isBlank()) {
                return Path.of(output);
            }
        } catch (IOException | InterruptedException e) {
            // Ignore and return null
        }

        return null;
    }

    private static Path detectSharedDir() {
        // Check common shared filesystem locations
        Path[] candidates = {
                Path.of("/shared/warpforge"),
                Path.of("/mnt/shared/warpforge"),
                Path.of("/nfs/warpforge"),
                Path.of(System.getProperty("user.home"), "shared", "warpforge")
        };

        for (Path candidate : candidates) {
            if (Files.isDirectory(candidate)) {
                return candidate;
            }
        }

        return null;
    }

    private static String getStringOrDefault(JsonNode node, String field, String defaultValue) {
        if (node.has(field)) {
            return node.get(field).asText();
        }
        return defaultValue;
    }
}
