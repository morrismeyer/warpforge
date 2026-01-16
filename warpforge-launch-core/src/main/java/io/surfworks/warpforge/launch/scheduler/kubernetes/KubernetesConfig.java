package io.surfworks.warpforge.launch.scheduler.kubernetes;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Configuration for the Kubernetes scheduler.
 *
 * @param namespace       Kubernetes namespace for jobs
 * @param warpforgeImage  Container image with WarpForge tools
 * @param kubeConfigPath  Path to kubeconfig (null = use default)
 */
public record KubernetesConfig(
        String namespace,
        String warpforgeImage,
        Path kubeConfigPath
) {

    /** Default namespace */
    public static final String DEFAULT_NAMESPACE = "warpforge";

    /** Default container image */
    public static final String DEFAULT_IMAGE = "surfworks/warpforge:latest";

    public KubernetesConfig {
        Objects.requireNonNull(namespace, "namespace cannot be null");
        Objects.requireNonNull(warpforgeImage, "warpforgeImage cannot be null");

        if (namespace.isBlank()) {
            throw new IllegalArgumentException("namespace cannot be blank");
        }
        if (warpforgeImage.isBlank()) {
            throw new IllegalArgumentException("warpforgeImage cannot be blank");
        }
    }

    /**
     * Creates a config with default settings.
     */
    public static KubernetesConfig defaults() {
        return new KubernetesConfig(DEFAULT_NAMESPACE, DEFAULT_IMAGE, null);
    }

    /**
     * Creates a config with a custom namespace.
     */
    public static KubernetesConfig withNamespace(String namespace) {
        return new KubernetesConfig(namespace, DEFAULT_IMAGE, null);
    }

    /**
     * Returns a new config with updated namespace.
     */
    public KubernetesConfig namespace(String namespace) {
        return new KubernetesConfig(namespace, warpforgeImage, kubeConfigPath);
    }

    /**
     * Returns a new config with updated image.
     */
    public KubernetesConfig image(String image) {
        return new KubernetesConfig(namespace, image, kubeConfigPath);
    }

    /**
     * Returns a new config with specified kubeconfig path.
     */
    public KubernetesConfig kubeConfig(Path path) {
        return new KubernetesConfig(namespace, warpforgeImage, path);
    }
}
