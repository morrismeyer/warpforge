package io.surfworks.warpforge.launch.config;

import io.surfworks.warpforge.launch.scheduler.kubernetes.KubernetesConfig;
import io.surfworks.warpforge.launch.scheduler.ray.RayConfig;
import io.surfworks.warpforge.launch.scheduler.slurm.SlurmConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for LaunchConfig and LaunchConfigLoader.
 */
class LaunchConfigTest {

    @TempDir
    Path tempDir;

    // ===== LaunchConfig tests =====

    @Test
    void defaultsReturnsValidConfig() {
        LaunchConfig config = LaunchConfig.defaults();

        assertEquals("local", config.defaultScheduler());
        assertEquals("local", config.defaultArtifactStore());
        assertNull(config.rayConfig());
        assertNull(config.kubernetesConfig());
        assertNull(config.slurmConfig());
    }

    @Test
    void withSchedulerCreatesNewInstance() {
        LaunchConfig base = LaunchConfig.defaults();
        LaunchConfig modified = base.withScheduler("ray");

        assertEquals("local", base.defaultScheduler());
        assertEquals("ray", modified.defaultScheduler());
    }

    @Test
    void withArtifactStoreCreatesNewInstance() {
        LaunchConfig base = LaunchConfig.defaults();
        LaunchConfig modified = base.withArtifactStore("shared");

        assertEquals("local", base.defaultArtifactStore());
        assertEquals("shared", modified.defaultArtifactStore());
    }

    @Test
    void withRayCreatesNewInstance() {
        LaunchConfig base = LaunchConfig.defaults();
        RayConfig rayConfig = RayConfig.of("http://ray:8265");
        LaunchConfig modified = base.withRay(rayConfig);

        assertNull(base.rayConfig());
        assertEquals(rayConfig, modified.rayConfig());
    }

    @Test
    void withKubernetesCreatesNewInstance() {
        LaunchConfig base = LaunchConfig.defaults();
        KubernetesConfig k8sConfig = KubernetesConfig.defaults();
        LaunchConfig modified = base.withKubernetes(k8sConfig);

        assertNull(base.kubernetesConfig());
        assertEquals(k8sConfig, modified.kubernetesConfig());
    }

    @Test
    void withSlurmCreatesNewInstance() {
        LaunchConfig base = LaunchConfig.defaults();
        SlurmConfig slurmConfig = SlurmConfig.of("slurm-head", "user");
        LaunchConfig modified = base.withSlurm(slurmConfig);

        assertNull(base.slurmConfig());
        assertEquals(slurmConfig, modified.slurmConfig());
    }

    @Test
    void nullSchedulerThrows() {
        assertThrows(NullPointerException.class, () ->
                new LaunchConfig(null, "local", null, null, null, null, null));
    }

    @Test
    void blankSchedulerThrows() {
        assertThrows(IllegalArgumentException.class, () ->
                new LaunchConfig("  ", "local", null, null, null, null, null));
    }

    // ===== LaunchConfigLoader tests =====

    @Test
    void loadReturnsDefaultsWhenNoConfigFile() {
        Path nonexistent = tempDir.resolve("nonexistent.json");
        LaunchConfig config = LaunchConfigLoader.load(nonexistent);

        assertEquals("local", config.defaultScheduler());
    }

    @Test
    void saveAndLoad() throws IOException {
        Path configFile = tempDir.resolve("launch.json");

        LaunchConfig original = LaunchConfig.defaults()
                .withScheduler("ray")
                .withRay(RayConfig.of("http://ray-cluster:8265"));

        LaunchConfigLoader.save(original, configFile);
        assertTrue(Files.exists(configFile));

        LaunchConfig loaded = LaunchConfigLoader.load(configFile);

        assertEquals("ray", loaded.defaultScheduler());
        assertNotNull(loaded.rayConfig());
        assertEquals("http://ray-cluster:8265", loaded.rayConfig().dashboardUrl());
    }

    @Test
    void loadParsesKubernetesConfig() throws IOException {
        Path configFile = tempDir.resolve("launch.json");
        String json = """
                {
                  "defaultScheduler": "kubernetes",
                  "defaultArtifactStore": "shared",
                  "kubernetes": {
                    "namespace": "ml-jobs",
                    "image": "surfworks/warpforge:1.0"
                  }
                }
                """;
        Files.writeString(configFile, json);

        LaunchConfig config = LaunchConfigLoader.load(configFile);

        assertEquals("kubernetes", config.defaultScheduler());
        assertEquals("shared", config.defaultArtifactStore());
        assertNotNull(config.kubernetesConfig());
        assertEquals("ml-jobs", config.kubernetesConfig().namespace());
        assertEquals("surfworks/warpforge:1.0", config.kubernetesConfig().warpforgeImage());
    }

    @Test
    void loadParsesSlurmConfig() throws IOException {
        Path configFile = tempDir.resolve("launch.json");
        String json = """
                {
                  "defaultScheduler": "slurm",
                  "defaultArtifactStore": "local",
                  "slurm": {
                    "host": "slurm-head.local",
                    "user": "researcher",
                    "partition": "gpu",
                    "workDir": "/scratch/warpforge"
                  }
                }
                """;
        Files.writeString(configFile, json);

        LaunchConfig config = LaunchConfigLoader.load(configFile);

        assertEquals("slurm", config.defaultScheduler());
        assertNotNull(config.slurmConfig());
        assertEquals("slurm-head.local", config.slurmConfig().sshHost());
        assertEquals("researcher", config.slurmConfig().sshUser());
        assertEquals("gpu", config.slurmConfig().partition());
        assertEquals(Path.of("/scratch/warpforge"), config.slurmConfig().remoteWorkDir());
    }

    @Test
    void loadParsesSharedDir() throws IOException {
        Path configFile = tempDir.resolve("launch.json");
        String json = """
                {
                  "defaultScheduler": "local",
                  "defaultArtifactStore": "shared",
                  "sharedDir": "/nfs/warpforge"
                }
                """;
        Files.writeString(configFile, json);

        LaunchConfig config = LaunchConfigLoader.load(configFile);

        assertEquals(Path.of("/nfs/warpforge"), config.sharedDir());
    }

    @Test
    void loadParsesSnakegrinderPath() throws IOException {
        Path configFile = tempDir.resolve("launch.json");
        String json = """
                {
                  "defaultScheduler": "local",
                  "defaultArtifactStore": "local",
                  "snakegrinderPath": "/opt/warpforge/bin/snakegrinder"
                }
                """;
        Files.writeString(configFile, json);

        LaunchConfig config = LaunchConfigLoader.load(configFile);

        assertEquals(Path.of("/opt/warpforge/bin/snakegrinder"), config.snakegrinderPath());
    }

    @Test
    void loadHandlesMalformedJson() throws IOException {
        Path configFile = tempDir.resolve("launch.json");
        Files.writeString(configFile, "{ malformed json }");

        // Should not throw, returns defaults
        LaunchConfig config = LaunchConfigLoader.load(configFile);
        assertEquals("local", config.defaultScheduler());
    }

    @Test
    void saveCreatesParentDirectories() throws IOException {
        Path configFile = tempDir.resolve("subdir/nested/launch.json");

        LaunchConfig config = LaunchConfig.defaults();
        LaunchConfigLoader.save(config, configFile);

        assertTrue(Files.exists(configFile));
    }
}
