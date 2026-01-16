package io.surfworks.warpforge.launch.scheduler.slurm;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Configuration for the Slurm scheduler.
 *
 * @param sshHost                 SSH hostname for Slurm head node
 * @param sshUser                 SSH username
 * @param sshKeyPath              Path to SSH private key (null = use default)
 * @param partition               Slurm partition/queue name (null = default)
 * @param remoteWorkDir           Remote directory for job scripts and output
 * @param sshConnectTimeoutSeconds SSH connection timeout in seconds
 */
public record SlurmConfig(
        String sshHost,
        String sshUser,
        Path sshKeyPath,
        String partition,
        Path remoteWorkDir,
        int sshConnectTimeoutSeconds
) {

    /** Default SSH connection timeout */
    public static final int DEFAULT_SSH_TIMEOUT = 10;

    /** Default remote work directory */
    public static final Path DEFAULT_WORK_DIR = Path.of("/tmp/warpforge-jobs");

    public SlurmConfig {
        Objects.requireNonNull(sshHost, "sshHost cannot be null");
        Objects.requireNonNull(sshUser, "sshUser cannot be null");
        Objects.requireNonNull(remoteWorkDir, "remoteWorkDir cannot be null");

        if (sshHost.isBlank()) {
            throw new IllegalArgumentException("sshHost cannot be blank");
        }
        if (sshUser.isBlank()) {
            throw new IllegalArgumentException("sshUser cannot be blank");
        }
        if (sshConnectTimeoutSeconds <= 0) {
            throw new IllegalArgumentException("sshConnectTimeoutSeconds must be positive");
        }
    }

    /**
     * Creates a config for connecting to a Slurm cluster.
     */
    public static SlurmConfig of(String host, String user) {
        return new SlurmConfig(host, user, null, null, DEFAULT_WORK_DIR, DEFAULT_SSH_TIMEOUT);
    }

    /**
     * Creates a config with SSH key authentication.
     */
    public static SlurmConfig withKey(String host, String user, Path keyPath) {
        return new SlurmConfig(host, user, keyPath, null, DEFAULT_WORK_DIR, DEFAULT_SSH_TIMEOUT);
    }

    /**
     * Returns a new config with specified partition.
     */
    public SlurmConfig partition(String partition) {
        return new SlurmConfig(sshHost, sshUser, sshKeyPath, partition, remoteWorkDir, sshConnectTimeoutSeconds);
    }

    /**
     * Returns a new config with specified work directory.
     */
    public SlurmConfig workDir(Path workDir) {
        return new SlurmConfig(sshHost, sshUser, sshKeyPath, partition, workDir, sshConnectTimeoutSeconds);
    }

    /**
     * Returns a new config with specified SSH timeout.
     */
    public SlurmConfig timeout(int seconds) {
        return new SlurmConfig(sshHost, sshUser, sshKeyPath, partition, remoteWorkDir, seconds);
    }

    /**
     * Returns the SSH connection string (user@host).
     */
    public String sshTarget() {
        return sshUser + "@" + sshHost;
    }
}
