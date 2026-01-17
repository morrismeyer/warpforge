package io.surfworks.warpforge.io.collective;

import io.surfworks.warpforge.io.collective.impl.UccCollectiveImpl;
import io.surfworks.warpforge.io.collective.mock.CollectiveMock;
import io.surfworks.warpforge.io.rdma.Rdma;

/**
 * Factory for creating CollectiveApi instances.
 *
 * <p>This class automatically detects the appropriate collective implementation
 * based on the platform and available libraries. On Linux systems with UCC
 * libraries, the UCC-backed implementation is used. On other systems or when
 * UCC is unavailable, a mock implementation is provided.
 *
 * <h2>Implementation Selection</h2>
 * <ol>
 *   <li>If {@code -Dwarpforge.collective.mode=mock} is set, use mock</li>
 *   <li>If UCC libraries not found, use mock</li>
 *   <li>Otherwise, use UCC implementation</li>
 * </ol>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * CollectiveConfig config = CollectiveConfig.of(worldSize, rank, masterAddr, port);
 * try (CollectiveApi collective = Collective.load(config)) {
 *     System.out.println("Using backend: " + collective.backendName());
 *     // ... use collective operations
 * }
 * }</pre>
 */
public final class Collective {

    private static final String MODE_PROPERTY = "warpforge.collective.mode";
    private static final String UCC_LIB_PATH_PROPERTY = "warpforge.ucc.lib.path";

    private Collective() {} // Utility class

    /**
     * Loads the appropriate collective implementation.
     *
     * @param config collective configuration
     * @return CollectiveApi instance
     */
    public static CollectiveApi load(CollectiveConfig config) {
        String mode = System.getProperty(MODE_PROPERTY);

        if ("mock".equalsIgnoreCase(mode)) {
            return new CollectiveMock(config);
        }

        if (canUseRealUcc()) {
            try {
                return new UccCollectiveImpl(config);
            } catch (Exception e) {
                System.err.println("Failed to initialize UCC, falling back to mock: " + e.getMessage());
                return new CollectiveMock(config);
            }
        }

        return new CollectiveMock(config);
    }

    /**
     * Loads the mock implementation regardless of platform capabilities.
     *
     * <p>Useful for testing on systems without UCC libraries.
     *
     * @param config collective configuration
     * @return mock CollectiveApi instance
     */
    public static CollectiveApi loadMock(CollectiveConfig config) {
        return new CollectiveMock(config);
    }

    /**
     * Loads the mock implementation with default configuration for testing.
     *
     * @param worldSize total number of ranks
     * @param rank this process's rank
     * @return mock CollectiveApi instance
     */
    public static CollectiveApi loadMock(int worldSize, int rank) {
        return new CollectiveMock(CollectiveConfig.of(worldSize, rank));
    }

    /**
     * Attempts to load the real UCC implementation.
     *
     * <p>Unlike {@link #load(CollectiveConfig)}, this method throws an exception
     * if the real implementation cannot be loaded.
     *
     * @param config collective configuration
     * @return UCC CollectiveApi instance
     * @throws CollectiveException if UCC cannot be loaded
     */
    public static CollectiveApi loadUcc(CollectiveConfig config) {
        if (!hasUccLibraries()) {
            throw new CollectiveException("UCC libraries not found", CollectiveException.ErrorCode.NOT_SUPPORTED);
        }
        return new UccCollectiveImpl(config);
    }

    /**
     * Checks if real UCC can be used on this system.
     *
     * @return true if UCC libraries are available
     */
    public static boolean canUseRealUcc() {
        return Rdma.isLinux() && hasUccLibraries();
    }

    /**
     * Returns whether UCC libraries are available.
     */
    public static boolean hasUccLibraries() {
        // Check custom path first
        String customPath = System.getProperty(UCC_LIB_PATH_PROPERTY);
        if (customPath != null) {
            java.io.File lib = new java.io.File(customPath, "libucc.so");
            return lib.exists();
        }

        // Check LD_LIBRARY_PATH
        String ldPath = System.getenv("LD_LIBRARY_PATH");
        if (ldPath != null) {
            for (String path : ldPath.split(":")) {
                java.io.File lib = new java.io.File(path, "libucc.so");
                if (lib.exists()) {
                    return true;
                }
            }
        }

        // Check standard paths
        String[] standardPaths = {
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/lib",
            "/opt/ucc/lib",
            "../ucc/install/lib"  // Relative to project root
        };

        for (String path : standardPaths) {
            java.io.File lib = new java.io.File(path, "libucc.so");
            if (lib.exists()) {
                return true;
            }
        }

        return false;
    }

    /**
     * Returns a string describing the collective capabilities of this system.
     */
    public static String systemInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("Collective System Information:\n");
        sb.append("  Platform: ").append(System.getProperty("os.name")).append("\n");
        sb.append("  Linux: ").append(Rdma.isLinux()).append("\n");
        sb.append("  UCC libraries: ").append(hasUccLibraries()).append("\n");
        sb.append("  RDMA available: ").append(Rdma.canUseRealRdma()).append("\n");
        sb.append("  Can use real UCC: ").append(canUseRealUcc()).append("\n");
        return sb.toString();
    }
}
