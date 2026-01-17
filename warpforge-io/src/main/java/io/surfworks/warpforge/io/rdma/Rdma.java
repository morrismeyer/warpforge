package io.surfworks.warpforge.io.rdma;

import io.surfworks.warpforge.io.rdma.impl.UcxRdmaImpl;
import io.surfworks.warpforge.io.rdma.mock.RdmaMock;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Factory for creating RDMA API instances.
 *
 * <p>This class automatically detects the appropriate RDMA implementation
 * based on the platform and available hardware. On Linux systems with
 * RDMA-capable NICs, the UCX-backed implementation is used. On other
 * systems or when RDMA is unavailable, a mock implementation is provided.
 *
 * <h2>Implementation Selection</h2>
 * <ol>
 *   <li>If {@code -Dwarpforge.rdma.mode=mock} is set, use mock</li>
 *   <li>If not on Linux, use mock</li>
 *   <li>If no InfiniBand devices in /sys/class/infiniband, use mock</li>
 *   <li>If UCX libraries not found, use mock</li>
 *   <li>Otherwise, use UCX implementation</li>
 * </ol>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * try (RdmaApi rdma = Rdma.load()) {
 *     System.out.println("Using backend: " + rdma.backendName());
 *     // ... use RDMA API
 * }
 * }</pre>
 */
public final class Rdma {

    private static final String MODE_PROPERTY = "warpforge.rdma.mode";
    private static final String UCX_LIB_PATH_PROPERTY = "warpforge.ucx.lib.path";

    private Rdma() {} // Utility class

    /**
     * Loads the appropriate RDMA implementation with default configuration.
     *
     * @return RDMA API instance
     */
    public static RdmaApi load() {
        return load(RdmaConfig.DEFAULT);
    }

    /**
     * Loads the appropriate RDMA implementation with the specified configuration.
     *
     * @param config RDMA configuration
     * @return RDMA API instance
     */
    public static RdmaApi load(RdmaConfig config) {
        String mode = System.getProperty(MODE_PROPERTY);

        if ("mock".equalsIgnoreCase(mode)) {
            return new RdmaMock(config);
        }

        if (canUseRealRdma()) {
            try {
                return new UcxRdmaImpl(config);
            } catch (Exception e) {
                System.err.println("Failed to initialize UCX RDMA, falling back to mock: " + e.getMessage());
                return new RdmaMock(config);
            }
        }

        return new RdmaMock(config);
    }

    /**
     * Loads the mock implementation regardless of platform capabilities.
     *
     * <p>Useful for testing on systems without RDMA hardware.
     *
     * @return mock RDMA API instance
     */
    public static RdmaApi loadMock() {
        return new RdmaMock(RdmaConfig.DEFAULT);
    }

    /**
     * Loads the mock implementation with configuration.
     *
     * @param config RDMA configuration
     * @return mock RDMA API instance
     */
    public static RdmaApi loadMock(RdmaConfig config) {
        return new RdmaMock(config);
    }

    /**
     * Attempts to load the real UCX implementation.
     *
     * <p>Unlike {@link #load()}, this method throws an exception if the
     * real implementation cannot be loaded.
     *
     * @return UCX RDMA API instance
     * @throws RdmaException if UCX cannot be loaded
     */
    public static RdmaApi loadUcx() {
        return loadUcx(RdmaConfig.DEFAULT);
    }

    /**
     * Attempts to load the real UCX implementation with configuration.
     *
     * @param config RDMA configuration
     * @return UCX RDMA API instance
     * @throws RdmaException if UCX cannot be loaded
     */
    public static RdmaApi loadUcx(RdmaConfig config) {
        if (!isLinux()) {
            throw new RdmaException("UCX RDMA requires Linux", RdmaException.ErrorCode.NOT_SUPPORTED);
        }
        if (!hasInfiniBandDevices()) {
            throw RdmaException.noDevice();
        }
        return new UcxRdmaImpl(config);
    }

    /**
     * Checks if real RDMA can be used on this system.
     *
     * @return true if RDMA hardware and libraries are available
     */
    public static boolean canUseRealRdma() {
        return isLinux() && hasInfiniBandDevices() && hasUcxLibraries();
    }

    /**
     * Returns whether the current platform is Linux.
     */
    public static boolean isLinux() {
        String os = System.getProperty("os.name", "").toLowerCase();
        return os.contains("linux");
    }

    /**
     * Returns whether InfiniBand devices are present.
     */
    public static boolean hasInfiniBandDevices() {
        try {
            Path ibPath = Path.of("/sys/class/infiniband");
            if (!Files.exists(ibPath)) {
                return false;
            }
            // Check if there's at least one device
            try (var stream = Files.list(ibPath)) {
                return stream.findAny().isPresent();
            }
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Returns whether UCX libraries are available.
     */
    public static boolean hasUcxLibraries() {
        // Check custom path first
        String customPath = System.getProperty(UCX_LIB_PATH_PROPERTY);
        if (customPath != null) {
            File lib = new File(customPath, "libucp.so");
            return lib.exists();
        }

        // Check LD_LIBRARY_PATH
        String ldPath = System.getenv("LD_LIBRARY_PATH");
        if (ldPath != null) {
            for (String path : ldPath.split(":")) {
                File lib = new File(path, "libucp.so");
                if (lib.exists()) {
                    return true;
                }
            }
        }

        // Check standard paths
        String[] standardPaths = {
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/lib",
            "/opt/ucx/lib",
            "../openucx/install/lib"  // Relative to project root
        };

        for (String path : standardPaths) {
            File lib = new File(path, "libucp.so");
            if (lib.exists()) {
                return true;
            }
        }

        return false;
    }

    /**
     * Returns a string describing the RDMA capabilities of this system.
     */
    public static String systemInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("RDMA System Information:\n");
        sb.append("  Platform: ").append(System.getProperty("os.name")).append("\n");
        sb.append("  Linux: ").append(isLinux()).append("\n");
        sb.append("  InfiniBand devices: ").append(hasInfiniBandDevices()).append("\n");
        sb.append("  UCX libraries: ").append(hasUcxLibraries()).append("\n");
        sb.append("  Can use real RDMA: ").append(canUseRealRdma()).append("\n");

        if (hasInfiniBandDevices()) {
            sb.append("  Devices:\n");
            try {
                Path ibPath = Path.of("/sys/class/infiniband");
                Files.list(ibPath).forEach(device -> {
                    sb.append("    - ").append(device.getFileName()).append("\n");
                });
            } catch (Exception e) {
                sb.append("    (error listing devices)\n");
            }
        }

        return sb.toString();
    }
}
