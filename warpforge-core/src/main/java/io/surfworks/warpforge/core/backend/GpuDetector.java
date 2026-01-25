package io.surfworks.warpforge.core.backend;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Detects available GPU hardware and manages backend installation.
 *
 * <p>This class provides automatic GPU detection for NVIDIA (CUDA) and AMD (ROCm) GPUs.
 * It can also check if the corresponding backend libraries are installed and trigger
 * on-demand download if needed.</p>
 *
 * <p>Detection methods:</p>
 * <ul>
 *   <li>NVIDIA: Check for /dev/nvidia0 or nvidia-smi command</li>
 *   <li>AMD: Check for /dev/kfd or rocm-smi command</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * var gpus = GpuDetector.detectAll();
 * for (GpuInfo gpu : gpus) {
 *     System.out.println(gpu.vendor() + " GPU: " + gpu.name());
 *     if (!gpu.isBackendInstalled()) {
 *         GpuDetector.ensureBackendInstalled(gpu.vendor());
 *     }
 * }
 * }</pre>
 */
public final class GpuDetector {

    private GpuDetector() {} // Utility class

    /**
     * GPU vendor types.
     */
    public enum GpuVendor {
        NVIDIA("cuda"),
        AMD("rocm");

        private final String backendName;

        GpuVendor(String backendName) {
            this.backendName = backendName;
        }

        /**
         * Returns the backend name used for this vendor.
         */
        public String backendName() {
            return backendName;
        }
    }

    /**
     * Information about a detected GPU.
     */
    public record GpuInfo(
            GpuVendor vendor,
            int deviceIndex,
            String name,
            long totalMemoryBytes,
            String driverVersion
    ) {
        /**
         * Check if the backend libraries for this GPU are installed.
         */
        public boolean isBackendInstalled() {
            return GpuDetector.isBackendInstalled(vendor);
        }
    }

    /**
     * Detect all available GPUs in the system.
     *
     * @return List of detected GPUs (may be empty)
     */
    public static List<GpuInfo> detectAll() {
        List<GpuInfo> gpus = new ArrayList<>();
        gpus.addAll(detectNvidia());
        gpus.addAll(detectAmd());
        return gpus;
    }

    /**
     * Detect the primary GPU (first available).
     *
     * @return The first detected GPU, or empty if none found
     */
    public static Optional<GpuInfo> detectPrimary() {
        // Prefer NVIDIA if both are available (more common in ML workloads)
        List<GpuInfo> nvidia = detectNvidia();
        if (!nvidia.isEmpty()) {
            return Optional.of(nvidia.getFirst());
        }

        List<GpuInfo> amd = detectAmd();
        if (!amd.isEmpty()) {
            return Optional.of(amd.getFirst());
        }

        return Optional.empty();
    }

    /**
     * Detect NVIDIA GPUs using nvidia-smi.
     *
     * @return List of detected NVIDIA GPUs
     */
    public static List<GpuInfo> detectNvidia() {
        List<GpuInfo> gpus = new ArrayList<>();

        // Quick check for device files first
        if (!Files.exists(Path.of("/dev/nvidia0")) && !commandExists("nvidia-smi")) {
            return gpus;
        }

        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"
            );
            pb.redirectErrorStream(true);

            Process process = pb.start();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {

                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split(",\\s*");
                    if (parts.length >= 4) {
                        int index = Integer.parseInt(parts[0].trim());
                        String name = parts[1].trim();
                        // Memory is in MiB from nvidia-smi
                        long memoryBytes = Long.parseLong(parts[2].trim()) * 1024 * 1024;
                        String driverVersion = parts[3].trim();

                        gpus.add(new GpuInfo(GpuVendor.NVIDIA, index, name, memoryBytes, driverVersion));
                    }
                }
            }

            process.waitFor();
        } catch (IOException | InterruptedException | NumberFormatException e) {
            // nvidia-smi failed or not available
        }

        return gpus;
    }

    /**
     * Detect AMD GPUs using rocm-smi.
     *
     * @return List of detected AMD GPUs
     */
    public static List<GpuInfo> detectAmd() {
        List<GpuInfo> gpus = new ArrayList<>();

        // Quick check for device files first
        if (!Files.exists(Path.of("/dev/kfd")) && !commandExists("rocm-smi")) {
            return gpus;
        }

        try {
            ProcessBuilder pb = new ProcessBuilder("rocm-smi", "--showproductname", "--showmeminfo", "vram");
            pb.redirectErrorStream(true);

            Process process = pb.start();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {

                // rocm-smi output is more complex, parse it line by line
                String currentGpu = null;
                int index = 0;
                long memory = 0;

                String line;
                while ((line = reader.readLine()) != null) {
                    // Look for GPU entries
                    if (line.contains("GPU[")) {
                        // Extract GPU info - format varies by rocm-smi version
                        if (line.contains("Card series")) {
                            currentGpu = line.split(":")[1].trim();
                        }
                    }
                    // Look for VRAM Total
                    if (line.contains("VRAM Total Memory")) {
                        String memStr = line.replaceAll("[^0-9]", "");
                        if (!memStr.isEmpty()) {
                            memory = Long.parseLong(memStr) * 1024 * 1024; // Assume MiB
                        }
                    }
                }

                // Fallback: check if any AMD GPU device exists
                if (currentGpu == null && Files.exists(Path.of("/dev/kfd"))) {
                    currentGpu = "AMD GPU";
                    memory = 0; // Unknown
                }

                if (currentGpu != null) {
                    // Get driver version
                    String driverVersion = getAmdDriverVersion();
                    gpus.add(new GpuInfo(GpuVendor.AMD, index, currentGpu, memory, driverVersion));
                }
            }

            process.waitFor();
        } catch (IOException | InterruptedException | NumberFormatException e) {
            // rocm-smi failed, try fallback detection
            if (Files.exists(Path.of("/dev/kfd"))) {
                gpus.add(new GpuInfo(GpuVendor.AMD, 0, "AMD GPU", 0, "unknown"));
            }
        }

        return gpus;
    }

    /**
     * Check if a backend is installed.
     *
     * @param vendor The GPU vendor
     * @return true if backend libraries are installed
     */
    public static boolean isBackendInstalled(GpuVendor vendor) {
        Path backendDir = getBackendPath(vendor);
        if (!Files.exists(backendDir)) {
            return false;
        }

        // Check for at least one .so file
        try (var stream = Files.list(backendDir)) {
            return stream.anyMatch(p -> p.toString().endsWith(".so") ||
                    p.toString().contains(".so."));
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Get the path where backend libraries should be installed.
     *
     * @param vendor The GPU vendor
     * @return Path to the backend library directory
     */
    public static Path getBackendPath(GpuVendor vendor) {
        return getWarpforgeHome().resolve("lib/backends").resolve(vendor.backendName());
    }

    /**
     * Ensure the backend for a GPU vendor is installed.
     * If not installed, this will attempt to download it.
     *
     * @param vendor The GPU vendor
     * @throws BackendNotAvailableException if the backend cannot be installed
     */
    public static void ensureBackendInstalled(GpuVendor vendor) {
        if (isBackendInstalled(vendor)) {
            return;
        }

        Path downloadScript = getWarpforgeHome().resolve("bin/download-backend.sh");
        if (!Files.exists(downloadScript)) {
            // Try the scripts directory as fallback
            downloadScript = getWarpforgeHome().resolve("scripts/download-backend.sh");
        }

        if (!Files.exists(downloadScript)) {
            throw new BackendNotAvailableException(
                    "Backend '" + vendor.backendName() + "' is not installed and download script not found. " +
                            "Please download manually from https://github.com/surfworks/warpforge-backends");
        }

        try {
            System.out.println("Downloading " + vendor.backendName() + " backend...");
            ProcessBuilder pb = new ProcessBuilder("bash", downloadScript.toString(), vendor.backendName());
            pb.inheritIO();
            Process process = pb.start();
            int exitCode = process.waitFor();

            if (exitCode != 0) {
                throw new BackendNotAvailableException(
                        "Failed to download " + vendor.backendName() + " backend (exit code " + exitCode + ")");
            }
        } catch (IOException | InterruptedException e) {
            throw new BackendNotAvailableException(
                    "Failed to download " + vendor.backendName() + " backend: " + e.getMessage(), e);
        }
    }

    /**
     * Get the WarpForge home directory.
     * This is where WarpForge is installed (parent of bin/).
     *
     * @return Path to WarpForge home directory
     */
    public static Path getWarpforgeHome() {
        // Check WARPFORGE_HOME environment variable first
        String envHome = System.getenv("WARPFORGE_HOME");
        if (envHome != null && !envHome.isEmpty()) {
            return Path.of(envHome);
        }

        // Try to detect from the running JAR/class location
        try {
            Path classPath = Path.of(GpuDetector.class.getProtectionDomain()
                    .getCodeSource().getLocation().toURI());

            // If running from a JAR in lib/, go up to find the home
            if (classPath.toString().endsWith(".jar")) {
                Path parent = classPath.getParent();
                if (parent != null && parent.endsWith("lib")) {
                    return parent.getParent();
                }
            }
        } catch (Exception ignored) {
            // Fall through to default
        }

        // Default to ~/.warpforge
        return Path.of(System.getProperty("user.home"), ".warpforge");
    }

    /**
     * Print GPU information to stdout (for CLI tools).
     */
    public static void printGpuInfo() {
        List<GpuInfo> gpus = detectAll();

        if (gpus.isEmpty()) {
            System.out.println("No GPUs detected.");
            System.out.println();
            System.out.println("Supported GPUs:");
            System.out.println("  - NVIDIA GPUs (CUDA)");
            System.out.println("  - AMD GPUs (ROCm)");
            return;
        }

        System.out.println("Detected GPUs:");
        System.out.println("==============");

        for (GpuInfo gpu : gpus) {
            System.out.printf("%s GPU %d: %s%n", gpu.vendor(), gpu.deviceIndex(), gpu.name());
            System.out.printf("  Memory:  %d MB%n", gpu.totalMemoryBytes() / (1024 * 1024));
            System.out.printf("  Driver:  %s%n", gpu.driverVersion());
            System.out.printf("  Backend: %s (%s)%n",
                    gpu.vendor().backendName(),
                    gpu.isBackendInstalled() ? "installed" : "not installed");
            System.out.println();
        }
    }

    // ==================== Helper Methods ====================

    private static boolean commandExists(String command) {
        try {
            ProcessBuilder pb = new ProcessBuilder("which", command);
            Process process = pb.start();
            return process.waitFor() == 0;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }

    private static String getAmdDriverVersion() {
        try {
            ProcessBuilder pb = new ProcessBuilder("rocm-smi", "--showdriverversion");
            pb.redirectErrorStream(true);
            Process process = pb.start();

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.contains("Driver version")) {
                        return line.split(":")[1].trim();
                    }
                }
            }
            process.waitFor();
        } catch (IOException | InterruptedException e) {
            // Ignore
        }
        return "unknown";
    }

    /**
     * Exception thrown when a GPU backend cannot be made available.
     */
    public static class BackendNotAvailableException extends RuntimeException {
        public BackendNotAvailableException(String message) {
            super(message);
        }

        public BackendNotAvailableException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
