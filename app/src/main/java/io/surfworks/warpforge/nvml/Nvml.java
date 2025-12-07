package io.surfworks.warpforge.nvml;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Entry point for getting an NvmlApi instance.
 *
 * Strategy:
 *  - If OS is Linux AND libnvidia-ml.so.1 exists AND no override,
 *    try NvmlFFMImpl.
 *  - If that fails for any reason, fall back to NvmlMock.
 *
 * You can force mock via:
 *  - System property: -Dwarpforge.nvml.mode=mock
 *  - Or if libnvidia-ml.so.1 is missing / unusable
 */
public final class Nvml {

    private static final String MODE_PROP = "warpforge.nvml.mode";

    private Nvml() {}

    public static NvmlApi load() {
        String forced = System.getProperty(MODE_PROP, "").trim().toLowerCase();
        if (forced.equals("mock")) {
            return new NvmlMock();
        }

        String os = System.getProperty("os.name", "").toLowerCase();
        boolean linux = os.contains("linux");

        if (linux && nvmlLibExists()) {
            try {
                return new NvmlFFMImpl();
            } catch (RuntimeException e) {
                System.err.println("[Nvml] Failed to init NVML FFM backend, falling back to mock: " + e);
                return new NvmlMock();
            }
        }

        // Non-Linux or missing NVML library
        return new NvmlMock();
    }

    private static boolean nvmlLibExists() {
        Path p = Path.of("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1");
        return Files.isReadable(p);
    }
}

