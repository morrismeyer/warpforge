package io.surfworks.warpforge.nvml;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class NvmlApiSmokeTest {

    @Test
    void nvmlLoadsAndReturnsSaneValues() {
        NvmlApi nvml = Nvml.load();
        assertNotNull(nvml, "Nvml.load() should never return null");

        String backend = nvml.backendName();
        assertNotNull(backend);
        assertFalse(backend.isBlank(), "backendName should not be blank");

        String version = nvml.driverVersion();
        assertNotNull(version);
        assertFalse(version.isBlank(), "driverVersion should not be blank");

        int count = nvml.deviceCount();
        assertTrue(count >= 0, "deviceCount should be >= 0");

        System.out.println("NVML backend: " + backend
                + ", driverVersion=" + version
                + ", deviceCount=" + count);
    }
}

