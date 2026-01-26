package io.surfworks.warpforge.core.concurrency;

import io.surfworks.warpforge.core.backend.GpuBackend;

import org.opentest4j.TestAbortedException;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * GPU test support utilities for dual-platform testing.
 *
 * <p>This class auto-detects the available GPU backend (NVIDIA or AMD) and provides
 * helpers for tests that run on both platforms. The same test code runs on both
 * NVIDIA and AMD machines - backend selection is automatic.
 *
 * <p>Usage in tests:
 * <pre>
 * {@code
 * @BeforeEach
 * void setUp() {
 *     backend = GpuTestSupport.createBackend();
 * }
 *
 * @AfterEach
 * void tearDown() {
 *     if (backend != null) {
 *         backend.close();
 *     }
 * }
 * }
 * </pre>
 */
public final class GpuTestSupport {

    private static final String NVIDIA_BACKEND_CLASS = "io.surfworks.warpforge.backend.nvidia.NvidiaBackend";
    private static final String AMD_BACKEND_CLASS = "io.surfworks.warpforge.backend.amd.AmdBackend";

    private GpuTestSupport() {
        // Utility class
    }

    /**
     * Auto-detects and creates the available GPU backend.
     *
     * <p>On NVIDIA machines: returns NvidiaBackend
     * <p>On AMD machines: returns AmdBackend
     * <p>If no GPU available: throws TestAbortedException to skip the test
     *
     * @return The created GPU backend
     * @throws TestAbortedException if no GPU backend is available
     */
    public static GpuBackend createBackend() {
        // Try NVIDIA first
        GpuBackend nvidia = tryCreateNvidiaBackend();
        if (nvidia != null) {
            return nvidia;
        }

        // Try AMD
        GpuBackend amd = tryCreateAmdBackend();
        if (amd != null) {
            return amd;
        }

        // No GPU available - skip the test
        throw new TestAbortedException("No GPU backend available - skipping test");
    }

    /**
     * Returns the expected backend name for JFR event assertions.
     *
     * @param backend The GPU backend
     * @return "CUDA" for NVIDIA, "HIP" for AMD
     */
    public static String expectedBackendName(GpuBackend backend) {
        String name = backend.name();
        if (name.toLowerCase().contains("nvidia") || name.toLowerCase().contains("cuda")) {
            return "CUDA";
        } else if (name.toLowerCase().contains("amd") || name.toLowerCase().contains("hip")) {
            return "HIP";
        }
        return name;
    }

    /**
     * Checks if the current machine has NVIDIA GPU available.
     *
     * @return true if NVIDIA/CUDA is available
     */
    public static boolean isNvidiaAvailable() {
        try {
            Class<?> nvidiaClass = Class.forName(NVIDIA_BACKEND_CLASS);
            Method isAvailable = nvidiaClass.getMethod("isCudaAvailable");
            return (boolean) isAvailable.invoke(null);
        } catch (ClassNotFoundException e) {
            return false;
        } catch (NoSuchMethodException e) {
            return false;
        } catch (IllegalAccessException e) {
            return false;
        } catch (InvocationTargetException e) {
            return false;
        }
    }

    /**
     * Checks if the current machine has AMD GPU available.
     *
     * @return true if AMD/HIP is available
     */
    public static boolean isAmdAvailable() {
        try {
            Class<?> amdClass = Class.forName(AMD_BACKEND_CLASS);
            Method isAvailable = amdClass.getMethod("isHipAvailable");
            return (boolean) isAvailable.invoke(null);
        } catch (ClassNotFoundException e) {
            return false;
        } catch (NoSuchMethodException e) {
            return false;
        } catch (IllegalAccessException e) {
            return false;
        } catch (InvocationTargetException e) {
            return false;
        }
    }

    /**
     * Returns a description of the current GPU environment.
     *
     * @return Description like "NVIDIA CUDA" or "AMD HIP" or "No GPU"
     */
    public static String describeEnvironment() {
        if (isNvidiaAvailable()) {
            return "NVIDIA CUDA";
        } else if (isAmdAvailable()) {
            return "AMD HIP";
        } else {
            return "No GPU available";
        }
    }

    private static GpuBackend tryCreateNvidiaBackend() {
        try {
            Class<?> nvidiaClass = Class.forName(NVIDIA_BACKEND_CLASS);

            // Check if CUDA is available
            Method isAvailable = nvidiaClass.getMethod("isCudaAvailable");
            boolean available = (boolean) isAvailable.invoke(null);
            if (!available) {
                return null;
            }

            // Create the backend
            return (GpuBackend) nvidiaClass.getConstructor().newInstance();
        } catch (ClassNotFoundException e) {
            // NVIDIA backend not on classpath
            return null;
        } catch (NoSuchMethodException e) {
            // API mismatch
            return null;
        } catch (IllegalAccessException e) {
            return null;
        } catch (InvocationTargetException e) {
            // CUDA init failed (no device, driver issues, etc.)
            Throwable cause = e.getCause();
            if (cause != null && cause.getMessage() != null) {
                String msg = cause.getMessage();
                if (msg.contains("no CUDA-capable device") || msg.contains("cuInit")) {
                    // Expected - no GPU
                    return null;
                }
            }
            return null;
        } catch (InstantiationException e) {
            return null;
        }
    }

    private static GpuBackend tryCreateAmdBackend() {
        try {
            Class<?> amdClass = Class.forName(AMD_BACKEND_CLASS);

            // Check if HIP is available
            Method isAvailable = amdClass.getMethod("isHipAvailable");
            boolean available = (boolean) isAvailable.invoke(null);
            if (!available) {
                return null;
            }

            // Create the backend
            return (GpuBackend) amdClass.getConstructor().newInstance();
        } catch (ClassNotFoundException e) {
            // AMD backend not on classpath
            return null;
        } catch (NoSuchMethodException e) {
            // API mismatch
            return null;
        } catch (IllegalAccessException e) {
            return null;
        } catch (InvocationTargetException e) {
            // HIP init failed (no device, driver issues, etc.)
            Throwable cause = e.getCause();
            if (cause != null && cause.getMessage() != null) {
                String msg = cause.getMessage();
                if (msg.contains("hipErrorNoDevice")) {
                    // Expected - no GPU
                    return null;
                }
            }
            return null;
        } catch (InstantiationException e) {
            return null;
        }
    }
}
