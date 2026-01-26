package io.surfworks.warpforge.ptest;

import io.surfworks.warpforge.core.jfr.GpuKernelEvent;
import io.surfworks.warpforge.core.jfr.GpuMemoryEvent;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * JFR GPU Validation - validates that GPU timing events are properly captured.
 *
 * <p>This tool runs simple GPU operations and emits JFR events that can be
 * verified in the resulting recording. It supports both NVIDIA and AMD backends.
 *
 * <p>Usage:
 * <pre>
 * java -XX:StartFlightRecording=filename=gpu.jfr,dumponexit=true \
 *      -jar ptest.jar --backend nvidia
 * </pre>
 */
public class JfrGpuValidation {

    private static final int ITERATIONS = 10;
    private static final int ALLOCATION_SIZE = 1024 * 1024; // 1MB

    public static void main(String[] args) {
        String backend = parseBackend(args);

        System.out.println("JFR GPU Validation");
        System.out.println("==================");
        System.out.println("Backend: " + backend);
        System.out.println("Iterations: " + ITERATIONS);
        System.out.println();

        try {
            boolean gpuAvailable;
            if ("nvidia".equalsIgnoreCase(backend)) {
                gpuAvailable = runNvidiaValidation();
            } else if ("amd".equalsIgnoreCase(backend)) {
                gpuAvailable = runAmdValidation();
            } else if ("cpu".equalsIgnoreCase(backend)) {
                runCpuValidation();
                gpuAvailable = true; // CPU always available
            } else {
                System.err.println("Unknown backend: " + backend);
                System.err.println("Supported: nvidia, amd, cpu");
                System.exit(1);
                return;
            }

            System.out.println();
            if (gpuAvailable) {
                // Write marker file to indicate GPU events were captured
                java.nio.file.Files.writeString(
                    java.nio.file.Path.of("build/jfr-" + backend.toLowerCase() + ".success"),
                    "GPU_EVENTS_CAPTURED=true\n"
                );
                System.out.println("Validation complete. Check JFR recording for GpuKernelEvent entries.");
            } else {
                System.out.println("GPU not available - validation skipped.");
            }

        } catch (Throwable t) {
            System.err.println("Validation failed: " + t.getMessage());
            t.printStackTrace();
            System.exit(1);
        }
    }

    private static String parseBackend(String[] args) {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--backend".equals(args[i])) {
                return args[i + 1];
            }
        }
        return "cpu"; // Default to CPU for basic validation
    }

    private static boolean runNvidiaValidation() throws Throwable {
        System.out.println("Checking NVIDIA/CUDA availability...");

        // Use reflection to avoid hard dependency
        Class<?> cudaRuntime = Class.forName("io.surfworks.warpforge.backend.nvidia.cuda.CudaRuntime");
        Class<?> cudaContext = Class.forName("io.surfworks.warpforge.backend.nvidia.cuda.CudaContext");

        // Check availability
        boolean available = (boolean) cudaRuntime.getMethod("isAvailable").invoke(null);
        if (!available) {
            System.out.println("CUDA not available - skipping NVIDIA validation");
            return false;
        }

        System.out.println("CUDA available. Running validation...");

        // Create context - may fail if no device even though library is available
        Object ctx;
        try {
            ctx = cudaContext.getMethod("create", int.class).invoke(null, 0);
        } catch (java.lang.reflect.InvocationTargetException e) {
            Throwable cause = e.getCause();
            if (cause != null && cause.getMessage() != null &&
                (cause.getMessage().contains("no CUDA-capable device") ||
                 cause.getMessage().contains("cuInit"))) {
                System.out.println("No CUDA device detected - skipping NVIDIA validation");
                return false;
            }
            throw e;
        }

        try {
            for (int i = 0; i < ITERATIONS; i++) {
                // Time a memory allocation operation
                float elapsedMs = (float) cudaContext.getMethod("timeOperation", Runnable.class)
                    .invoke(ctx, (Runnable) () -> {
                        try {
                            long ptr = (long) cudaContext.getMethod("allocate", long.class)
                                .invoke(ctx, (long) ALLOCATION_SIZE);
                            cudaContext.getMethod("free", long.class).invoke(ctx, ptr);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });

                // Emit JFR event
                emitKernelEvent("MemoryAllocFree", "1MB", elapsedMs, "CUDA", 0);

                System.out.printf("  Iteration %d: %.3f ms%n", i + 1, elapsedMs);
            }
        } finally {
            cudaContext.getMethod("close").invoke(ctx);
        }
        return true;
    }

    private static boolean runAmdValidation() throws Throwable {
        System.out.println("Checking AMD/HIP availability...");

        // Use reflection to avoid hard dependency
        Class<?> hipRuntime = Class.forName("io.surfworks.warpforge.backend.amd.hip.HipRuntime");
        Class<?> hipContext = Class.forName("io.surfworks.warpforge.backend.amd.hip.HipContext");

        // Check availability
        boolean available = (boolean) hipRuntime.getMethod("isAvailable").invoke(null);
        if (!available) {
            System.out.println("HIP/ROCm not available - skipping AMD validation");
            return false;
        }

        System.out.println("HIP/ROCm available. Running validation...");

        // Create context - may fail if no device even though library is available
        Object ctx;
        try {
            ctx = hipContext.getMethod("create", int.class).invoke(null, 0);
        } catch (java.lang.reflect.InvocationTargetException e) {
            Throwable cause = e.getCause();
            if (cause != null && cause.getMessage() != null &&
                cause.getMessage().contains("hipErrorNoDevice")) {
                System.out.println("No ROCm device detected - skipping AMD validation");
                return false;
            }
            throw e;
        }

        try {
            for (int i = 0; i < ITERATIONS; i++) {
                // Time a memory allocation operation
                float elapsedMs = (float) hipContext.getMethod("timeOperation", Runnable.class)
                    .invoke(ctx, (Runnable) () -> {
                        try {
                            long ptr = (long) hipContext.getMethod("allocate", long.class)
                                .invoke(ctx, (long) ALLOCATION_SIZE);
                            hipContext.getMethod("free", long.class).invoke(ctx, ptr);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    });

                // Emit JFR event
                emitKernelEvent("MemoryAllocFree", "1MB", elapsedMs, "HIP", 0);

                System.out.printf("  Iteration %d: %.3f ms%n", i + 1, elapsedMs);
            }
        } finally {
            hipContext.getMethod("close").invoke(ctx);
        }
        return true;
    }

    private static void runCpuValidation() {
        System.out.println("Running CPU validation (no GPU required)...");

        try (Arena arena = Arena.ofConfined()) {
            for (int i = 0; i < ITERATIONS; i++) {
                long startNs = System.nanoTime();

                // Simulate work - allocate and fill memory
                MemorySegment segment = arena.allocate(ALLOCATION_SIZE);
                for (int j = 0; j < ALLOCATION_SIZE; j += 4) {
                    segment.set(ValueLayout.JAVA_INT, j, j);
                }

                long elapsedNs = System.nanoTime() - startNs;
                float elapsedMs = elapsedNs / 1_000_000.0f;

                // Emit JFR event (simulating GPU event for validation)
                emitKernelEvent("CpuMemoryFill", "1MB", elapsedMs, "CPU", 0);

                System.out.printf("  Iteration %d: %.3f ms%n", i + 1, elapsedMs);
            }
        }
    }

    private static void emitKernelEvent(String operation, String shape, float elapsedMs,
                                         String backend, int deviceIndex) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = operation;
        event.shape = shape;
        event.gpuTimeMicros = (long) (elapsedMs * 1000);
        event.backend = backend;
        event.deviceIndex = deviceIndex;
        event.tier = "VALIDATION";

        // Calculate approximate throughput for memory ops
        double bytesPerSec = ALLOCATION_SIZE / (elapsedMs / 1000.0);
        event.memoryBandwidthGBps = bytesPerSec / (1024 * 1024 * 1024);

        event.commit();
    }
}
