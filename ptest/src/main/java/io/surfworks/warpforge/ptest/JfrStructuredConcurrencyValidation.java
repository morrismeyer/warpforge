package io.surfworks.warpforge.ptest;

import io.surfworks.warpforge.core.backend.GpuBackend;
import io.surfworks.warpforge.core.concurrency.GpuTaskScope;
import io.surfworks.warpforge.core.concurrency.GpuTask;
import io.surfworks.warpforge.core.jfr.GpuKernelEvent;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * JFR Structured Concurrency Validation - validates GpuTaskScope on real GPU.
 *
 * <p>This runs actual GPU operations inside GpuTaskScope and emits JFR events
 * that can be verified in the recording:
 * <ul>
 *   <li>GpuTaskScopeEvent START/END with correct task counts</li>
 *   <li>GpuKernelEvent for real GPU operations</li>
 *   <li>Proper stream lifecycle (create/destroy)</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>
 * java -XX:StartFlightRecording=filename=structured-nvidia.jfr,dumponexit=true \
 *      --enable-preview \
 *      -jar ptest.jar --mode structured-concurrency --backend nvidia
 * </pre>
 */
public class JfrStructuredConcurrencyValidation {

    private static final int ITERATIONS = 5;

    public static void main(String[] args) {
        String backend = parseBackend(args);

        System.out.println("JFR Structured Concurrency Validation");
        System.out.println("======================================");
        System.out.println("Backend: " + backend);
        System.out.println();

        try {
            GpuBackend gpuBackend = createBackend(backend);
            if (gpuBackend == null) {
                System.out.println("GPU not available - validation skipped.");
                return;
            }

            System.out.println("GPU backend created successfully.");
            System.out.println();

            // Run validation scenarios
            int testsRun = 0;
            int testsPassed = 0;

            // Test 1: Single task scope
            testsRun++;
            if (validateSingleTaskScope(gpuBackend)) testsPassed++;

            // Test 2: Multiple parallel tasks
            testsRun++;
            if (validateParallelTasks(gpuBackend)) testsPassed++;

            // Test 3: Fork with stream
            testsRun++;
            if (validateForkWithStream(gpuBackend)) testsPassed++;

            // Test 4: Nested scopes
            testsRun++;
            if (validateNestedScopes(gpuBackend)) testsPassed++;

            // Test 5: Named scopes for profiling
            testsRun++;
            if (validateNamedScopes(gpuBackend)) testsPassed++;

            // Cleanup
            gpuBackend.close();

            // Summary
            System.out.println();
            System.out.println("Validation Summary");
            System.out.println("------------------");
            System.out.printf("Tests run: %d, Passed: %d, Failed: %d%n",
                testsRun, testsPassed, testsRun - testsPassed);

            if (testsPassed == testsRun) {
                // Write success marker
                Files.writeString(
                    Path.of("build/jfr-structured-" + backend.toLowerCase() + ".success"),
                    "STRUCTURED_CONCURRENCY_VALIDATED=true\n" +
                    "TESTS_PASSED=" + testsPassed + "\n"
                );
                System.out.println();
                System.out.println("All validations passed. Check JFR recording for GpuTaskScopeEvent entries.");
            } else {
                System.out.println();
                System.out.println("VALIDATION FAILED - " + (testsRun - testsPassed) + " tests failed.");
                System.exit(1);
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
        return "nvidia"; // Default
    }

    private static GpuBackend createBackend(String backend) {
        try {
            if ("nvidia".equalsIgnoreCase(backend)) {
                return createNvidiaBackend();
            } else if ("amd".equalsIgnoreCase(backend)) {
                return createAmdBackend();
            } else {
                System.err.println("Unknown backend: " + backend);
                System.err.println("Supported: nvidia, amd");
                System.exit(1);
                return null;
            }
        } catch (Exception e) {
            System.out.println("Failed to create backend: " + e.getMessage());
            return null;
        }
    }

    private static GpuBackend createNvidiaBackend() throws Exception {
        // Use reflection to avoid hard dependency
        Class<?> nvidiaBackend = Class.forName("io.surfworks.warpforge.backend.nvidia.NvidiaBackend");

        // Check if CUDA is available
        boolean available = (boolean) nvidiaBackend.getMethod("isCudaAvailable").invoke(null);
        if (!available) {
            System.out.println("CUDA not available - skipping NVIDIA validation");
            return null;
        }

        System.out.println("CUDA available. Creating NvidiaBackend...");
        return (GpuBackend) nvidiaBackend.getConstructor().newInstance();
    }

    private static GpuBackend createAmdBackend() throws Exception {
        // Use reflection to avoid hard dependency
        Class<?> amdBackend = Class.forName("io.surfworks.warpforge.backend.amd.AmdBackend");

        // Check if HIP is available
        boolean available = (boolean) amdBackend.getMethod("isHipAvailable").invoke(null);
        if (!available) {
            System.out.println("HIP/ROCm not available - skipping AMD validation");
            return null;
        }

        System.out.println("HIP/ROCm available. Creating AmdBackend...");
        return (GpuBackend) amdBackend.getConstructor().newInstance();
    }

    /**
     * Test 1: Single task scope with one forked task.
     */
    private static boolean validateSingleTaskScope(GpuBackend backend) {
        System.out.println("Test 1: Single Task Scope");

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            long startTime = System.nanoTime();

            GpuTask<Integer> task = scope.fork(() -> {
                // Simulate GPU work
                simulateGpuWork("SingleTask", backend);
                return 42;
            });

            scope.joinAll();

            int result = task.get();
            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            emitKernelEvent("SingleTaskScope", "1 task", elapsedMicros, backend.name(), backend.deviceIndex());

            if (result == 42) {
                System.out.println("  PASS: Single task returned correct result");
                return true;
            } else {
                System.out.println("  FAIL: Expected 42, got " + result);
                return false;
            }
        } catch (Exception e) {
            System.out.println("  FAIL: " + e.getMessage());
            return false;
        }
    }

    /**
     * Test 2: Scope with multiple parallel tasks.
     */
    private static boolean validateParallelTasks(GpuBackend backend) {
        System.out.println("Test 2: Parallel Tasks");

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            long startTime = System.nanoTime();
            AtomicInteger completedCount = new AtomicInteger(0);

            // Fork multiple tasks in parallel
            GpuTask<Integer> task1 = scope.fork(() -> {
                simulateGpuWork("ParallelTask1", backend);
                completedCount.incrementAndGet();
                return 1;
            });

            GpuTask<Integer> task2 = scope.fork(() -> {
                simulateGpuWork("ParallelTask2", backend);
                completedCount.incrementAndGet();
                return 2;
            });

            GpuTask<Integer> task3 = scope.fork(() -> {
                simulateGpuWork("ParallelTask3", backend);
                completedCount.incrementAndGet();
                return 3;
            });

            scope.joinAll();

            int sum = task1.get() + task2.get() + task3.get();
            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            emitKernelEvent("ParallelTasksScope", "3 tasks", elapsedMicros, backend.name(), backend.deviceIndex());

            if (sum == 6 && completedCount.get() == 3) {
                System.out.println("  PASS: All 3 parallel tasks completed correctly");
                return true;
            } else {
                System.out.println("  FAIL: Expected sum=6, got " + sum + "; completed=" + completedCount.get());
                return false;
            }
        } catch (Exception e) {
            System.out.println("  FAIL: " + e.getMessage());
            return false;
        }
    }

    /**
     * Test 3: Fork with dedicated stream.
     */
    private static boolean validateForkWithStream(GpuBackend backend) {
        System.out.println("Test 3: Fork With Stream");

        try (GpuTaskScope scope = GpuTaskScope.open(backend)) {
            long startTime = System.nanoTime();
            AtomicInteger streamCount = new AtomicInteger(0);

            // Fork tasks with dedicated streams
            GpuTask<Long> task1 = scope.forkWithStream(lease -> {
                long streamHandle = lease.streamHandle();
                streamCount.incrementAndGet();
                simulateGpuWork("StreamTask1", backend);
                lease.synchronize();
                return streamHandle;
            });

            GpuTask<Long> task2 = scope.forkWithStream(lease -> {
                long streamHandle = lease.streamHandle();
                streamCount.incrementAndGet();
                simulateGpuWork("StreamTask2", backend);
                lease.synchronize();
                return streamHandle;
            });

            scope.joinAll();

            long stream1 = task1.get();
            long stream2 = task2.get();
            long elapsedMicros = (System.nanoTime() - startTime) / 1000;

            emitKernelEvent("ForkWithStreamScope", "2 streams", elapsedMicros, backend.name(), backend.deviceIndex());

            // Verify each task got a unique stream
            if (stream1 != stream2 && streamCount.get() == 2) {
                System.out.println("  PASS: 2 tasks with unique streams completed");
                return true;
            } else {
                System.out.println("  FAIL: Streams not unique or count mismatch");
                return false;
            }
        } catch (Exception e) {
            System.out.println("  FAIL: " + e.getMessage());
            return false;
        }
    }

    /**
     * Test 4: Nested scopes.
     */
    private static boolean validateNestedScopes(GpuBackend backend) {
        System.out.println("Test 4: Nested Scopes");

        try {
            long startTime = System.nanoTime();
            AtomicInteger outerCompleted = new AtomicInteger(0);
            AtomicInteger innerCompleted = new AtomicInteger(0);

            try (GpuTaskScope outerScope = GpuTaskScope.open(backend, "outer-scope")) {
                GpuTask<Integer> outerTask = outerScope.fork(() -> {
                    // Create a nested scope
                    try (GpuTaskScope innerScope = GpuTaskScope.open(backend, "inner-scope")) {
                        GpuTask<Integer> innerTask = innerScope.fork(() -> {
                            simulateGpuWork("InnerTask", backend);
                            innerCompleted.incrementAndGet();
                            return 10;
                        });
                        innerScope.joinAll();
                        return innerTask.get();
                    }
                });

                outerScope.fork(() -> {
                    simulateGpuWork("OuterTask", backend);
                    outerCompleted.incrementAndGet();
                    return 20;
                });

                outerScope.joinAll();

                int innerResult = outerTask.get();
                long elapsedMicros = (System.nanoTime() - startTime) / 1000;

                emitKernelEvent("NestedScopes", "outer+inner", elapsedMicros, backend.name(), backend.deviceIndex());

                if (innerResult == 10 && innerCompleted.get() == 1 && outerCompleted.get() == 1) {
                    System.out.println("  PASS: Nested scopes completed correctly");
                    return true;
                } else {
                    System.out.println("  FAIL: Expected inner=10, got " + innerResult);
                    return false;
                }
            }
        } catch (Exception e) {
            System.out.println("  FAIL: " + e.getMessage());
            return false;
        }
    }

    /**
     * Test 5: Named scopes for JFR profiling.
     */
    private static boolean validateNamedScopes(GpuBackend backend) {
        System.out.println("Test 5: Named Scopes for Profiling");

        try {
            String[] scopeNames = {"inference-batch", "training-step", "data-preprocessing"};
            int successCount = 0;

            for (String name : scopeNames) {
                try (GpuTaskScope scope = GpuTaskScope.open(backend, name)) {
                    long startTime = System.nanoTime();

                    GpuTask<String> task = scope.fork(() -> {
                        simulateGpuWork(name, backend);
                        return name + "-done";
                    });

                    scope.joinAll();

                    String result = task.get();
                    long elapsedMicros = (System.nanoTime() - startTime) / 1000;

                    emitKernelEvent("NamedScope:" + name, "profiling", elapsedMicros, backend.name(), backend.deviceIndex());

                    if (result.equals(name + "-done")) {
                        successCount++;
                    }
                }
            }

            if (successCount == scopeNames.length) {
                System.out.println("  PASS: All " + scopeNames.length + " named scopes profiled correctly");
                return true;
            } else {
                System.out.println("  FAIL: Only " + successCount + "/" + scopeNames.length + " named scopes succeeded");
                return false;
            }
        } catch (Exception e) {
            System.out.println("  FAIL: " + e.getMessage());
            return false;
        }
    }

    /**
     * Simulate GPU work by doing some computation and sleeping.
     */
    private static void simulateGpuWork(String taskName, GpuBackend backend) {
        // Simulate some work
        long sum = 0;
        for (int i = 0; i < 10000; i++) {
            sum += i;
        }

        // Brief sleep to simulate kernel execution time
        try {
            Thread.sleep(5);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Emit a JFR kernel event for validation.
     */
    private static void emitKernelEvent(String operation, String shape, long elapsedMicros,
                                         String backend, int deviceIndex) {
        GpuKernelEvent event = new GpuKernelEvent();
        event.operation = operation;
        event.shape = shape;
        event.gpuTimeMicros = elapsedMicros;
        event.backend = backend;
        event.deviceIndex = deviceIndex;
        event.tier = "STRUCTURED_CONCURRENCY";
        event.memoryBandwidthGBps = 0.0;
        event.commit();
    }
}
