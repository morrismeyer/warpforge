package io.surfworks.warpforge.ptest.research;

import io.surfworks.warpforge.core.backend.GpuBackend;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Research Validation Runner - executes research-informed validation scenarios.
 *
 * <p>This runner executes validation scenarios inspired by recent ML systems research:
 * <ul>
 *   <li><b>Tally (ASPLOS 2025)</b> - Compute/memory overlap patterns</li>
 *   <li><b>PipeFill (MLSys 2025)</b> - Pipeline bubble filling</li>
 *   <li><b>Orion (EuroSys 2024)</b> - Occupancy-based admission control</li>
 *   <li><b>Alibaba Aegaeon (SOSP 2025)</b> - SLO-bounded inference</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>
 * java -XX:StartFlightRecording=filename=build/jfr-research.jfr,dumponexit=true \
 *      --enable-preview --enable-native-access=ALL-UNNAMED \
 *      -cp ptest.jar io.surfworks.warpforge.ptest.research.ResearchValidationRunner \
 *      --backend nvidia
 * </pre>
 */
public class ResearchValidationRunner {

    private static final String NVIDIA_BACKEND_CLASS = "io.surfworks.warpforge.backend.nvidia.NvidiaBackend";
    private static final String AMD_BACKEND_CLASS = "io.surfworks.warpforge.backend.amd.AmdBackend";

    public static void main(String[] args) {
        String backend = parseBackend(args);
        boolean verbose = hasFlag(args, "--verbose");

        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║         Research Validation Runner                           ║");
        System.out.println("║  Validating ML systems research-informed scenarios           ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Backend: " + backend);
        System.out.println("Verbose: " + verbose);
        System.out.println();

        try {
            GpuBackend gpuBackend = createBackend(backend);
            if (gpuBackend == null) {
                System.out.println("GPU not available - validation skipped.");
                System.exit(0);
            }

            System.out.println("GPU backend created: " + gpuBackend.name());
            System.out.printf("Device memory: %.1f GB%n", gpuBackend.totalDeviceMemory() / 1e9);
            System.out.println();

            // Run all validation suites
            List<ValidationResult> results = new ArrayList<>();
            Instant startTime = Instant.now();

            // 1. Tally-inspired: Overlapping I/O validation
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            System.out.println("1. Tally (ASPLOS 2025) - Compute/Memory Overlap");
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            results.addAll(OverlappingIoValidation.runAll(gpuBackend, verbose));
            System.out.println();

            // 2. PipeFill-inspired: Pipeline bubble validation
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            System.out.println("2. PipeFill (MLSys 2025) - Pipeline Bubble Filling");
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            results.addAll(PipelineBubbleValidation.runAll(gpuBackend, verbose));
            System.out.println();

            // 3. Orion-inspired: Occupancy admission validation
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            System.out.println("3. Orion (EuroSys 2024) - Occupancy-Based Admission");
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            results.addAll(OccupancyAdmissionValidation.runAll(gpuBackend, verbose));
            System.out.println();

            // 4. Alibaba-inspired: SLO inference validation
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            System.out.println("4. Alibaba Aegaeon (SOSP 2025) - SLO-Bounded Inference");
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            results.addAll(SloInferenceValidation.runAll(gpuBackend, verbose));
            System.out.println();

            Duration totalDuration = Duration.between(startTime, Instant.now());

            // Cleanup
            gpuBackend.close();

            // Summary
            printSummary(results, totalDuration, backend);

        } catch (Exception e) {
            System.err.println("Research validation failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void printSummary(List<ValidationResult> results, Duration totalDuration, String backend) {
        int passed = (int) results.stream().filter(ValidationResult::passed).count();
        int failed = results.size() - passed;

        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║                    Validation Summary                        ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
        System.out.println();

        // Print results table
        System.out.println("┌────────────────────────────────────────┬──────────┬──────────┐");
        System.out.println("│ Scenario                               │ Status   │ Duration │");
        System.out.println("├────────────────────────────────────────┼──────────┼──────────┤");
        for (ValidationResult result : results) {
            String status = result.passed() ? "✓ PASS" : "✗ FAIL";
            String name = truncate(result.name(), 38);
            String duration = formatDuration(result.duration());
            System.out.printf("│ %-38s │ %-8s │ %8s │%n", name, status, duration);
        }
        System.out.println("└────────────────────────────────────────┴──────────┴──────────┘");
        System.out.println();

        System.out.printf("Total scenarios: %d, Passed: %d, Failed: %d%n", results.size(), passed, failed);
        System.out.printf("Total duration: %s%n", formatDuration(totalDuration));
        System.out.println();

        if (failed == 0) {
            System.out.println("All research validations PASSED.");
            try {
                Files.writeString(
                    Path.of("build/jfr-research-" + backend.toLowerCase() + ".success"),
                    "RESEARCH_VALIDATION_PASSED=true\n" +
                    "SCENARIOS_PASSED=" + passed + "\n" +
                    "DURATION_MS=" + totalDuration.toMillis() + "\n"
                );
            } catch (Exception e) {
                // Ignore file write errors
            }
        } else {
            System.out.println("Research validation FAILED - " + failed + " scenarios failed.");
            System.out.println();
            System.out.println("Failed scenarios:");
            for (ValidationResult result : results) {
                if (!result.passed()) {
                    System.out.println("  - " + result.name() + ": " + result.message());
                }
            }
            System.exit(1);
        }
    }

    private static String truncate(String s, int maxLen) {
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen - 3) + "...";
    }

    private static String formatDuration(Duration d) {
        if (d.toMillis() < 1000) {
            return d.toMillis() + "ms";
        } else if (d.toSeconds() < 60) {
            return String.format("%.1fs", d.toMillis() / 1000.0);
        } else {
            return String.format("%dm%ds", d.toMinutes(), d.toSecondsPart());
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

    private static boolean hasFlag(String[] args, String flag) {
        for (String arg : args) {
            if (flag.equals(arg)) return true;
        }
        return false;
    }

    private static GpuBackend createBackend(String backendName) {
        String className = backendName.toLowerCase().contains("nvidia") ?
            NVIDIA_BACKEND_CLASS : AMD_BACKEND_CLASS;

        try {
            Class<?> backendClass = Class.forName(className);

            // Check availability
            String availMethod = backendName.toLowerCase().contains("nvidia") ?
                "isCudaAvailable" : "isRocmAvailable";
            Method isAvailable = backendClass.getMethod(availMethod);
            boolean available = (boolean) isAvailable.invoke(null);

            if (!available) {
                return null;
            }

            return (GpuBackend) backendClass.getConstructor().newInstance();
        } catch (ClassNotFoundException e) {
            System.out.println("Backend class not found: " + className);
            return null;
        } catch (Exception e) {
            System.out.println("Failed to create backend: " + e.getMessage());
            return null;
        }
    }

    /**
     * Result of a single validation scenario.
     */
    public record ValidationResult(
        String name,
        boolean passed,
        String message,
        Duration duration
    ) {
        public static ValidationResult pass(String name, Duration duration) {
            return new ValidationResult(name, true, "OK", duration);
        }

        public static ValidationResult pass(String name, String message, Duration duration) {
            return new ValidationResult(name, true, message, duration);
        }

        public static ValidationResult fail(String name, String message, Duration duration) {
            return new ValidationResult(name, false, message, duration);
        }
    }
}
