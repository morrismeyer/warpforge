package io.surfworks.snakegrinder.cli;

import io.surfworks.snakegrinder.core.MockTraceExport;
import io.surfworks.snakegrinder.core.MvpStableHlo;
import io.surfworks.snakegrinder.core.SnakeGrinder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class SnakeGrinderCli {

    public static void main(String[] args) {
        if (args == null) {
            args = new String[0];
        }

        if (hasFlag(args, "--self-test")) {
            assertGraalVmHost();
            System.out.println(SnakeGrinder.selfTest());
            return;
        }

        if (hasFlag(args, "--mvp-stablehlo")) {
            assertGraalVmHost();
            runMvpStableHlo(args);
            return;
        }

        if (hasFlag(args, "--trace")) {
            assertGraalVmHost();
            runTrace(args);
            return;
        }

        if (hasFlag(args, "--trace-example")) {
            assertGraalVmHost();
            runTraceExample(args);
            return;
        }

        if (hasFlag(args, "--help") || hasFlag(args, "-h")) {
            printHelp();
            return;
        }

        System.err.println("Unknown or missing arguments. Use --help.");
        System.exit(2);
    }

    private static void runMvpStableHlo(String[] args) {
        // Parse --out <dir> option
        Path outputDir = Paths.get("build/snakegrinder/mvp");
        boolean keepTmp = hasFlag(args, "--keep-tmp");

        for (int i = 0; i < args.length; i++) {
            if ("--out".equals(args[i]) && i + 1 < args.length) {
                outputDir = Paths.get(args[i + 1]);
            }
        }

        System.out.println("SnakeGrinder MVP StableHLO Export");
        System.out.println("Output directory: " + outputDir.toAbsolutePath());
        System.out.println();

        MvpStableHlo.ExportResult result = MvpStableHlo.run(outputDir, keepTmp);

        if (result.success) {
            System.out.println("SUCCESS: StableHLO export completed");
            System.out.println();
            System.out.println("Output files:");
            System.out.println("  MLIR:     " + result.mlirFile);
            System.out.println("  Manifest: " + result.manifestFile);
            System.out.println("  Log:      " + result.logFile);

            if (!result.warnings.isEmpty()) {
                System.out.println();
                System.out.println("Warnings:");
                for (String warning : result.warnings) {
                    System.out.println("  - " + warning);
                }
            }
        } else {
            System.err.println("FAILED: " + result.error);
            System.err.println();
            if (result.logFile != null) {
                System.err.println("See log for details: " + result.logFile);
            }
            if (result.manifestFile != null) {
                System.err.println("Manifest: " + result.manifestFile);
            }
            System.exit(1);
        }
    }

    private static void runTrace(String[] args) {
        // Parse options
        Path outputDir = Paths.get("build/snakegrinder/trace");
        String sourceFile = null;
        String functionName = null;
        String inputsSpec = null;
        String framework = "torch";

        for (int i = 0; i < args.length; i++) {
            if ("--out".equals(args[i]) && i + 1 < args.length) {
                outputDir = Paths.get(args[i + 1]);
            } else if ("--source".equals(args[i]) && i + 1 < args.length) {
                sourceFile = args[i + 1];
            } else if ("--function".equals(args[i]) && i + 1 < args.length) {
                functionName = args[i + 1];
            } else if ("--inputs".equals(args[i]) && i + 1 < args.length) {
                inputsSpec = args[i + 1];
            } else if ("--framework".equals(args[i]) && i + 1 < args.length) {
                framework = args[i + 1];
            }
        }

        if (sourceFile == null || functionName == null || inputsSpec == null) {
            System.err.println("Error: --trace requires --source, --function, and --inputs");
            System.err.println("Use --help for usage information");
            System.exit(2);
            return;
        }

        System.out.println("SnakeGrinder Mock Trace");
        System.out.println("Source:   " + sourceFile);
        System.out.println("Function: " + functionName);
        System.out.println("Inputs:   " + inputsSpec);
        System.out.println("Output:   " + outputDir.toAbsolutePath());
        System.out.println();

        // Read source file
        String pythonSource;
        try {
            pythonSource = Files.readString(Paths.get(sourceFile));
        } catch (IOException e) {
            System.err.println("Error reading source file: " + e.getMessage());
            System.exit(1);
            return;
        }

        // Parse input specs: "[(2,3),(3,4)]" or "[(2,3,'f32'),(3,4,'f32')]"
        List<MockTraceExport.InputSpec> inputSpecs = parseInputSpecs(inputsSpec);

        // Run trace
        MockTraceExport.TraceResult result = MockTraceExport.trace(
                pythonSource, functionName, inputSpecs, framework);

        if (result.success) {
            try {
                MockTraceExport.writeResult(result, outputDir);
                System.out.println("SUCCESS: Trace completed");
                System.out.println();
                System.out.println("Output files:");
                System.out.println("  MLIR:     " + outputDir.resolve("model.mlir"));
                System.out.println("  Manifest: " + outputDir.resolve("manifest.json"));
            } catch (IOException e) {
                System.err.println("Error writing output: " + e.getMessage());
                System.exit(1);
            }
        } else {
            System.err.println("FAILED: " + result.error);
            System.exit(1);
        }
    }

    private static void runTraceExample(String[] args) {
        // Parse options
        Path outputDir = Paths.get("build/snakegrinder/trace-example");
        String example = "matmul";

        for (int i = 0; i < args.length; i++) {
            if ("--out".equals(args[i]) && i + 1 < args.length) {
                outputDir = Paths.get(args[i + 1]);
            } else if ("--example".equals(args[i]) && i + 1 < args.length) {
                example = args[i + 1];
            }
        }

        System.out.println("SnakeGrinder Trace Example: " + example);
        System.out.println("Output: " + outputDir.toAbsolutePath());
        System.out.println();

        MockTraceExport.TraceResult result;
        if ("mlp".equals(example)) {
            result = MockTraceExport.traceMlpExample();
        } else {
            result = MockTraceExport.traceMatmulExample();
        }

        if (result.success) {
            try {
                MockTraceExport.writeResult(result, outputDir);
                System.out.println("SUCCESS: Example trace completed");
                System.out.println();
                System.out.println("Output files:");
                System.out.println("  MLIR:     " + outputDir.resolve("model.mlir"));
                System.out.println("  Manifest: " + outputDir.resolve("manifest.json"));
                System.out.println();
                System.out.println("StableHLO MLIR:");
                System.out.println("----------------------------------------");
                System.out.println(result.mlir);
                System.out.println("----------------------------------------");
            } catch (IOException e) {
                System.err.println("Error writing output: " + e.getMessage());
                System.exit(1);
            }
        } else {
            System.err.println("FAILED: " + result.error);
            System.exit(1);
        }
    }

    private static List<MockTraceExport.InputSpec> parseInputSpecs(String spec) {
        // Parse input specs like "[(2,3),(3,4)]" or "[(2,3,'f32'),(3,4,'f64')]"
        List<MockTraceExport.InputSpec> result = new ArrayList<>();

        // Remove outer brackets and whitespace
        spec = spec.trim();
        if (spec.startsWith("[")) spec = spec.substring(1);
        if (spec.endsWith("]")) spec = spec.substring(0, spec.length() - 1);

        // Split on ),( pattern (with optional spaces)
        String[] parts = spec.split("\\)\\s*,\\s*\\(");
        for (String part : parts) {
            part = part.trim();
            if (part.startsWith("(")) part = part.substring(1);
            if (part.endsWith(")")) part = part.substring(0, part.length() - 1);

            // Check if dtype is specified
            String dtype = "f32";
            if (part.contains("'")) {
                int quoteStart = part.indexOf("'");
                int quoteEnd = part.lastIndexOf("'");
                if (quoteEnd > quoteStart) {
                    dtype = part.substring(quoteStart + 1, quoteEnd);
                    part = part.substring(0, quoteStart).trim();
                    if (part.endsWith(",")) part = part.substring(0, part.length() - 1);
                }
            }

            // Parse shape
            String[] dims = part.split(",");
            int[] shape = new int[dims.length];
            for (int i = 0; i < dims.length; i++) {
                shape[i] = Integer.parseInt(dims[i].trim());
            }

            result.add(new MockTraceExport.InputSpec(shape, dtype));
        }

        return result;
    }

    private static boolean hasFlag(String[] args, String flag) {
        return Arrays.asList(args).contains(flag);
    }

    private static void printHelp() {
        System.out.println("snakegrinder-cli");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  snakegrinder-cli --self-test");
        System.out.println("  snakegrinder-cli --mvp-stablehlo [--out <dir>] [--keep-tmp]");
        System.out.println("  snakegrinder-cli --trace --source <file> --function <name> --inputs <specs> [options]");
        System.out.println("  snakegrinder-cli --trace-example [--example <name>] [--out <dir>]");
        System.out.println("  snakegrinder-cli --help");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  --self-test       Run GraalPy self-test");
        System.out.println("  --mvp-stablehlo   Export using real PyTorch/JAX (requires pip install)");
        System.out.println("  --trace           Trace Python code using mock PyTorch/JAX (no pip needed)");
        System.out.println("  --trace-example   Run a built-in trace example");
        System.out.println();
        System.out.println("Trace options:");
        System.out.println("  --source <file>   Python source file to trace");
        System.out.println("  --function <name> Function name to trace");
        System.out.println("  --inputs <specs>  Input shapes, e.g., '[(2,3),(3,4)]' or '[(2,3,\"f32\"),(3,4,\"f64\")]'");
        System.out.println("  --framework <fw>  Framework style: 'torch' or 'jax' (default: torch)");
        System.out.println("  --out <dir>       Output directory (default: build/snakegrinder/trace)");
        System.out.println();
        System.out.println("Trace example options:");
        System.out.println("  --example <name>  Example name: 'matmul' or 'mlp' (default: matmul)");
        System.out.println("  --out <dir>       Output directory (default: build/snakegrinder/trace-example)");
        System.out.println();
        System.out.println("MVP StableHLO options:");
        System.out.println("  --out <dir>       Output directory (default: build/snakegrinder/mvp)");
        System.out.println("  --keep-tmp        Keep temporary files for debugging");
    }

    /**
     * We want SnakeGrinder to run on a GraalVM JDK so GraalPy can use runtime compilation.
     *
     * Do not key only off java.vm.name, because Oracle GraalVM JDK may still report HotSpot.
     * Use java.home and GraalVM specific properties instead.
     */
    private static void assertGraalVmHost() {
        String javaHome = System.getProperty("java.home", "");
        String javaHomeLc = javaHome.toLowerCase();

        String graalHome = System.getProperty("org.graalvm.home");
        String graalVersion = System.getProperty("org.graalvm.version");

        String vmName = System.getProperty("java.vm.name", "");
        String runtimeName = System.getProperty("java.runtime.name", "");
        String vendor = System.getProperty("java.vendor", "");

        boolean looksLikeGraal =
                (graalHome != null && !graalHome.isBlank()) ||
                (graalVersion != null && !graalVersion.isBlank()) ||
                javaHomeLc.contains("graalvm") ||
                vmName.toLowerCase().contains("graal") ||
                runtimeName.toLowerCase().contains("graal");

        if (!looksLikeGraal) {
            throw new IllegalStateException(
                    "SnakeGrinder must run on GraalVM JDK 25 (for GraalPy runtime compilation). " +
                    "Current VM: " + vmName + " / " + runtimeName + " / " + vendor + ". " +
                    "Current java.home: " + javaHome + ". " +
                    "If Gradle toolchains selected a non-Graal JDK 25, install Oracle GraalVM 25 " +
                    "and configure Gradle toolchain discovery to find it."
            );
        }
    }
}
