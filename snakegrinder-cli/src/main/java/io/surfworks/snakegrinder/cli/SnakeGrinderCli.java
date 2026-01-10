package io.surfworks.snakegrinder.cli;

import io.surfworks.snakegrinder.core.FxStableHloExport;
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

        if (hasFlag(args, "--pytorch-info")) {
            assertGraalVmHost();
            runPyTorchInfo();
            return;
        }

        if (hasFlag(args, "--help") || hasFlag(args, "-h")) {
            printHelp();
            return;
        }

        System.err.println("Unknown or missing arguments. Use --help.");
        System.exit(2);
    }

    private static void runTrace(String[] args) {
        Path outputDir = Paths.get("build/snakegrinder/trace");
        String sourceFile = null;
        String className = null;
        String inputsSpec = null;

        for (int i = 0; i < args.length; i++) {
            if ("--out".equals(args[i]) && i + 1 < args.length) {
                outputDir = Paths.get(args[i + 1]);
            } else if ("--source".equals(args[i]) && i + 1 < args.length) {
                sourceFile = args[i + 1];
            } else if ("--class".equals(args[i]) && i + 1 < args.length) {
                className = args[i + 1];
            } else if ("--inputs".equals(args[i]) && i + 1 < args.length) {
                inputsSpec = args[i + 1];
            }
        }

        if (sourceFile == null || className == null || inputsSpec == null) {
            System.err.println("Error: --trace requires --source, --class, and --inputs");
            System.err.println("Use --help for usage information");
            System.exit(2);
            return;
        }

        System.out.println("SnakeGrinder Trace");
        System.out.println("Source: " + sourceFile);
        System.out.println("Class:  " + className);
        System.out.println("Inputs: " + inputsSpec);
        System.out.println("Output: " + outputDir.toAbsolutePath());
        System.out.println();

        String pythonSource;
        try {
            pythonSource = Files.readString(Paths.get(sourceFile));
        } catch (IOException e) {
            System.err.println("Error reading source file: " + e.getMessage());
            System.exit(1);
            return;
        }

        List<FxStableHloExport.InputSpec> inputSpecs = parseInputSpecs(inputsSpec);

        FxStableHloExport.TraceResult result = FxStableHloExport.trace(
                pythonSource, className, inputSpecs);

        if (result.success) {
            try {
                FxStableHloExport.writeResult(result, outputDir);
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
            if (result.traceback != null) {
                System.err.println();
                System.err.println("Traceback:");
                System.err.println(result.traceback);
            }
            System.exit(1);
        }
    }

    private static void runTraceExample(String[] args) {
        Path outputDir = Paths.get("build/snakegrinder/trace-example");

        for (int i = 0; i < args.length; i++) {
            if ("--out".equals(args[i]) && i + 1 < args.length) {
                outputDir = Paths.get(args[i + 1]);
            }
        }

        System.out.println("SnakeGrinder Trace Example");
        System.out.println("Output: " + outputDir.toAbsolutePath());
        System.out.println();

        FxStableHloExport.TraceResult result = FxStableHloExport.traceMlpExample();

        if (result.success) {
            try {
                FxStableHloExport.writeResult(result, outputDir);
                System.out.println("SUCCESS: Trace example completed");
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

    private static void runPyTorchInfo() {
        System.out.println("SnakeGrinder PyTorch Info");
        System.out.println();

        var info = FxStableHloExport.getPyTorchInfo();

        if (info.containsKey("error")) {
            System.err.println("PyTorch not available: " + info.get("error"));
            System.exit(1);
            return;
        }

        System.out.println("PyTorch version:  " + info.get("pytorch_version"));
        System.out.println("CUDA available:   " + info.get("cuda_available"));
        System.out.println("MPS available:    " + info.get("mps_available"));
        System.out.println("FX available:     " + info.get("fx_available"));
    }

    private static List<FxStableHloExport.InputSpec> parseInputSpecs(String spec) {
        List<FxStableHloExport.InputSpec> result = new ArrayList<>();

        spec = spec.trim();
        if (spec.startsWith("[")) spec = spec.substring(1);
        if (spec.endsWith("]")) spec = spec.substring(0, spec.length() - 1);

        String[] parts = spec.split("\\)\\s*,\\s*\\(");
        for (String part : parts) {
            part = part.trim();
            if (part.startsWith("(")) part = part.substring(1);
            if (part.endsWith(")")) part = part.substring(0, part.length() - 1);

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

            String[] dims = part.split(",");
            int[] shape = new int[dims.length];
            for (int i = 0; i < dims.length; i++) {
                shape[i] = Integer.parseInt(dims[i].trim());
            }

            result.add(new FxStableHloExport.InputSpec(shape, dtype));
        }

        return result;
    }

    private static boolean hasFlag(String[] args, String flag) {
        return Arrays.asList(args).contains(flag);
    }

    private static void printHelp() {
        System.out.println("snakegrinder - PyTorch to StableHLO converter");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  snakegrinder --trace --source <file> --class <name> --inputs <specs> [--out <dir>]");
        System.out.println("  snakegrinder --trace-example [--out <dir>]");
        System.out.println("  snakegrinder --pytorch-info");
        System.out.println("  snakegrinder --self-test");
        System.out.println("  snakegrinder --help");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  --trace          Trace a PyTorch nn.Module and convert to StableHLO MLIR");
        System.out.println("  --trace-example  Run a built-in example (SimpleMLP)");
        System.out.println("  --pytorch-info   Show PyTorch version and capabilities");
        System.out.println("  --self-test      Run GraalPy self-test");
        System.out.println();
        System.out.println("Trace options:");
        System.out.println("  --source <file>  Python source file containing nn.Module class");
        System.out.println("  --class <name>   nn.Module class name to trace");
        System.out.println("  --inputs <specs> Input shapes, e.g., '[(1,8)]' or '[(1,8),(8,16)]'");
        System.out.println("  --out <dir>      Output directory (default: build/snakegrinder/trace)");
        System.out.println();
        System.out.println("Example:");
        System.out.println("  snakegrinder --trace --source model.py --class MyModel --inputs '[(1,32)]'");
        System.out.println();
        System.out.println("Output: StableHLO MLIR written to <out>/model.mlir");
    }

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
