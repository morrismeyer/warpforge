package io.surfworks.snakegrinder.cli;

import io.surfworks.snakegrinder.core.FxStableHloExport;
import io.surfworks.snakegrinder.core.SnakeGrinder;
import io.surfworks.warpforge.license.ActivationResult;
import io.surfworks.warpforge.license.LicenseCheckResult;
import io.surfworks.warpforge.license.LicenseInfo;
import io.surfworks.warpforge.license.LicenseManager;

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

        if (hasFlag(args, "--trace-with-values")) {
            assertGraalVmHost();
            runTraceWithValues(args);
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

        if (hasFlag(args, "--activate")) {
            runActivate(args);
            return;
        }

        if (hasFlag(args, "--deactivate")) {
            runDeactivate();
            return;
        }

        if (hasFlag(args, "--license-info")) {
            runLicenseInfo();
            return;
        }

        System.err.println("Unknown or missing arguments. Use --help.");
        System.exit(2);
    }

    private static void runTrace(String[] args) {
        if (!checkLicenseForTrace()) {
            return;
        }

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

    private static void runTraceWithValues(String[] args) {
        if (!checkLicenseForTrace()) {
            return;
        }

        Path outputDir = Paths.get("build/snakegrinder/trace-with-values");
        String sourceFile = null;
        String className = null;
        String inputsSpec = null;
        long seed = 42; // Default seed

        for (int i = 0; i < args.length; i++) {
            if ("--out".equals(args[i]) && i + 1 < args.length) {
                outputDir = Paths.get(args[i + 1]);
            } else if ("--source".equals(args[i]) && i + 1 < args.length) {
                sourceFile = args[i + 1];
            } else if ("--class".equals(args[i]) && i + 1 < args.length) {
                className = args[i + 1];
            } else if ("--inputs".equals(args[i]) && i + 1 < args.length) {
                inputsSpec = args[i + 1];
            } else if ("--seed".equals(args[i]) && i + 1 < args.length) {
                seed = Long.parseLong(args[i + 1]);
            }
        }

        if (sourceFile == null || className == null || inputsSpec == null) {
            System.err.println("Error: --trace-with-values requires --source, --class, and --inputs");
            System.err.println("Use --help for usage information");
            System.exit(2);
            return;
        }

        System.out.println("SnakeGrinder Trace With Values");
        System.out.println("Source: " + sourceFile);
        System.out.println("Class:  " + className);
        System.out.println("Inputs: " + inputsSpec);
        System.out.println("Seed:   " + seed);
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

        FxStableHloExport.TraceResult result = FxStableHloExport.traceWithValues(
                pythonSource, className, inputSpecs, seed);

        if (result.success) {
            try {
                FxStableHloExport.writeResult(result, outputDir);
                System.out.println("SUCCESS: Trace with values completed");
                System.out.println();
                System.out.println("Output files:");
                System.out.println("  MLIR:     " + outputDir.resolve("model.mlir"));
                System.out.println("  Manifest: " + outputDir.resolve("manifest.json"));
                System.out.println("  Inputs:   " + outputDir.resolve("inputs/"));
                for (int i = 0; i < result.inputTensorsNpy.size(); i++) {
                    System.out.println("            - input_" + i + ".npy");
                }
                System.out.println("  Outputs:  " + outputDir.resolve("outputs/"));
                for (int i = 0; i < result.outputTensorsNpy.size(); i++) {
                    System.out.println("            - output_" + i + ".npy");
                }
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
        if (!checkLicenseForTrace()) {
            return;
        }

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

    private static boolean checkLicenseForTrace() {
        LicenseCheckResult result = LicenseManager.getInstance().checkLicense();
        if (!result.allowed()) {
            System.err.println(result.message());
            System.exit(1);
            return false;
        }
        if (result.warning() != null) {
            System.err.println("Warning: " + result.warning());
        }
        return true;
    }

    private static void runActivate(String[] args) {
        String licenseKey = null;
        for (int i = 0; i < args.length; i++) {
            if ("--activate".equals(args[i]) && i + 1 < args.length) {
                licenseKey = args[i + 1];
            }
        }

        if (licenseKey == null || licenseKey.startsWith("--")) {
            System.err.println("Error: --activate requires a license key");
            System.err.println("Usage: snakegrinder --activate YOUR_LICENSE_KEY");
            System.exit(2);
            return;
        }

        System.out.println("Activating license...");
        ActivationResult result = LicenseManager.getInstance().activate(licenseKey);

        if (result.success()) {
            LicenseInfo license = result.license();
            System.out.println("License activated successfully!");
            System.out.println();
            System.out.println("Product: " + license.product().getDisplayName());
            if (license.validUntil() != null) {
                System.out.println("Valid until: " + license.validUntil());
            }
            System.out.println("Machine: " + LicenseManager.getInstance().getMachineName());
        } else {
            System.err.println("Activation failed: " + result.error());
            System.exit(1);
        }
    }

    private static void runDeactivate() {
        LicenseInfo current = LicenseManager.getInstance().getCurrentLicense();
        if (current == null) {
            System.out.println("No license is currently active.");
            return;
        }

        System.out.println("Deactivating license...");
        LicenseManager.getInstance().deactivate();
        System.out.println("License deactivated.");
    }

    private static void runLicenseInfo() {
        LicenseInfo license = LicenseManager.getInstance().getCurrentLicense();

        System.out.println("SnakeGrinder License Info");
        System.out.println();

        if (license == null) {
            System.out.println("Status: No license (Free tier)");
            System.out.println();
            System.out.println("To activate a license:");
            System.out.println("  snakegrinder --activate YOUR_LICENSE_KEY");
            System.out.println();
            System.out.println("Get a license at: https://surfworks.energy/pricing");
        } else {
            System.out.println("Product:     " + license.product().getDisplayName());
            System.out.println("Status:      " + (license.isExpired() ? "Expired" : "Active"));
            if (license.validUntil() != null) {
                System.out.println("Valid until: " + license.validUntil());
            }
            if (license.customerEmail() != null) {
                System.out.println("Email:       " + license.customerEmail());
            }
            System.out.println("Machine:     " + LicenseManager.getInstance().getMachineName());
            System.out.println("Fingerprint: " + LicenseManager.getInstance().getMachineFingerprint());
        }
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
        System.out.println("  snakegrinder --trace-with-values --source <file> --class <name> --inputs <specs> [--seed <n>] [--out <dir>]");
        System.out.println("  snakegrinder --trace-example [--out <dir>]");
        System.out.println("  snakegrinder --pytorch-info");
        System.out.println("  snakegrinder --self-test");
        System.out.println("  snakegrinder --help");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  --trace              Trace a PyTorch nn.Module and convert to StableHLO MLIR");
        System.out.println("  --trace-with-values  Trace and capture tensor values (for E2E testing)");
        System.out.println("  --trace-example      Run a built-in example (SimpleMLP)");
        System.out.println("  --pytorch-info       Show PyTorch version and capabilities");
        System.out.println("  --self-test          Run GraalPy self-test");
        System.out.println();
        System.out.println("Trace options:");
        System.out.println("  --source <file>  Python source file containing nn.Module class");
        System.out.println("  --class <name>   nn.Module class name to trace");
        System.out.println("  --inputs <specs> Input shapes, e.g., '[(1,8)]' or '[(1,8),(8,16)]'");
        System.out.println("  --out <dir>      Output directory (default: build/snakegrinder/trace)");
        System.out.println("  --seed <n>       Random seed for reproducible inputs (default: 42)");
        System.out.println();
        System.out.println("License commands:");
        System.out.println("  --activate <key>     Activate a license key");
        System.out.println("  --deactivate         Deactivate current license");
        System.out.println("  --license-info       Show license information");
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  snakegrinder --trace --source model.py --class MyModel --inputs '[(1,32)]'");
        System.out.println("  snakegrinder --trace-with-values --source model.py --class MyModel --inputs '[(1,8)]' --seed 42");
        System.out.println("  snakegrinder --activate XXXX-XXXX-XXXX-XXXX");
        System.out.println();
        System.out.println("Output:");
        System.out.println("  --trace:             model.mlir, manifest.json");
        System.out.println("  --trace-with-values: model.mlir, manifest.json, inputs/*.npy, outputs/*.npy");
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
