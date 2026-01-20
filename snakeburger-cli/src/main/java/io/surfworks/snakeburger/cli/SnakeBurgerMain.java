package io.surfworks.snakeburger.cli;

import io.surfworks.snakeburger.cli.BabylonVersion;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import io.surfworks.snakeburger.stablehlo.StableHloParser;
import io.surfworks.snakeburger.stablehlo.StableHloToBabylon;
import io.surfworks.snakeburger.stablehlo.StableHloTypeChecker;
import io.surfworks.warpforge.license.ActivationResult;
import io.surfworks.warpforge.license.LicenseCheckResult;
import io.surfworks.warpforge.license.LicenseInfo;
import io.surfworks.warpforge.license.LicenseManager;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public final class SnakeBurgerMain {

    public static void main(String[] args) {
        if (args.length == 0) {
            printUsage();
            return;
        }

        String command = args[0];
        switch (command) {
            case "--help", "-h" -> printUsage();
            case "--babylon" -> runBabylon();
            case "--stablehlo-ingest" -> runStableHloIngest(args);
            case "--stablehlo-example" -> runStableHloExample();
            case "--activate" -> runActivate(args);
            case "--deactivate" -> runDeactivate();
            case "--license-info" -> runLicenseInfo();
            default -> {
                System.err.println("Unknown command: " + command);
                printUsage();
                System.exit(1);
            }
        }
    }

    private static void printUsage() {
        System.out.println("SnakeBurger CLI - Babylon Code Reflection Tools");
        System.out.println();
        System.out.println("Usage: snakeburger <command> [options]");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  --babylon                Verify Babylon code reflection and show version");
        System.out.println("  --stablehlo-ingest FILE  Parse StableHLO MLIR and emit Babylon Op tree");
        System.out.println("  --stablehlo-example      Parse and emit built-in MLP example");
        System.out.println("  --help, -h               Print this help message");
        System.out.println();
        System.out.println("License commands:");
        System.out.println("  --activate KEY           Activate a license key");
        System.out.println("  --deactivate             Deactivate current license");
        System.out.println("  --license-info           Show license information");
    }

    private static void runBabylon() {
        System.out.println("Babylon Code Reflection Status");
        System.out.println("==============================");
        System.out.println();
        System.out.println("Available: " + BabylonVersion.isAvailable());
        System.out.println();
        System.out.println("Code model for reflected method:");
        System.out.println(BabylonVersion.getCodeModelText());
    }

    private static void runStableHloIngest(String[] args) {
        if (!checkLicenseForProcessing()) {
            return;
        }

        if (args.length < 2) {
            System.err.println("Error: --stablehlo-ingest requires a FILE argument");
            System.exit(1);
        }

        Path inputPath = Path.of(args[1]);
        if (!Files.exists(inputPath)) {
            System.err.println("Error: File not found: " + inputPath);
            System.exit(1);
        }

        try {
            String mlirText = Files.readString(inputPath);
            processStableHlo(mlirText, inputPath.toString());
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            System.exit(1);
        }
    }

    private static void runStableHloExample() {
        if (!checkLicenseForProcessing()) {
            return;
        }

        // Built-in MLP example from the plan
        String exampleMlir = """
            module @main {
              func.func public @forward(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                %1 = stablehlo.constant dense<0.0> : tensor<4x16xf32>
                %2 = stablehlo.maximum %0, %1 : tensor<4x16xf32>
                stablehlo.return %2 : tensor<4x16xf32>
              }
            }
            """;

        System.out.println("=== StableHLO Input ===");
        System.out.println(exampleMlir);

        processStableHlo(exampleMlir, "<example>");
    }

    private static void processStableHlo(String mlirText, String sourceName) {
        try {
            // Parse
            System.out.println("=== Parsing StableHLO ===");
            Module module = StableHloParser.parse(mlirText);
            System.out.println("Parsed module: @" + module.name());
            System.out.println("Functions: " + module.functions().size());
            for (var func : module.functions()) {
                System.out.println("  - @" + func.name() + " (" + func.arguments().size() + " args)");
            }
            System.out.println();

            // Type check
            System.out.println("=== Type Checking ===");
            StableHloTypeChecker checker = new StableHloTypeChecker();
            List<String> errors = checker.validate(module);
            if (errors.isEmpty()) {
                System.out.println("Type check passed!");
            } else {
                System.out.println("Type check failed:");
                for (String error : errors) {
                    System.out.println("  - " + error);
                }
                System.exit(1);
            }
            System.out.println();

            // Emit Babylon Op tree
            System.out.println("=== Babylon Op Tree ===");
            StableHloToBabylon emitter = new StableHloToBabylon();
            var result = emitter.emit(module);
            System.out.println(result.babylonText());

        } catch (Exception e) {
            System.err.println("Error processing " + sourceName + ": " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static boolean checkLicenseForProcessing() {
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
        if (args.length < 2) {
            System.err.println("Error: --activate requires a license key");
            System.err.println("Usage: snakeburger --activate YOUR_LICENSE_KEY");
            System.exit(2);
            return;
        }

        String licenseKey = args[1];
        if (licenseKey.startsWith("--")) {
            System.err.println("Error: --activate requires a license key");
            System.err.println("Usage: snakeburger --activate YOUR_LICENSE_KEY");
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

        System.out.println("SnakeBurger License Info");
        System.out.println();

        if (license == null) {
            System.out.println("Status: No license (Free tier)");
            System.out.println();
            System.out.println("To activate a license:");
            System.out.println("  snakeburger --activate YOUR_LICENSE_KEY");
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
}
