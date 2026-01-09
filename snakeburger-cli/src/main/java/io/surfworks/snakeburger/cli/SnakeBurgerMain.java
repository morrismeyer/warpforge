package io.surfworks.snakeburger.cli;

import io.surfworks.snakeburger.core.BabylonHello;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import io.surfworks.snakeburger.stablehlo.StableHloParser;
import io.surfworks.snakeburger.stablehlo.StableHloToBabylon;
import io.surfworks.snakeburger.stablehlo.StableHloTypeChecker;

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
            case "--hello" -> runHello();
            case "--stablehlo-ingest" -> runStableHloIngest(args);
            case "--stablehlo-example" -> runStableHloExample();
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
        System.out.println("  --hello                  Print Babylon hello world example");
        System.out.println("  --stablehlo-ingest FILE  Parse StableHLO MLIR and emit Babylon Op tree");
        System.out.println("  --stablehlo-example      Parse and emit built-in MLP example");
        System.out.println("  --help, -h               Print this help message");
    }

    private static void runHello() {
        System.out.println("hello world babylon");
        System.out.println();
        System.out.println(BabylonHello.helloModelText());
    }

    private static void runStableHloIngest(String[] args) {
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
}
