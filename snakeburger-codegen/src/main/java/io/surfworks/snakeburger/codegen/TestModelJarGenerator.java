package io.surfworks.snakeburger.codegen;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloParser;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Generates test model JARs from StableHLO MLIR.
 *
 * <p>This is a build-time utility that runs on Babylon JDK to generate
 * model JARs that can later be loaded by Espresso smoke tests running
 * on GraalVM.
 *
 * <p>Usage: java TestModelJarGenerator <output-directory>
 */
public final class TestModelJarGenerator {

    // Simple add model: %0 = add(%arg0, %arg1)
    private static final String ADD_MODEL_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
            stablehlo.return %0 : tensor<2x2xf32>
          }
        }
        """;

    // Multiply-add model: %0 = mul(%arg0, %arg1), %1 = add(%0, %arg2)
    private static final String MULADD_MODEL_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
            %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x2xf32>
            %1 = stablehlo.add %0, %arg2 : tensor<2x2xf32>
            stablehlo.return %1 : tensor<2x2xf32>
          }
        }
        """;

    // Subtract model for testing more operations
    private static final String SUB_MODEL_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xf32>
            stablehlo.return %0 : tensor<4xf32>
          }
        }
        """;

    // Negate model (unary operation)
    private static final String NEGATE_MODEL_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<3x3xf32>) -> (tensor<3x3xf32>) {
            %0 = stablehlo.negate %arg0 : tensor<3x3xf32>
            stablehlo.return %0 : tensor<3x3xf32>
          }
        }
        """;

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: TestModelJarGenerator <output-directory>");
            System.exit(1);
        }

        Path outputDir = Path.of(args[0]);
        Files.createDirectories(outputDir);

        System.out.println("Generating test model JARs to: " + outputDir);

        // Generate each model
        generateModel(ADD_MODEL_MLIR, "AddModel", outputDir);
        generateModel(MULADD_MODEL_MLIR, "MulAddModel", outputDir);
        generateModel(SUB_MODEL_MLIR, "SubModel", outputDir);
        generateModel(NEGATE_MODEL_MLIR, "NegateModel", outputDir);

        System.out.println("Successfully generated " + 4 + " test model JARs");
    }

    private static void generateModel(String mlir, String modelName, Path outputDir) throws Exception {
        System.out.println("  Generating " + modelName + "...");

        StableHloAst.Module module = StableHloParser.parse(mlir);
        StableHloAst.Function function = module.functions().getFirst();

        String className = "io.surfworks.warpforge.generated." + modelName;
        var generated = ModelClassGenerator.generate(className, function, mlir);

        Path jarPath = outputDir.resolve(modelName.toLowerCase() + ".jar");
        ModelJarBuilder.build(jarPath, className, generated.bytecode(), generated.metadata(), mlir);

        if (!Files.exists(jarPath)) {
            throw new RuntimeException("Failed to generate JAR: " + jarPath);
        }

        System.out.println("    -> " + jarPath.getFileName() + " (" + Files.size(jarPath) + " bytes)");
    }
}
