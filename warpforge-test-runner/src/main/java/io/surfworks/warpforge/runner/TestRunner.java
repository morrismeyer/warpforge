package io.surfworks.warpforge.runner;

import io.surfworks.warpforge.codegen.api.CompiledModel;
import io.surfworks.warpforge.codegen.api.ModelLoader;
import io.surfworks.warpforge.core.backend.Backend;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.backend.cpu.CpuBackend;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Test runner for executing compiled WarpForge models.
 *
 * <p>Usage:
 * <pre>
 * warpforge-test-runner --jar model.jar [--backend cpu|nvidia|amd]
 * </pre>
 *
 * <p>This runner loads a compiled model JAR and executes it with the specified backend.
 * It's designed for integration testing in the hardware CI pipeline.
 */
public final class TestRunner {

    public static void main(String[] args) {
        try {
            RunnerArgs parsedArgs = parseArgs(args);
            run(parsedArgs);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            if (System.getenv("DEBUG") != null) {
                e.printStackTrace();
            }
            System.exit(1);
        }
    }

    /**
     * Run the test with parsed arguments.
     */
    public static void run(RunnerArgs args) throws Exception {
        System.out.println("WarpForge Test Runner");
        System.out.println("  Model JAR: " + args.jarPath);
        System.out.println("  Backend: " + args.backend);

        // Load the model
        CompiledModel model = ModelLoader.load(args.jarPath);
        System.out.println("\nModel loaded:");
        System.out.println("  Name: " + model.metadata().name());
        System.out.println("  Inputs: " + model.inputCount());
        System.out.println("  Outputs: " + model.outputCount());
        System.out.println("  Generator: " + model.metadata().generatorVersion());
        System.out.println("  Source hash: " + model.metadata().sourceHash().substring(0, 16) + "...");

        // Create backend
        try (Backend backend = createBackend(args.backend)) {
            System.out.println("\nBackend initialized: " + backend.name());

            // Generate test inputs (zeros for now - in real use, would load from file)
            List<Tensor> inputs = new ArrayList<>();
            for (int i = 0; i < model.inputCount(); i++) {
                // Default to 2x2 f32 tensors for testing
                // In a real runner, we'd parse input specs from the model
                inputs.add(Tensor.zeros(2, 2));
            }

            System.out.println("\nExecuting model...");
            long startTime = System.nanoTime();

            List<Tensor> outputs = model.forward(inputs, backend);

            long endTime = System.nanoTime();
            double durationMs = (endTime - startTime) / 1_000_000.0;

            System.out.println("\nExecution complete:");
            System.out.println("  Duration: " + String.format("%.3f", durationMs) + " ms");
            System.out.println("  Outputs: " + outputs.size());

            for (int i = 0; i < outputs.size(); i++) {
                Tensor output = outputs.get(i);
                System.out.println("  Output " + i + ":");
                System.out.println("    Shape: " + java.util.Arrays.toString(output.shape()));
                System.out.println("    Dtype: " + output.dtype());
                if (output.elementCount() <= 16) {
                    System.out.println("    Data: " + java.util.Arrays.toString(output.toFloatArray()));
                }
            }

            // Cleanup
            for (Tensor input : inputs) {
                input.close();
            }
            for (Tensor output : outputs) {
                output.close();
            }
        }

        System.out.println("\nTest completed successfully.");
    }

    private static Backend createBackend(String backendName) {
        return switch (backendName.toLowerCase()) {
            case "cpu" -> new CpuBackend();
            case "nvidia" -> loadBackend("io.surfworks.warpforge.backend.nvidia.NvidiaBackend");
            case "amd" -> loadBackend("io.surfworks.warpforge.backend.amd.AmdBackend");
            default -> throw new IllegalArgumentException("Unknown backend: " + backendName);
        };
    }

    @SuppressWarnings("unchecked")
    private static Backend loadBackend(String className) {
        try {
            Class<?> clazz = Class.forName(className);
            return (Backend) clazz.getDeclaredConstructor().newInstance();
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException(
                "Backend class not found: " + className + ". Make sure the backend JAR is on the classpath.");
        } catch (Exception e) {
            throw new IllegalStateException("Failed to instantiate backend: " + className, e);
        }
    }

    private static RunnerArgs parseArgs(String[] args) {
        Path jarPath = null;
        String backend = "cpu";

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--jar", "-j" -> {
                    if (++i >= args.length) throw new IllegalArgumentException("Missing value for --jar");
                    jarPath = Path.of(args[i]);
                }
                case "--backend", "-b" -> {
                    if (++i >= args.length) throw new IllegalArgumentException("Missing value for --backend");
                    backend = args[i];
                }
                case "--help", "-h" -> {
                    printUsage();
                    System.exit(0);
                }
                default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
            }
        }

        if (jarPath == null) {
            throw new IllegalArgumentException("--jar is required");
        }

        return new RunnerArgs(jarPath, backend);
    }

    private static void printUsage() {
        System.out.println("WarpForge Test Runner");
        System.out.println();
        System.out.println("Usage: warpforge-test-runner --jar <model.jar> [options]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --jar, -j <path>      Path to compiled model JAR (required)");
        System.out.println("  --backend, -b <name>  Backend to use: cpu, nvidia, amd (default: cpu)");
        System.out.println("  --help, -h            Show this help");
    }

    /**
     * Parsed runner arguments.
     */
    public record RunnerArgs(
        Path jarPath,
        String backend
    ) {}
}
