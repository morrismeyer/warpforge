package io.surfworks.warpforge.runner;

import io.surfworks.warpforge.codegen.api.CompiledModel;
import io.surfworks.warpforge.codegen.api.ModelLoader;
import io.surfworks.warpforge.codegen.api.ModelMetadata;
import io.surfworks.warpforge.core.backend.Backend;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.backend.cpu.CpuBackend;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Test runner for executing compiled WarpForge models.
 *
 * <p>Supports three execution modes:
 * <ul>
 *   <li><b>JVM</b>: Standard class loading (default)</li>
 *   <li><b>Espresso</b>: Guest JVM in native-image</li>
 *   <li><b>Native</b>: Fully AOT-compiled (future)</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>
 * warpforge-test-runner --jar model.jar [--mode jvm|espresso|native] [--backend cpu|nvidia|amd]
 * </pre>
 */
public final class TestRunner {

    public static void main(String[] args) {
        try {
            RunnerArgs parsedArgs = parseArgs(args);
            RunResult result = run(parsedArgs);
            System.exit(result.success() ? 0 : 1);
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
     *
     * @return The result of the run including outputs for verification
     */
    public static RunResult run(RunnerArgs args) throws Exception {
        System.out.println("WarpForge Test Runner");
        System.out.println("  Model JAR: " + args.jarPath);
        System.out.println("  Mode: " + args.mode);
        System.out.println("  Backend: " + args.backend);

        if (!args.mode.isSupported()) {
            throw new UnsupportedOperationException(
                "Execution mode " + args.mode + " is not yet supported");
        }

        // Load the model using the appropriate mode
        ModelHandle model = loadModel(args.jarPath, args.mode);

        System.out.println("\nModel loaded:");
        System.out.println("  Name: " + model.metadata().name());
        System.out.println("  Inputs: " + model.inputCount());
        System.out.println("  Outputs: " + model.outputCount());
        System.out.println("  Generator: " + model.metadata().generatorVersion());
        System.out.println("  Source hash: " + truncateHash(model.metadata().sourceHash()));

        // Create backend
        try (Backend backend = createBackend(args.backend)) {
            System.out.println("\nBackend initialized: " + backend.name());

            // Generate test inputs
            List<Tensor> inputs = createTestInputs(model.inputCount(), args.inputShape);

            System.out.println("\nExecuting model...");
            long startTime = System.nanoTime();

            List<Tensor> outputs = model.forward(inputs, backend);

            long endTime = System.nanoTime();
            double durationMs = (endTime - startTime) / 1_000_000.0;

            System.out.println("\nExecution complete:");
            System.out.println("  Duration: " + String.format("%.3f", durationMs) + " ms");
            System.out.println("  Outputs: " + outputs.size());

            // Collect output data for verification
            List<float[]> outputData = new ArrayList<>();
            for (int i = 0; i < outputs.size(); i++) {
                Tensor output = outputs.get(i);
                System.out.println("  Output " + i + ":");
                System.out.println("    Shape: " + Arrays.toString(output.shape()));
                System.out.println("    Dtype: " + output.dtype());

                float[] data = output.toFloatArray();
                outputData.add(data);

                if (output.elementCount() <= 16) {
                    System.out.println("    Data: " + Arrays.toString(data));
                }
            }

            // Cleanup
            for (Tensor input : inputs) {
                input.close();
            }
            for (Tensor output : outputs) {
                output.close();
            }

            // Close model handle (important for Espresso mode)
            model.close();

            System.out.println("\nTest completed successfully.");
            return new RunResult(true, outputData, durationMs);
        }
    }

    /**
     * Load a model using the specified execution mode.
     */
    private static ModelHandle loadModel(Path jarPath, ExecutionMode mode) throws Exception {
        return switch (mode) {
            case JVM -> {
                CompiledModel model = ModelLoader.load(jarPath);
                yield new JvmModelHandle(model);
            }
            case ESPRESSO -> {
                EspressoModelLoader loader = EspressoModelLoader.load(jarPath);
                yield new EspressoModelHandle(loader);
            }
            case NATIVE -> throw new UnsupportedOperationException(
                "Native mode requires the model to be compiled into the binary");
        };
    }

    /**
     * Create test inputs for the model.
     */
    private static List<Tensor> createTestInputs(int count, int[] shape) {
        List<Tensor> inputs = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            // Create tensors with predictable test data
            int[] actualShape = shape != null ? shape : new int[]{2, 2};
            Tensor tensor = Tensor.zeros(actualShape);

            // Fill with test pattern: input[i] has values starting at i*10
            float[] data = new float[(int) tensor.elementCount()];
            for (int j = 0; j < data.length; j++) {
                data[j] = i * 10 + j + 1;
            }
            tensor.copyFrom(data);

            inputs.add(tensor);
        }
        return inputs;
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

    private static String truncateHash(String hash) {
        return hash != null && hash.length() > 16 ? hash.substring(0, 16) + "..." : hash;
    }

    private static RunnerArgs parseArgs(String[] args) {
        Path jarPath = null;
        ExecutionMode mode = ExecutionMode.JVM;
        String backend = "cpu";
        int[] inputShape = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--jar", "-j" -> {
                    if (++i >= args.length) throw new IllegalArgumentException("Missing value for --jar");
                    jarPath = Path.of(args[i]);
                }
                case "--mode", "-m" -> {
                    if (++i >= args.length) throw new IllegalArgumentException("Missing value for --mode");
                    mode = ExecutionMode.fromString(args[i]);
                }
                case "--backend", "-b" -> {
                    if (++i >= args.length) throw new IllegalArgumentException("Missing value for --backend");
                    backend = args[i];
                }
                case "--shape", "-s" -> {
                    if (++i >= args.length) throw new IllegalArgumentException("Missing value for --shape");
                    inputShape = parseShape(args[i]);
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

        return new RunnerArgs(jarPath, mode, backend, inputShape);
    }

    private static int[] parseShape(String shapeStr) {
        String[] parts = shapeStr.split("[x,]");
        int[] shape = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            shape[i] = Integer.parseInt(parts[i].trim());
        }
        return shape;
    }

    private static void printUsage() {
        System.out.println("WarpForge Test Runner");
        System.out.println();
        System.out.println("Usage: warpforge-test-runner --jar <model.jar> [options]");
        System.out.println();
        System.out.println("Options:");
        System.out.println("  --jar, -j <path>      Path to compiled model JAR (required)");
        System.out.println("  --mode, -m <mode>     Execution mode: jvm, espresso, native (default: jvm)");
        System.out.println("  --backend, -b <name>  Backend to use: cpu, nvidia, amd (default: cpu)");
        System.out.println("  --shape, -s <dims>    Input tensor shape, e.g., 2x2 or 4,4 (default: 2x2)");
        System.out.println("  --help, -h            Show this help");
        System.out.println();
        System.out.println("Execution Modes:");
        System.out.println("  jvm      - Load model via standard class loading (default)");
        System.out.println("  espresso - Load model via Espresso in native-image");
        System.out.println("  native   - Fully AOT-compiled model (not yet supported)");
    }

    /**
     * Parsed runner arguments.
     */
    public record RunnerArgs(
        Path jarPath,
        ExecutionMode mode,
        String backend,
        int[] inputShape
    ) {}

    /**
     * Result of a test run, including outputs for verification.
     */
    public record RunResult(
        boolean success,
        List<float[]> outputs,
        double durationMs
    ) {}

    /**
     * Abstract handle for loaded models (supports different execution modes).
     */
    interface ModelHandle extends AutoCloseable {
        List<Tensor> forward(List<Tensor> inputs, Backend backend);
        int inputCount();
        int outputCount();
        ModelMetadata metadata();
    }

    /**
     * JVM mode model handle.
     */
    private static class JvmModelHandle implements ModelHandle {
        private final CompiledModel model;

        JvmModelHandle(CompiledModel model) {
            this.model = model;
        }

        @Override
        public List<Tensor> forward(List<Tensor> inputs, Backend backend) {
            return model.forward(inputs, backend);
        }

        @Override
        public int inputCount() {
            return model.inputCount();
        }

        @Override
        public int outputCount() {
            return model.outputCount();
        }

        @Override
        public ModelMetadata metadata() {
            return model.metadata();
        }

        @Override
        public void close() {
            // JVM mode doesn't need explicit cleanup
        }
    }

    /**
     * Espresso mode model handle.
     */
    private static class EspressoModelHandle implements ModelHandle {
        private final EspressoModelLoader loader;

        EspressoModelHandle(EspressoModelLoader loader) {
            this.loader = loader;
        }

        @Override
        public List<Tensor> forward(List<Tensor> inputs, Backend backend) {
            return loader.forward(inputs, backend);
        }

        @Override
        public int inputCount() {
            return loader.inputCount();
        }

        @Override
        public int outputCount() {
            return loader.outputCount();
        }

        @Override
        public ModelMetadata metadata() {
            return loader.metadata();
        }

        @Override
        public void close() {
            loader.close();
        }
    }
}
