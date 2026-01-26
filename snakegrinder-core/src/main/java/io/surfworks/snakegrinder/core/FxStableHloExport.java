package io.surfworks.snakegrinder.core;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.PolyglotException;
import org.graalvm.polyglot.Source;
import org.graalvm.polyglot.Value;

/**
 * Full-fidelity StableHLO export using real PyTorch and torch.fx.
 *
 * This class uses torch.fx.symbolic_trace to capture computation graphs
 * from PyTorch models with full fidelity (real shapes, dtypes, operators).
 *
 * Distribution: The native-image distribution bundles all dependencies.
 * End users receive a self-contained directory and run it directly.
 * No environment variables, no pip install, no configuration needed.
 *
 * Development: When running via Gradle (JVM mode), set PYTORCH_VENV to
 * point to the GraalPy virtualenv containing PyTorch. This is for
 * development/testing only and is not exposed to end users.
 */
public final class FxStableHloExport {

    private FxStableHloExport() {
    }

    /**
     * Result of an FX trace operation.
     */
    public static final class TraceResult {
        public final boolean success;
        public final String mlir;
        public final String fxGraph;
        public final String error;
        public final String traceback;
        public final List<String> warnings;
        public final Map<String, Object> metadata;

        // Extended fields for trace-with-values
        public final List<byte[]> inputTensorsNpy;
        public final List<byte[]> weightTensorsNpy;
        public final List<String> weightNames;
        public final List<byte[]> outputTensorsNpy;
        public final long seed;

        private TraceResult(boolean success, String mlir, String fxGraph, String error,
                            String traceback, List<String> warnings, Map<String, Object> metadata,
                            List<byte[]> inputTensorsNpy, List<byte[]> weightTensorsNpy,
                            List<String> weightNames, List<byte[]> outputTensorsNpy, long seed) {
            this.success = success;
            this.mlir = mlir;
            this.fxGraph = fxGraph;
            this.error = error;
            this.traceback = traceback;
            this.warnings = warnings != null ? warnings : List.of();
            this.metadata = metadata != null ? metadata : Map.of();
            this.inputTensorsNpy = inputTensorsNpy != null ? inputTensorsNpy : List.of();
            this.weightTensorsNpy = weightTensorsNpy != null ? weightTensorsNpy : List.of();
            this.weightNames = weightNames != null ? weightNames : List.of();
            this.outputTensorsNpy = outputTensorsNpy != null ? outputTensorsNpy : List.of();
            this.seed = seed;
        }

        public static TraceResult ok(String mlir, String fxGraph, Map<String, Object> metadata) {
            return new TraceResult(true, mlir, fxGraph, null, null, List.of(), metadata,
                                   null, null, null, null, 0);
        }

        public static TraceResult okWithValues(String mlir, Map<String, Object> metadata,
                                               List<byte[]> inputTensorsNpy,
                                               List<byte[]> weightTensorsNpy, List<String> weightNames,
                                               List<byte[]> outputTensorsNpy, long seed) {
            return new TraceResult(true, mlir, null, null, null, List.of(), metadata,
                                   inputTensorsNpy, weightTensorsNpy, weightNames, outputTensorsNpy, seed);
        }

        public static TraceResult fail(String error, String traceback) {
            return new TraceResult(false, null, null, error, traceback, List.of(), Map.of(),
                                   null, null, null, null, 0);
        }

        /** Check if this result includes tensor values. */
        public boolean hasValues() {
            return !inputTensorsNpy.isEmpty() || !outputTensorsNpy.isEmpty();
        }

        /** Check if this result includes model weights. */
        public boolean hasWeights() {
            return !weightTensorsNpy.isEmpty();
        }
    }

    /**
     * Input specification for a tensor argument.
     */
    public static final class InputSpec {
        public final int[] shape;
        public final String dtype;

        public InputSpec(int[] shape, String dtype) {
            this.shape = shape;
            this.dtype = dtype != null ? dtype : "f32";
        }

        public InputSpec(int[] shape) {
            this(shape, "f32");
        }

        public String toPythonTuple() {
            StringBuilder sb = new StringBuilder("((");
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(shape[i]);
            }
            if (shape.length == 1) sb.append(",");
            sb.append("), '").append(dtype).append("')");
            return sb.toString();
        }
    }

    /**
     * Trace a PyTorch model from source code using torch.fx.symbolic_trace.
     *
     * @param pythonSource Python source code containing the model definition
     * @param className    Name of the nn.Module class to trace
     * @param inputSpecs   Input tensor specifications (shapes and dtypes)
     * @return TraceResult containing StableHLO MLIR or error info
     */
    public static TraceResult trace(String pythonSource,
                                    String className,
                                    List<InputSpec> inputSpecs) {

        ByteArrayOutputStream pythonOut = new ByteArrayOutputStream();
        ByteArrayOutputStream pythonErr = new ByteArrayOutputStream();

        Context.Builder builder = Context.newBuilder("python")
                .allowAllAccess(true)
                .option("python.ForceImportSite", "true")
                .out(pythonOut)
                .err(pythonErr);

        // If PYTORCH_VENV is set, configure Python path
        String pytorchVenv = System.getenv("PYTORCH_VENV");
        if (pytorchVenv != null && !pytorchVenv.isBlank()) {
            builder.option("python.PythonPath", pytorchVenv + "/lib/python3.12/site-packages");
        }

        try (Context ctx = builder.build()) {

            // Load the FX-to-StableHLO converter module
            String fxModule = readResource("/snakegrinder/fx_to_stablehlo.py");
            ctx.eval("python", fxModule);

            // Build input_shapes as Python list of tuples with dtype
            // Format: [(dim1, dim2, ..., 'dtype'), ...]
            StringBuilder inputShapesPy = new StringBuilder("[");
            for (int i = 0; i < inputSpecs.size(); i++) {
                if (i > 0) inputShapesPy.append(", ");
                inputShapesPy.append("(");
                for (int j = 0; j < inputSpecs.get(i).shape.length; j++) {
                    inputShapesPy.append(inputSpecs.get(i).shape[j]);
                    inputShapesPy.append(", ");
                }
                // Add dtype as the last element
                inputShapesPy.append("'").append(inputSpecs.get(i).dtype).append("')");
            }
            inputShapesPy.append("]");

            // Escape the source code for Python
            String escapedSource = escapePythonMultilineString(pythonSource);

            // Call trace_model
            String callCode = String.format(
                "trace_model('''%s''', '%s', %s)",
                escapedSource, className, inputShapesPy.toString()
            );

            Value resultValue = ctx.eval("python", callCode);
            String mlir = resultValue.asString();

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("class", className);
            metadata.put("tracer", "torch.fx.symbolic_trace");
            metadata.put("input_count", inputSpecs.size());
            metadata.put("timestamp", Instant.now().toString());

            return TraceResult.ok(mlir, null, metadata);

        } catch (PolyglotException e) {
            String error = "GraalPy error: " + e.getMessage();
            String stderr = pythonErr.toString(StandardCharsets.UTF_8);
            if (!stderr.isBlank()) {
                error += "\nPython stderr:\n" + stderr;
            }

            // Check for common issues
            if (e.getMessage().contains("No module named 'torch'")) {
                error += "\n\nPyTorch is not installed in the GraalPy environment.\n" +
                        "Please ensure PyTorch 2.7+ is built for GraalPy and installed.";
            } else if (e.getMessage().contains("dlopen") || e.getMessage().contains("libgomp")) {
                error += "\n\nNative library loading failed.\n" +
                        "Set DYLD_LIBRARY_PATH (macOS) or LD_LIBRARY_PATH (Linux) to include:\n" +
                        "  <graalpy-venv>/lib/python3.12/site-packages/torch/lib/";
            }

            return TraceResult.fail(error, null);
        } catch (Exception e) {
            return TraceResult.fail("Trace failed: " + e.getMessage(), null);
        }
    }

    /**
     * Trace a PyTorch model and capture actual tensor values for E2E verification.
     *
     * <p>This method runs the model forward pass with deterministic inputs
     * and captures both the graph (MLIR) and the actual tensor values.
     *
     * @param pythonSource Python source code containing the model definition
     * @param className    Name of the nn.Module class to trace
     * @param inputSpecs   Input tensor specifications (shapes and dtypes)
     * @param seed         Random seed for reproducible inputs
     * @return TraceResult containing StableHLO MLIR and tensor data
     */
    public static TraceResult traceWithValues(String pythonSource,
                                              String className,
                                              List<InputSpec> inputSpecs,
                                              long seed) {

        ByteArrayOutputStream pythonOut = new ByteArrayOutputStream();
        ByteArrayOutputStream pythonErr = new ByteArrayOutputStream();

        Context.Builder builder = Context.newBuilder("python")
                .allowAllAccess(true)
                .option("python.ForceImportSite", "true")
                .out(pythonOut)
                .err(pythonErr);

        // If PYTORCH_VENV is set, configure Python path
        String pytorchVenv = System.getenv("PYTORCH_VENV");
        if (pytorchVenv != null && !pytorchVenv.isBlank()) {
            builder.option("python.PythonPath", pytorchVenv + "/lib/python3.12/site-packages");
        }

        try (Context ctx = builder.build()) {

            // Load the FX-to-StableHLO converter module
            String fxModule = readResource("/snakegrinder/fx_to_stablehlo.py");
            ctx.eval("python", fxModule);

            // Build input_shapes as Python list of tuples with dtype
            // Format: [(dim1, dim2, ..., 'dtype'), ...]
            StringBuilder inputShapesPy = new StringBuilder("[");
            for (int i = 0; i < inputSpecs.size(); i++) {
                if (i > 0) inputShapesPy.append(", ");
                inputShapesPy.append("(");
                for (int j = 0; j < inputSpecs.get(i).shape.length; j++) {
                    inputShapesPy.append(inputSpecs.get(i).shape[j]);
                    inputShapesPy.append(", ");
                }
                // Add dtype as the last element
                inputShapesPy.append("'").append(inputSpecs.get(i).dtype).append("')");
            }
            inputShapesPy.append("]");

            // Escape the source code for Python
            String escapedSource = escapePythonMultilineString(pythonSource);

            // Call trace_with_values_npy to get tensor data as .npy bytes
            String callCode = String.format(
                "trace_with_values_npy('''%s''', '%s', %s, %d)",
                escapedSource, className, inputShapesPy.toString(), seed
            );

            Value resultValue = ctx.eval("python", callCode);

            // Extract results - use getHashValue for Python dict access
            if (resultValue == null) {
                return TraceResult.fail("Python returned null result", null);
            }
            if (!resultValue.hasHashEntries()) {
                return TraceResult.fail("Python result is not a dict: " + resultValue.toString(), null);
            }
            Value mlirValue = resultValue.getHashValue("mlir");
            if (mlirValue == null || mlirValue.isNull()) {
                return TraceResult.fail("Python result missing 'mlir' key", null);
            }
            String mlir = mlirValue.asString();
            long returnedSeed = resultValue.getHashValue("seed").asLong();

            // Extract input tensor .npy bytes
            Value inputNpyList = resultValue.getHashValue("input_npy");
            List<byte[]> inputTensorsNpy = new ArrayList<>();
            for (int i = 0; i < inputNpyList.getArraySize(); i++) {
                Value pyBytes = inputNpyList.getArrayElement(i);
                byte[] npyBytes = extractBytes(pyBytes);
                inputTensorsNpy.add(npyBytes);
            }

            // Extract weight tensor .npy bytes
            Value weightNpyList = resultValue.getHashValue("weight_npy");
            List<byte[]> weightTensorsNpy = new ArrayList<>();
            if (weightNpyList != null && !weightNpyList.isNull() && weightNpyList.hasArrayElements()) {
                for (int i = 0; i < weightNpyList.getArraySize(); i++) {
                    Value pyBytes = weightNpyList.getArrayElement(i);
                    byte[] npyBytes = extractBytes(pyBytes);
                    weightTensorsNpy.add(npyBytes);
                }
            }

            // Extract weight names
            Value weightNamesList = resultValue.getHashValue("weight_names");
            List<String> weightNames = new ArrayList<>();
            if (weightNamesList != null && !weightNamesList.isNull() && weightNamesList.hasArrayElements()) {
                for (int i = 0; i < weightNamesList.getArraySize(); i++) {
                    weightNames.add(weightNamesList.getArrayElement(i).asString());
                }
            }

            // Extract output tensor .npy bytes
            Value outputNpyList = resultValue.getHashValue("output_npy");
            List<byte[]> outputTensorsNpy = new ArrayList<>();
            for (int i = 0; i < outputNpyList.getArraySize(); i++) {
                Value pyBytes = outputNpyList.getArrayElement(i);
                byte[] npyBytes = extractBytes(pyBytes);
                outputTensorsNpy.add(npyBytes);
            }

            // Extract shapes for metadata
            Value inputShapesList = resultValue.getMember("input_shapes");
            Value outputShapesList = resultValue.getMember("output_shapes");

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("class", className);
            metadata.put("tracer", "torch.fx.symbolic_trace");
            metadata.put("input_count", inputSpecs.size());
            metadata.put("weight_count", weightTensorsNpy.size());
            metadata.put("output_count", outputTensorsNpy.size());
            metadata.put("seed", returnedSeed);
            metadata.put("timestamp", Instant.now().toString());

            return TraceResult.okWithValues(mlir, metadata, inputTensorsNpy,
                                           weightTensorsNpy, weightNames, outputTensorsNpy, returnedSeed);

        } catch (PolyglotException e) {
            String error = "GraalPy error: " + e.getMessage();
            String stderr = pythonErr.toString(StandardCharsets.UTF_8);
            if (!stderr.isBlank()) {
                error += "\nPython stderr:\n" + stderr;
            }

            if (e.getMessage().contains("No module named 'torch'")) {
                error += "\n\nPyTorch is not installed in the GraalPy environment.\n" +
                        "Please ensure PyTorch 2.7+ is built for GraalPy and installed.";
            } else if (e.getMessage().contains("dlopen") || e.getMessage().contains("libgomp")) {
                error += "\n\nNative library loading failed.\n" +
                        "Set DYLD_LIBRARY_PATH (macOS) or LD_LIBRARY_PATH (Linux) to include:\n" +
                        "  <graalpy-venv>/lib/python3.12/site-packages/torch/lib/";
            }

            return TraceResult.fail(error, null);
        } catch (Exception e) {
            String msg = e.getMessage();
            if (msg == null) {
                msg = e.getClass().getName();
            }
            // Include stack trace for debugging
            java.io.StringWriter sw = new java.io.StringWriter();
            e.printStackTrace(new java.io.PrintWriter(sw));
            return TraceResult.fail("Trace with values failed: " + msg, sw.toString());
        }
    }

    /**
     * Extract bytes from a Python bytes object.
     */
    private static byte[] extractBytes(Value pyBytes) {
        if (pyBytes.hasBufferElements()) {
            int size = (int) pyBytes.getBufferSize();
            byte[] bytes = new byte[size];
            for (int i = 0; i < size; i++) {
                bytes[i] = pyBytes.readBufferByte(i);
            }
            return bytes;
        } else if (pyBytes.hasArrayElements()) {
            int size = (int) pyBytes.getArraySize();
            byte[] bytes = new byte[size];
            for (int i = 0; i < size; i++) {
                bytes[i] = (byte) pyBytes.getArrayElement(i).asInt();
            }
            return bytes;
        }
        throw new IllegalArgumentException("Cannot extract bytes from Python value: " + pyBytes);
    }

    /**
     * Trace the built-in MLP example.
     */
    public static TraceResult traceMlpExample() {
        ByteArrayOutputStream pythonOut = new ByteArrayOutputStream();
        ByteArrayOutputStream pythonErr = new ByteArrayOutputStream();

        Context.Builder builder = Context.newBuilder("python")
                .allowAllAccess(true)
                .option("python.ForceImportSite", "true")
                .out(pythonOut)
                .err(pythonErr);

        String pytorchVenv = System.getenv("PYTORCH_VENV");
        if (pytorchVenv != null && !pytorchVenv.isBlank()) {
            builder.option("python.PythonPath", pytorchVenv + "/lib/python3.12/site-packages");
        }

        try (Context ctx = builder.build()) {

            // Load the FX-to-StableHLO converter module
            String fxModule = readResource("/snakegrinder/fx_to_stablehlo.py");
            ctx.eval("python", fxModule);

            // Call the built-in example function
            Value result = ctx.eval("python", "trace_builtin_example()");
            String mlir = result.asString();

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("example", "SimpleMLP");
            metadata.put("tracer", "torch.fx.symbolic_trace");
            metadata.put("timestamp", Instant.now().toString());

            return TraceResult.ok(mlir, null, metadata);

        } catch (PolyglotException e) {
            String error = "GraalPy error: " + e.getMessage();
            String stderr = pythonErr.toString(StandardCharsets.UTF_8);

            if (e.getMessage().contains("No module named 'torch'")) {
                error = "PyTorch is not installed in the GraalPy environment.\n" +
                        "The --trace command requires PyTorch 2.7+ built for GraalPy.";
            } else if (e.getMessage().contains("dlopen") || e.getMessage().contains("library")) {
                error = "Native library loading failed.\n" +
                        "Set DYLD_LIBRARY_PATH (macOS) or LD_LIBRARY_PATH (Linux) to:\n" +
                        "  <graalpy-venv>/lib/python3.12/site-packages/torch/lib/\n\n" +
                        "Original error: " + e.getMessage();
            }

            if (!stderr.isBlank()) {
                error += "\nPython stderr:\n" + stderr;
            }
            return TraceResult.fail(error, null);
        } catch (Exception e) {
            return TraceResult.fail("Example trace failed: " + e.getMessage(), null);
        }
    }

    /**
     * Get PyTorch info from the GraalPy environment.
     */
    public static Map<String, Object> getPyTorchInfo() {
        Context.Builder builder = Context.newBuilder("python")
                .allowAllAccess(true)
                .option("python.ForceImportSite", "true");

        String pytorchVenv = System.getenv("PYTORCH_VENV");
        if (pytorchVenv != null && !pytorchVenv.isBlank()) {
            builder.option("python.PythonPath", pytorchVenv + "/lib/python3.12/site-packages");
        }

        try (Context ctx = builder.build()) {
            // Check PyTorch availability
            String checkCode = """
                import torch
                result = {
                    'pytorch_version': torch.__version__,
                    'cuda_available': str(torch.cuda.is_available()),
                    'mps_available': str(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
                    'fx_available': 'True'
                }
                result
                """;
            Value result = ctx.eval("python", checkCode);
            return valueToMap(result);

        } catch (Exception e) {
            Map<String, Object> errorInfo = new LinkedHashMap<>();
            errorInfo.put("error", e.getMessage());
            errorInfo.put("pytorch_available", false);
            return errorInfo;
        }
    }

    /**
     * Write trace result to output directory.
     */
    public static void writeResult(TraceResult result, Path outputDir) throws IOException {
        Files.createDirectories(outputDir);

        if (result.success && result.mlir != null) {
            Files.writeString(outputDir.resolve("model.mlir"), result.mlir);
        }

        if (result.fxGraph != null) {
            Files.writeString(outputDir.resolve("fx_graph.txt"), result.fxGraph);
        }

        // Write input tensor .npy files if present
        List<String> inputPaths = new ArrayList<>();
        if (result.hasValues() && !result.inputTensorsNpy.isEmpty()) {
            Path inputsDir = outputDir.resolve("inputs");
            Files.createDirectories(inputsDir);
            for (int i = 0; i < result.inputTensorsNpy.size(); i++) {
                String filename = "input_" + i + ".npy";
                Files.write(inputsDir.resolve(filename), result.inputTensorsNpy.get(i));
                inputPaths.add("inputs/" + filename);
            }
        }

        // Write weight tensor .npy files if present
        List<String> weightPaths = new ArrayList<>();
        List<String> weightNames = new ArrayList<>();
        if (result.hasWeights() && !result.weightTensorsNpy.isEmpty()) {
            Path weightsDir = outputDir.resolve("weights");
            Files.createDirectories(weightsDir);
            for (int i = 0; i < result.weightTensorsNpy.size(); i++) {
                String filename = "weight_" + i + ".npy";
                Files.write(weightsDir.resolve(filename), result.weightTensorsNpy.get(i));
                weightPaths.add("weights/" + filename);
                if (i < result.weightNames.size()) {
                    weightNames.add(result.weightNames.get(i));
                }
            }
        }

        // Write output tensor .npy files if present
        List<String> outputPaths = new ArrayList<>();
        if (result.hasValues() && !result.outputTensorsNpy.isEmpty()) {
            Path outputsDir = outputDir.resolve("outputs");
            Files.createDirectories(outputsDir);
            for (int i = 0; i < result.outputTensorsNpy.size(); i++) {
                String filename = "output_" + i + ".npy";
                Files.write(outputsDir.resolve(filename), result.outputTensorsNpy.get(i));
                outputPaths.add("outputs/" + filename);
            }
        }

        // Write manifest
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("kind", "snakegrinder-fx-trace-bundle");
        manifest.put("version", result.hasValues() ? 2 : 1);
        manifest.put("status", result.success ? "ok" : "failed");
        manifest.put("tracer", "torch.fx.symbolic_trace");

        if (result.error != null) {
            manifest.put("error", result.error);
        }

        if (!result.warnings.isEmpty()) {
            manifest.put("warnings", result.warnings);
        }

        manifest.put("metadata", result.metadata);

        Map<String, Object> artifacts = new LinkedHashMap<>();
        if (result.success) {
            artifacts.put("mlir", "model.mlir");
            if (result.fxGraph != null) {
                artifacts.put("fx_graph", "fx_graph.txt");
            }
            if (!inputPaths.isEmpty()) {
                artifacts.put("inputs", inputPaths);
            }
            if (!weightPaths.isEmpty()) {
                artifacts.put("weights", weightPaths);
                artifacts.put("weight_names", weightNames);
            }
            if (!outputPaths.isEmpty()) {
                artifacts.put("outputs", outputPaths);
            }
        }
        manifest.put("artifacts", artifacts);

        if (result.hasValues()) {
            manifest.put("seed", result.seed);
        }
        if (result.hasWeights()) {
            manifest.put("weight_count", result.weightTensorsNpy.size());
        }

        manifest.put("timestamp", Instant.now().toString());

        Files.writeString(outputDir.resolve("manifest.json"), toJson(manifest));
    }

    // --- Private helpers ---

    private static String escapePythonMultilineString(String s) {
        return s.replace("\\", "\\\\")
                .replace("'''", "\\'\\'\\'");
    }

    private static String readResource(String resourcePath) {
        InputStream in = FxStableHloExport.class.getResourceAsStream(resourcePath);
        if (in == null) {
            throw new IllegalArgumentException("Missing classpath resource: " + resourcePath);
        }
        try (in) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed reading resource: " + resourcePath, e);
        }
    }

    private static Map<String, Object> valueToMap(Value value) {
        Map<String, Object> result = new LinkedHashMap<>();
        if (value.hasHashEntries()) {
            Value keysIterator = value.getHashKeysIterator();
            while (keysIterator.hasIteratorNextElement()) {
                Value keyValue = keysIterator.getIteratorNextElement();
                String key = keyValue.asString();
                Value member = value.getHashValue(keyValue);
                result.put(key, valueToJava(member));
            }
        } else if (value.hasMembers()) {
            for (String key : value.getMemberKeys()) {
                Value member = value.getMember(key);
                result.put(key, valueToJava(member));
            }
        }
        return result;
    }

    private static Object valueToJava(Value value) {
        if (value == null || value.isNull()) {
            return null;
        } else if (value.isBoolean()) {
            return value.asBoolean();
        } else if (value.isNumber()) {
            if (value.fitsInInt()) {
                return value.asInt();
            } else if (value.fitsInLong()) {
                return value.asLong();
            } else {
                return value.asDouble();
            }
        } else if (value.isString()) {
            return value.asString();
        } else if (value.hasArrayElements()) {
            List<Object> list = new ArrayList<>();
            for (int i = 0; i < value.getArraySize(); i++) {
                list.add(valueToJava(value.getArrayElement(i)));
            }
            return list;
        } else if (value.hasHashEntries()) {
            return valueToMap(value);
        } else if (value.hasMembers()) {
            return valueToMap(value);
        } else {
            return value.toString();
        }
    }

    private static String toJson(Map<String, Object> map) {
        return toJson(map, 0);
    }

    private static String toJson(Object obj, int indent) {
        String indentStr = "  ".repeat(indent);
        String innerIndent = "  ".repeat(indent + 1);

        if (obj == null) {
            return "null";
        } else if (obj instanceof Boolean) {
            return obj.toString();
        } else if (obj instanceof Number) {
            return obj.toString();
        } else if (obj instanceof String s) {
            return "\"" + escapeJson(s) + "\"";
        } else if (obj instanceof List<?> list) {
            if (list.isEmpty()) {
                return "[]";
            }
            StringBuilder sb = new StringBuilder("[\n");
            for (int i = 0; i < list.size(); i++) {
                sb.append(innerIndent).append(toJson(list.get(i), indent + 1));
                if (i < list.size() - 1) {
                    sb.append(",");
                }
                sb.append("\n");
            }
            sb.append(indentStr).append("]");
            return sb.toString();
        } else if (obj instanceof Map<?, ?> map) {
            if (map.isEmpty()) {
                return "{}";
            }
            StringBuilder sb = new StringBuilder("{\n");
            int i = 0;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                sb.append(innerIndent).append("\"").append(entry.getKey()).append("\": ");
                sb.append(toJson(entry.getValue(), indent + 1));
                if (i < map.size() - 1) {
                    sb.append(",");
                }
                sb.append("\n");
                i++;
            }
            sb.append(indentStr).append("}");
            return sb.toString();
        } else {
            return "\"" + escapeJson(obj.toString()) + "\"";
        }
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
