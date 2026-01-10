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

        private TraceResult(boolean success, String mlir, String fxGraph, String error,
                            String traceback, List<String> warnings, Map<String, Object> metadata) {
            this.success = success;
            this.mlir = mlir;
            this.fxGraph = fxGraph;
            this.error = error;
            this.traceback = traceback;
            this.warnings = warnings != null ? warnings : List.of();
            this.metadata = metadata != null ? metadata : Map.of();
        }

        public static TraceResult ok(String mlir, String fxGraph, Map<String, Object> metadata) {
            return new TraceResult(true, mlir, fxGraph, null, null, List.of(), metadata);
        }

        public static TraceResult fail(String error, String traceback) {
            return new TraceResult(false, null, null, error, traceback, List.of(), Map.of());
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

            // Build input_shapes as Python list of tuples
            StringBuilder inputShapesPy = new StringBuilder("[");
            for (int i = 0; i < inputSpecs.size(); i++) {
                if (i > 0) inputShapesPy.append(", ");
                inputShapesPy.append("(");
                for (int j = 0; j < inputSpecs.get(i).shape.length; j++) {
                    if (j > 0) inputShapesPy.append(", ");
                    inputShapesPy.append(inputSpecs.get(i).shape[j]);
                }
                inputShapesPy.append(")");
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

        // Write manifest
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("kind", "snakegrinder-fx-trace-bundle");
        manifest.put("version", 1);
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
        }
        manifest.put("artifacts", artifacts);
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
