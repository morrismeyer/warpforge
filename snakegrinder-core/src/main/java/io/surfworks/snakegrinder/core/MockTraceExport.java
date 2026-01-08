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
 * Mock tracer for StableHLO export without real PyTorch/JAX.
 *
 * Uses pure-Python mock modules bundled as resources to trace Python
 * ML code and emit StableHLO MLIR. No native dependencies required.
 * Safe for GraalVM native image bundling.
 */
public final class MockTraceExport {

    /**
     * Python modules in dependency order.
     * Each entry is [resource_path, module_name].
     */
    private static final String[][] PYTHON_MODULES = {
        {"/snakegrinder/graph_ir.py", "graph_ir"},
        {"/snakegrinder/tracer.py", "tracer"},
        {"/snakegrinder/stablehlo_emitter.py", "stablehlo_emitter"},
        {"/snakegrinder/mock_torch.py", "mock_torch"},
        {"/snakegrinder/mock_jax.py", "mock_jax"},
        {"/snakegrinder/trace_entry.py", "trace_entry"}
    };

    private MockTraceExport() {
    }

    /**
     * Result of a trace operation.
     */
    public static final class TraceResult {
        public final boolean success;
        public final String mlir;
        public final String error;
        public final List<String> warnings;
        public final Map<String, Object> metadata;

        private TraceResult(boolean success, String mlir, String error,
                            List<String> warnings, Map<String, Object> metadata) {
            this.success = success;
            this.mlir = mlir;
            this.error = error;
            this.warnings = warnings != null ? warnings : List.of();
            this.metadata = metadata != null ? metadata : Map.of();
        }

        public static TraceResult ok(String mlir, Map<String, Object> metadata) {
            return new TraceResult(true, mlir, null, List.of(), metadata);
        }

        public static TraceResult fail(String error) {
            return new TraceResult(false, null, error, List.of(), Map.of());
        }

        public static TraceResult fail(String error, List<String> warnings) {
            return new TraceResult(false, null, error, warnings, Map.of());
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
            if (shape.length == 1) sb.append(",");  // Single-element tuple needs trailing comma
            sb.append("), '").append(dtype).append("')");
            return sb.toString();
        }
    }

    /**
     * Trace Python source code and emit StableHLO MLIR.
     *
     * @param pythonSource The Python source code containing the function to trace
     * @param functionName The name of the function to trace
     * @param inputSpecs   Input tensor specifications (shapes and dtypes)
     * @param framework    "torch" or "jax" - which mock module style to use
     * @return TraceResult containing the MLIR or error information
     */
    public static TraceResult trace(String pythonSource,
                                    String functionName,
                                    List<InputSpec> inputSpecs,
                                    String framework) {

        ByteArrayOutputStream pythonOut = new ByteArrayOutputStream();
        ByteArrayOutputStream pythonErr = new ByteArrayOutputStream();

        try (Context ctx = Context.newBuilder("python")
                .allowAllAccess(true)
                .option("python.ForceImportSite", "true")
                .out(pythonOut)
                .err(pythonErr)
                .build()) {

            // Load all mock modules in dependency order and register in sys.modules
            loadModules(ctx);

            // Build input_specs as Python list
            StringBuilder inputSpecsPy = new StringBuilder("[");
            for (int i = 0; i < inputSpecs.size(); i++) {
                if (i > 0) inputSpecsPy.append(", ");
                inputSpecsPy.append(inputSpecs.get(i).toPythonTuple());
            }
            inputSpecsPy.append("]");

            // Escape the source code for Python string literal
            String escapedSource = escapePythonString(pythonSource);

            // Call trace_source_code from the trace_entry module
            String callCode = String.format(
                "trace_entry.trace_source_code('''%s''', '%s', %s, '%s')",
                escapedSource, functionName, inputSpecsPy.toString(), framework
            );

            Value resultValue = ctx.eval("python", callCode);
            Map<String, Object> result = valueToMap(resultValue);

            String status = (String) result.get("status");
            if ("ok".equals(status)) {
                String mlir = (String) result.get("mlir");
                Map<String, Object> metadata = new LinkedHashMap<>();
                metadata.put("function", functionName);
                metadata.put("framework", framework);
                metadata.put("input_count", inputSpecs.size());
                metadata.put("timestamp", Instant.now().toString());
                return TraceResult.ok(mlir, metadata);
            } else {
                String error = (String) result.get("error");
                return TraceResult.fail(error != null ? error : "Unknown error");
            }

        } catch (PolyglotException e) {
            String error = "GraalPy error: " + e.getMessage();
            String stderr = pythonErr.toString(StandardCharsets.UTF_8);
            if (!stderr.isBlank()) {
                error += "\nPython stderr:\n" + stderr;
            }
            return TraceResult.fail(error);
        } catch (Exception e) {
            return TraceResult.fail("Trace failed: " + e.getMessage());
        }
    }

    /**
     * Trace a built-in example (matmul) for testing.
     */
    public static TraceResult traceMatmulExample() {
        ByteArrayOutputStream pythonOut = new ByteArrayOutputStream();
        ByteArrayOutputStream pythonErr = new ByteArrayOutputStream();

        try (Context ctx = Context.newBuilder("python")
                .allowAllAccess(true)
                .option("python.ForceImportSite", "true")
                .out(pythonOut)
                .err(pythonErr)
                .build()) {

            // Load modules
            loadModules(ctx);

            // Call built-in example
            Value result = ctx.eval("python", "trace_entry.trace_matmul_example()");
            String mlir = result.asString();

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("function", "matmul_example");
            metadata.put("framework", "torch");
            metadata.put("timestamp", Instant.now().toString());

            return TraceResult.ok(mlir, metadata);

        } catch (Exception e) {
            return TraceResult.fail("Example trace failed: " + e.getMessage());
        }
    }

    /**
     * Trace a built-in MLP example for testing.
     */
    public static TraceResult traceMlpExample() {
        ByteArrayOutputStream pythonOut = new ByteArrayOutputStream();
        ByteArrayOutputStream pythonErr = new ByteArrayOutputStream();

        try (Context ctx = Context.newBuilder("python")
                .allowAllAccess(true)
                .option("python.ForceImportSite", "true")
                .out(pythonOut)
                .err(pythonErr)
                .build()) {

            // Load modules
            loadModules(ctx);

            // Call built-in example
            Value result = ctx.eval("python", "trace_entry.trace_mlp_example()");
            String mlir = result.asString();

            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("function", "mlp_example");
            metadata.put("framework", "torch");
            metadata.put("timestamp", Instant.now().toString());

            return TraceResult.ok(mlir, metadata);

        } catch (Exception e) {
            return TraceResult.fail("MLP example trace failed: " + e.getMessage());
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

        // Write manifest
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("kind", "snakegrinder-trace-bundle");
        manifest.put("version", 1);
        manifest.put("status", result.success ? "ok" : "failed");

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
        }
        manifest.put("artifacts", artifacts);
        manifest.put("timestamp", Instant.now().toString());

        Files.writeString(outputDir.resolve("manifest.json"), toJson(manifest));
    }

    // --- Private helpers ---

    /**
     * Load all Python modules and register them in sys.modules.
     * This enables modules to import each other.
     */
    private static void loadModules(Context ctx) {
        // First, set up the import mechanism
        ctx.eval("python", "import sys");
        ctx.eval("python", "from types import ModuleType");

        for (String[] moduleInfo : PYTHON_MODULES) {
            String resourcePath = moduleInfo[0];
            String moduleName = moduleInfo[1];

            String code = readResource(resourcePath);

            // Create a module object, execute code in its namespace, and register it
            String setupCode = String.format(
                "_mod = ModuleType('%s')\n" +
                "_mod.__file__ = '%s'\n" +
                "sys.modules['%s'] = _mod\n" +
                "exec(compile('''%s''', '%s', 'exec'), _mod.__dict__)\n" +
                "%s = _mod\n",
                moduleName,
                resourcePath,
                moduleName,
                escapePythonMultilineString(code),
                resourcePath,
                moduleName
            );

            try {
                ctx.eval("python", setupCode);
            } catch (Exception e) {
                throw new IllegalStateException("Failed to load module: " + moduleName, e);
            }
        }
    }

    private static String escapePythonMultilineString(String s) {
        // Escape for use in triple-quoted string
        return s.replace("\\", "\\\\")
                .replace("'''", "\\'\\'\\'");
    }

    private static String escapePythonString(String s) {
        return s.replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private static String readResource(String resourcePath) {
        InputStream in = MockTraceExport.class.getResourceAsStream(resourcePath);
        if (in == null) {
            throw new IllegalArgumentException("Missing classpath resource: " + resourcePath);
        }
        try (in) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed reading resource: " + resourcePath, e);
        }
    }

    private static Source buildSource(String code, String name) {
        try {
            return Source.newBuilder("python", code, name).build();
        } catch (IOException e) {
            throw new IllegalStateException("Failed building Source: " + name, e);
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
