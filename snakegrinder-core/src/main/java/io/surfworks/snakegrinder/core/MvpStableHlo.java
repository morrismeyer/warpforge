package io.surfworks.snakegrinder.core;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
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
 * MVP StableHLO export orchestration.
 *
 * This class coordinates:
 * 1. Running the capability probe to detect available ML frameworks
 * 2. Running the export script to generate StableHLO MLIR
 * 3. Writing the output bundle (manifest.json, model.mlir, run.log)
 */
public final class MvpStableHlo {

    private MvpStableHlo() {
    }

    /**
     * Result of the MVP StableHLO export process.
     */
    public static final class ExportResult {
        public final boolean success;
        public final Path outputDir;
        public final Path mlirFile;
        public final Path manifestFile;
        public final Path logFile;
        public final String error;
        public final List<String> warnings;

        private ExportResult(boolean success, Path outputDir, Path mlirFile,
                             Path manifestFile, Path logFile, String error, List<String> warnings) {
            this.success = success;
            this.outputDir = outputDir;
            this.mlirFile = mlirFile;
            this.manifestFile = manifestFile;
            this.logFile = logFile;
            this.error = error;
            this.warnings = warnings != null ? warnings : List.of();
        }
    }

    /**
     * Run the MVP StableHLO export pipeline.
     *
     * @param outputDir Directory to write output bundle
     * @param keepTmp Whether to keep temporary files for debugging
     * @return ExportResult with status and file paths
     */
    public static ExportResult run(Path outputDir, boolean keepTmp) {
        List<String> warnings = new ArrayList<>();
        StringBuilder logBuilder = new StringBuilder();

        try {
            Files.createDirectories(outputDir);
        } catch (IOException e) {
            return new ExportResult(false, outputDir, null, null, null,
                    "Failed to create output directory: " + e.getMessage(), warnings);
        }

        Path logFile = outputDir.resolve("run.log");
        Path manifestFile = outputDir.resolve("manifest.json");
        Path mlirFile = outputDir.resolve("model.mlir");

        Map<String, Object> probeResult = null;
        Map<String, Object> exportResult = null;

        log(logBuilder, "=== SnakeGrinder MVP StableHLO Export ===");
        log(logBuilder, "Started: " + Instant.now());
        log(logBuilder, "Output directory: " + outputDir.toAbsolutePath());
        log(logBuilder, "");

        // Capture Python stdout/stderr
        ByteArrayOutputStream pythonOut = new ByteArrayOutputStream();
        ByteArrayOutputStream pythonErr = new ByteArrayOutputStream();

        // Create GraalPy context with file system access for output
        try (Context ctx = Context.newBuilder("python")
                .allowAllAccess(true)  // Need file access for export
                .option("python.ForceImportSite", "true")
                .out(pythonOut)
                .err(pythonErr)
                .build()) {

            // Step 1: Run capability probe
            log(logBuilder, "=== Phase 1: Capability Probe ===");
            try {
                probeResult = runProbe(ctx, logBuilder);
                log(logBuilder, "Probe completed successfully");
                log(logBuilder, "");
            } catch (Exception e) {
                log(logBuilder, "Probe failed: " + e.getMessage());
                logPythonOutput(logBuilder, pythonOut, pythonErr);
                writeLog(logFile, logBuilder.toString());
                writeManifest(manifestFile, "failed", "Probe phase failed: " + e.getMessage(),
                        probeResult, exportResult, warnings);
                return new ExportResult(false, outputDir, null, manifestFile, logFile,
                        "Probe phase failed: " + e.getMessage(), warnings);
            }

            // Check export path availability
            String exportPath = (String) probeResult.get("export_path");
            String exportPathReason = (String) probeResult.get("export_path_reason");
            log(logBuilder, "Export path: " + exportPath);
            log(logBuilder, "Reason: " + exportPathReason);
            log(logBuilder, "");

            if ("none".equals(exportPath)) {
                String error = buildMissingDependenciesError(probeResult);
                log(logBuilder, "ERROR: " + error);
                logPythonOutput(logBuilder, pythonOut, pythonErr);
                writeLog(logFile, logBuilder.toString());
                writeManifest(manifestFile, "failed", error, probeResult, exportResult, warnings);
                return new ExportResult(false, outputDir, null, manifestFile, logFile, error, warnings);
            }

            if ("torch_only".equals(exportPath)) {
                String error = buildMissingExportPathError(probeResult);
                log(logBuilder, "ERROR: " + error);
                logPythonOutput(logBuilder, pythonOut, pythonErr);
                writeLog(logFile, logBuilder.toString());
                writeManifest(manifestFile, "failed", error, probeResult, exportResult, warnings);
                return new ExportResult(false, outputDir, null, manifestFile, logFile, error, warnings);
            }

            // Step 2: Run export
            log(logBuilder, "=== Phase 2: StableHLO Export ===");
            try {
                exportResult = runExport(ctx, outputDir, logBuilder);
                log(logBuilder, "Export phase completed");
                log(logBuilder, "");
            } catch (Exception e) {
                log(logBuilder, "Export failed: " + e.getMessage());
                logPythonOutput(logBuilder, pythonOut, pythonErr);
                writeLog(logFile, logBuilder.toString());
                writeManifest(manifestFile, "failed", "Export phase failed: " + e.getMessage(),
                        probeResult, exportResult, warnings);
                return new ExportResult(false, outputDir, null, manifestFile, logFile,
                        "Export phase failed: " + e.getMessage(), warnings);
            }

            // Check export result
            String status = (String) exportResult.get("status");
            if (!"ok".equals(status)) {
                String error = (String) exportResult.get("error");
                log(logBuilder, "Export failed: " + error);
                logPythonOutput(logBuilder, pythonOut, pythonErr);
                writeLog(logFile, logBuilder.toString());
                writeManifest(manifestFile, "failed", error, probeResult, exportResult, warnings);
                return new ExportResult(false, outputDir, null, manifestFile, logFile, error, warnings);
            }

            // Collect warnings from export
            Object exportWarnings = exportResult.get("warnings");
            if (exportWarnings instanceof List<?>) {
                for (Object w : (List<?>) exportWarnings) {
                    warnings.add(String.valueOf(w));
                }
            }

            // Verify MLIR file exists and contains stablehlo ops
            log(logBuilder, "=== Phase 3: Validation ===");
            if (!Files.exists(mlirFile)) {
                String error = "model.mlir was not created";
                log(logBuilder, "Validation failed: " + error);
                logPythonOutput(logBuilder, pythonOut, pythonErr);
                writeLog(logFile, logBuilder.toString());
                writeManifest(manifestFile, "failed", error, probeResult, exportResult, warnings);
                return new ExportResult(false, outputDir, null, manifestFile, logFile, error, warnings);
            }

            String mlirContent;
            try {
                mlirContent = Files.readString(mlirFile);
            } catch (IOException e) {
                String error = "Failed to read model.mlir: " + e.getMessage();
                log(logBuilder, "Validation failed: " + error);
                logPythonOutput(logBuilder, pythonOut, pythonErr);
                writeLog(logFile, logBuilder.toString());
                writeManifest(manifestFile, "failed", error, probeResult, exportResult, warnings);
                return new ExportResult(false, outputDir, null, manifestFile, logFile, error, warnings);
            }

            boolean hasStableHloOps = mlirContent.toLowerCase().contains("stablehlo.");
            boolean hasMhloOps = mlirContent.toLowerCase().contains("mhlo.");
            boolean hasHloOps = mlirContent.contains("HloModule") || mlirContent.contains("hlo.");

            if (hasStableHloOps) {
                log(logBuilder, "Validation passed: model.mlir contains stablehlo. ops");
            } else if (hasMhloOps) {
                warnings.add("model.mlir contains MHLO ops but not StableHLO ops");
                log(logBuilder, "Warning: model.mlir contains MHLO ops (StableHLO predecessor)");
            } else if (hasHloOps) {
                warnings.add("model.mlir contains HLO ops but not StableHLO ops");
                log(logBuilder, "Warning: model.mlir contains HLO ops (XLA intermediate representation)");
            } else {
                String error = "model.mlir does not contain recognizable HLO/MHLO/StableHLO ops";
                log(logBuilder, "Validation failed: " + error);
                logPythonOutput(logBuilder, pythonOut, pythonErr);
                writeLog(logFile, logBuilder.toString());
                writeManifest(manifestFile, "failed", error, probeResult, exportResult, warnings);
                return new ExportResult(false, outputDir, null, manifestFile, logFile, error, warnings);
            }

            log(logBuilder, "MLIR file size: " + mlirContent.length() + " bytes");
            log(logBuilder, "");

            // Success
            log(logBuilder, "=== Export Complete ===");
            log(logBuilder, "Status: SUCCESS");
            log(logBuilder, "Output files:");
            log(logBuilder, "  - " + mlirFile);
            log(logBuilder, "  - " + manifestFile);
            log(logBuilder, "  - " + logFile);

            logPythonOutput(logBuilder, pythonOut, pythonErr);
            writeLog(logFile, logBuilder.toString());
            writeManifest(manifestFile, "ok", null, probeResult, exportResult, warnings);

            return new ExportResult(true, outputDir, mlirFile, manifestFile, logFile, null, warnings);

        } catch (PolyglotException e) {
            String error = "GraalPy error: " + e.getMessage();
            log(logBuilder, "FATAL: " + error);
            logPythonOutput(logBuilder, pythonOut, pythonErr);
            writeLog(logFile, logBuilder.toString());
            writeManifest(manifestFile, "failed", error, probeResult, exportResult, warnings);
            return new ExportResult(false, outputDir, null, manifestFile, logFile, error, warnings);
        }
    }

    private static void logPythonOutput(StringBuilder logBuilder,
                                         ByteArrayOutputStream pythonOut,
                                         ByteArrayOutputStream pythonErr) {
        String stdout = pythonOut.toString(StandardCharsets.UTF_8);
        String stderr = pythonErr.toString(StandardCharsets.UTF_8);

        if (!stdout.isBlank() || !stderr.isBlank()) {
            log(logBuilder, "");
            log(logBuilder, "=== Python Output ===");
            if (!stdout.isBlank()) {
                log(logBuilder, "stdout:");
                log(logBuilder, stdout);
            }
            if (!stderr.isBlank()) {
                log(logBuilder, "stderr:");
                log(logBuilder, stderr);
            }
        }
    }

    private static Map<String, Object> runProbe(Context ctx, StringBuilder logBuilder) {
        String probeCode = readResource("/snakegrinder/probe.py");
        Source probeSource = buildSource(probeCode, "probe.py");
        ctx.eval(probeSource);

        Value probeResult = ctx.eval("python", "probe()");
        return valueToMap(probeResult);
    }

    private static Map<String, Object> runExport(Context ctx, Path outputDir, StringBuilder logBuilder) {
        String exportCode = readResource("/snakegrinder/export_stablehlo.py");
        Source exportSource = buildSource(exportCode, "export_stablehlo.py");
        ctx.eval(exportSource);

        // Set the output directory and run export
        String escapedPath = outputDir.toAbsolutePath().toString().replace("\\", "\\\\").replace("'", "\\'");
        Value exportResult = ctx.eval("python", "export_stablehlo('" + escapedPath + "')");
        return valueToMap(exportResult);
    }

    private static String buildMissingDependenciesError(Map<String, Object> probeResult) {
        StringBuilder sb = new StringBuilder();
        sb.append("No ML frameworks available for StableHLO export.\n\n");
        sb.append("Required: PyTorch with torch_xla, or JAX\n\n");
        sb.append("Installation options:\n\n");
        sb.append("Option 1 (PyTorch + torch_xla):\n");
        sb.append("  pip install torch torch_xla\n\n");
        sb.append("Option 2 (JAX):\n");
        sb.append("  pip install jax jaxlib\n\n");

        String torchError = (String) probeResult.get("torch_error");
        String jaxError = (String) probeResult.get("jax_error");

        sb.append("Current environment errors:\n");
        if (torchError != null) {
            sb.append("  PyTorch: ").append(torchError).append("\n");
        }
        if (jaxError != null) {
            sb.append("  JAX: ").append(jaxError).append("\n");
        }

        return sb.toString();
    }

    private static String buildMissingExportPathError(Map<String, Object> probeResult) {
        StringBuilder sb = new StringBuilder();
        sb.append("PyTorch is available but no StableHLO export path is installed.\n\n");
        sb.append("PyTorch version: ").append(probeResult.get("torch_version")).append("\n\n");
        sb.append("To export StableHLO, install one of:\n\n");
        sb.append("Option 1 (torch_xla - recommended):\n");
        sb.append("  pip install torch_xla\n\n");
        sb.append("Option 2 (JAX fallback):\n");
        sb.append("  pip install jax jaxlib\n\n");

        String torchXlaError = (String) probeResult.get("torch_xla_error");
        String jaxError = (String) probeResult.get("jax_error");

        sb.append("Current errors:\n");
        if (torchXlaError != null) {
            sb.append("  torch_xla: ").append(torchXlaError).append("\n");
        }
        if (jaxError != null) {
            sb.append("  JAX: ").append(jaxError).append("\n");
        }

        return sb.toString();
    }

    private static void log(StringBuilder logBuilder, String message) {
        logBuilder.append(message).append("\n");
    }

    private static void writeLog(Path logFile, String content) {
        try {
            Files.writeString(logFile, content);
        } catch (IOException e) {
            System.err.println("Warning: Failed to write log file: " + e.getMessage());
        }
    }

    private static void writeManifest(Path manifestFile, String status, String error,
                                       Map<String, Object> probeResult, Map<String, Object> exportResult,
                                       List<String> warnings) {
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("kind", "snakegrinder-mlir-bundle");
        manifest.put("version", 1);
        manifest.put("status", status);

        if (error != null) {
            manifest.put("error", error);
        }

        if (!warnings.isEmpty()) {
            manifest.put("warnings", warnings);
        }

        Map<String, Object> source = new LinkedHashMap<>();
        source.put("language", "python");
        source.put("entrypoint", "mvp_torch_matmul");
        manifest.put("source", source);

        if (probeResult != null) {
            Map<String, Object> environment = new LinkedHashMap<>();
            environment.put("python_version", probeResult.get("python_version"));
            environment.put("platform", probeResult.get("platform"));
            environment.put("torch_version", probeResult.get("torch_version"));
            environment.put("export_path", probeResult.get("export_path"));
            manifest.put("environment", environment);
        }

        Map<String, Object> artifacts = new LinkedHashMap<>();
        if ("ok".equals(status)) {
            artifacts.put("mlir", "model.mlir");
        }
        artifacts.put("log", "run.log");
        manifest.put("artifacts", artifacts);

        manifest.put("timestamp", Instant.now().toString());

        String json = toJson(manifest);
        try {
            Files.writeString(manifestFile, json);
        } catch (IOException e) {
            System.err.println("Warning: Failed to write manifest file: " + e.getMessage());
        }
    }

    private static Map<String, Object> valueToMap(Value value) {
        Map<String, Object> result = new LinkedHashMap<>();
        // Python dicts use hash entries, not members
        if (value.hasHashEntries()) {
            Value keysIterator = value.getHashKeysIterator();
            while (keysIterator.hasIteratorNextElement()) {
                Value keyValue = keysIterator.getIteratorNextElement();
                String key = keyValue.asString();
                Value member = value.getHashValue(keyValue);
                result.put(key, valueToJava(member));
            }
        } else if (value.hasMembers()) {
            // Fallback for objects with members
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
            // Python dict
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

    private static String readResource(String resourcePath) {
        InputStream in = MvpStableHlo.class.getResourceAsStream(resourcePath);
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
}
