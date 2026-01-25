package io.surfworks.warpforge.core.endtoend;

import io.surfworks.warpforge.core.io.NpyIO;
import io.surfworks.warpforge.core.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents an end-to-end test fixture loaded from disk.
 *
 * <p>Fixture directory structure:
 * <pre>
 * fixture_name/
 * ├── model.mlir
 * ├── manifest.json
 * ├── inputs/
 * │   ├── input_0.npy
 * │   └── input_1.npy
 * ├── weights/          (optional - for fixtures with model parameters)
 * │   ├── weight_0.npy
 * │   └── weight_1.npy
 * └── outputs/
 *     └── output_0.npy
 * </pre>
 *
 * <p>When weights are present, they are treated as additional function arguments
 * after the regular inputs. The MLIR function signature will be:
 * {@code @forward(%arg0_input, %arg1_input, %arg2_weight, %arg3_bias, ...) -> ...}
 */
public final class EndToEndTestFixture implements AutoCloseable {

    private final Path fixtureDir;
    private final String name;
    private final String mlir;
    private final List<Tensor> inputs;
    private final List<Tensor> weights;
    private final List<Tensor> expectedOutputs;
    private final Map<String, Object> manifest;
    private final long seed;

    private EndToEndTestFixture(Path fixtureDir, String name, String mlir,
                                List<Tensor> inputs, List<Tensor> weights,
                                List<Tensor> expectedOutputs,
                                Map<String, Object> manifest, long seed) {
        this.fixtureDir = fixtureDir;
        this.name = name;
        this.mlir = mlir;
        this.inputs = inputs;
        this.weights = weights;
        this.expectedOutputs = expectedOutputs;
        this.manifest = manifest;
        this.seed = seed;
    }

    /**
     * Load a fixture from a directory.
     *
     * @param fixtureDir Path to the fixture directory
     * @return Loaded fixture with tensors
     * @throws IOException if files cannot be read
     */
    public static EndToEndTestFixture load(Path fixtureDir) throws IOException {
        String name = fixtureDir.getFileName().toString();

        // Load MLIR
        Path mlirPath = fixtureDir.resolve("model.mlir");
        if (!Files.exists(mlirPath)) {
            throw new IOException("Missing model.mlir in fixture: " + fixtureDir);
        }
        String mlir = Files.readString(mlirPath);

        // Load manifest
        Map<String, Object> manifest = new LinkedHashMap<>();
        long seed = 42; // Default
        Path manifestPath = fixtureDir.resolve("manifest.json");
        if (Files.exists(manifestPath)) {
            String manifestJson = Files.readString(manifestPath);
            manifest = parseSimpleJson(manifestJson);
            if (manifest.containsKey("seed")) {
                seed = ((Number) manifest.get("seed")).longValue();
            }
        }

        // Load input tensors
        List<Tensor> inputs = loadTensorsFromDir(fixtureDir.resolve("inputs"));

        // Load weight tensors (optional - for fixtures with model parameters)
        List<Tensor> weights = loadTensorsFromDir(fixtureDir.resolve("weights"));

        // Load expected output tensors
        List<Tensor> outputs = loadTensorsFromDir(fixtureDir.resolve("outputs"));

        return new EndToEndTestFixture(fixtureDir, name, mlir, inputs, weights, outputs, manifest, seed);
    }

    /**
     * Load all .npy files from a directory in sorted order.
     */
    private static List<Tensor> loadTensorsFromDir(Path dir) throws IOException {
        List<Tensor> tensors = new ArrayList<>();
        if (!Files.exists(dir)) {
            return tensors;
        }

        // Find all .npy files sorted by name
        List<Path> npyFiles = Files.list(dir)
            .filter(p -> p.toString().endsWith(".npy"))
            .sorted()
            .toList();

        for (Path npyFile : npyFiles) {
            tensors.add(NpyIO.read(npyFile));
        }

        return tensors;
    }

    /**
     * Simple JSON parser for manifest files.
     * Handles basic types: strings, numbers, booleans, arrays, objects.
     */
    private static Map<String, Object> parseSimpleJson(String json) {
        // Very simple JSON parsing - handles basic manifest structure
        Map<String, Object> result = new LinkedHashMap<>();
        json = json.trim();

        if (!json.startsWith("{") || !json.endsWith("}")) {
            return result;
        }

        json = json.substring(1, json.length() - 1).trim();
        if (json.isEmpty()) {
            return result;
        }

        // Split by top-level commas (not inside nested structures)
        int depth = 0;
        int start = 0;
        boolean inString = false;

        for (int i = 0; i < json.length(); i++) {
            char c = json.charAt(i);

            if (c == '"' && (i == 0 || json.charAt(i - 1) != '\\')) {
                inString = !inString;
            } else if (!inString) {
                if (c == '{' || c == '[') depth++;
                else if (c == '}' || c == ']') depth--;
                else if (c == ',' && depth == 0) {
                    parseKeyValue(json.substring(start, i).trim(), result);
                    start = i + 1;
                }
            }
        }

        // Parse last key-value pair
        if (start < json.length()) {
            parseKeyValue(json.substring(start).trim(), result);
        }

        return result;
    }

    private static void parseKeyValue(String pair, Map<String, Object> result) {
        int colonPos = pair.indexOf(':');
        if (colonPos < 0) return;

        String key = pair.substring(0, colonPos).trim();
        String value = pair.substring(colonPos + 1).trim();

        // Remove quotes from key
        if (key.startsWith("\"") && key.endsWith("\"")) {
            key = key.substring(1, key.length() - 1);
        }

        result.put(key, parseValue(value));
    }

    private static Object parseValue(String value) {
        value = value.trim();

        if (value.startsWith("\"") && value.endsWith("\"")) {
            return value.substring(1, value.length() - 1);
        } else if ("true".equals(value)) {
            return true;
        } else if ("false".equals(value)) {
            return false;
        } else if ("null".equals(value)) {
            return null;
        } else if (value.startsWith("{")) {
            return parseSimpleJson(value);
        } else if (value.startsWith("[")) {
            // Skip array parsing for simplicity
            return value;
        } else {
            // Try to parse as number
            try {
                if (value.contains(".")) {
                    return Double.parseDouble(value);
                } else {
                    return Long.parseLong(value);
                }
            } catch (NumberFormatException e) {
                return value;
            }
        }
    }

    // ==================== Getters ====================

    public Path fixtureDir() {
        return fixtureDir;
    }

    public String name() {
        return name;
    }

    public String mlir() {
        return mlir;
    }

    public List<Tensor> inputs() {
        return inputs;
    }

    public List<Tensor> weights() {
        return weights;
    }

    /**
     * Returns all function arguments in order: inputs followed by weights.
     * This is the complete argument list for the MLIR function.
     */
    public List<Tensor> allInputs() {
        if (weights.isEmpty()) {
            return inputs;
        }
        List<Tensor> all = new ArrayList<>(inputs.size() + weights.size());
        all.addAll(inputs);
        all.addAll(weights);
        return all;
    }

    public List<Tensor> expectedOutputs() {
        return expectedOutputs;
    }

    public Map<String, Object> manifest() {
        return manifest;
    }

    public long seed() {
        return seed;
    }

    public int inputCount() {
        return inputs.size();
    }

    public int weightCount() {
        return weights.size();
    }

    /**
     * Total number of function arguments (inputs + weights).
     */
    public int totalArgCount() {
        return inputs.size() + weights.size();
    }

    public int outputCount() {
        return expectedOutputs.size();
    }

    @Override
    public void close() {
        for (Tensor t : inputs) {
            t.close();
        }
        for (Tensor t : weights) {
            t.close();
        }
        for (Tensor t : expectedOutputs) {
            t.close();
        }
    }

    @Override
    public String toString() {
        return "EndToEndTestFixture{" +
               "name='" + name + '\'' +
               ", inputs=" + inputs.size() +
               ", weights=" + weights.size() +
               ", outputs=" + expectedOutputs.size() +
               ", seed=" + seed +
               '}';
    }
}
