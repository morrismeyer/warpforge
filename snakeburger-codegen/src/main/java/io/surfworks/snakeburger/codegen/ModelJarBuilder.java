package io.surfworks.snakeburger.codegen;

import io.surfworks.warpforge.codegen.api.ModelMetadata;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.jar.Attributes;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.jar.Manifest;

/**
 * Builds JAR files containing compiled models.
 *
 * <p>The JAR structure is:
 * <pre>
 * model.jar
 * ├── META-INF/
 * │   ├── MANIFEST.MF
 * │   └── warpforge/
 * │       ├── model.json          # Metadata
 * │       └── source.mlir         # Original MLIR (optional)
 * └── io/surfworks/warpforge/generated/
 *     └── Model.class             # implements CompiledModel
 * </pre>
 */
public final class ModelJarBuilder {

    private ModelJarBuilder() {} // Utility class

    /**
     * Build a model JAR file.
     *
     * @param outputPath  Path to write the JAR
     * @param className   Fully qualified class name
     * @param bytecode    The generated class bytecode
     * @param metadata    Model metadata
     * @param mlirSource  Original MLIR source (optional, may be null)
     * @throws CodegenException if JAR creation fails
     */
    public static void build(
            Path outputPath,
            String className,
            byte[] bytecode,
            ModelMetadata metadata,
            String mlirSource) throws CodegenException {

        try {
            // Ensure parent directory exists
            Path parent = outputPath.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }

            try (OutputStream os = Files.newOutputStream(outputPath);
                 JarOutputStream jos = new JarOutputStream(os, createManifest(className))) {

                // Add the class file
                String classPath = className.replace('.', '/') + ".class";
                addEntry(jos, classPath, bytecode);

                // Add metadata JSON
                String metadataJson = formatMetadataJson(metadata);
                addEntry(jos, "META-INF/warpforge/model.json", metadataJson.getBytes(StandardCharsets.UTF_8));

                // Add source MLIR if provided
                if (mlirSource != null && !mlirSource.isBlank()) {
                    addEntry(jos, "META-INF/warpforge/source.mlir", mlirSource.getBytes(StandardCharsets.UTF_8));
                }
            }

        } catch (IOException e) {
            throw new CodegenException("Failed to create JAR: " + outputPath, e);
        }
    }

    /**
     * Build a model JAR to a byte array (for in-memory use).
     */
    public static byte[] buildToBytes(
            String className,
            byte[] bytecode,
            ModelMetadata metadata,
            String mlirSource) throws CodegenException {

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             JarOutputStream jos = new JarOutputStream(baos, createManifest(className))) {

            // Add the class file
            String classPath = className.replace('.', '/') + ".class";
            addEntry(jos, classPath, bytecode);

            // Add metadata JSON
            String metadataJson = formatMetadataJson(metadata);
            addEntry(jos, "META-INF/warpforge/model.json", metadataJson.getBytes(StandardCharsets.UTF_8));

            // Add source MLIR if provided
            if (mlirSource != null && !mlirSource.isBlank()) {
                addEntry(jos, "META-INF/warpforge/source.mlir", mlirSource.getBytes(StandardCharsets.UTF_8));
            }

            jos.finish();
            return baos.toByteArray();

        } catch (IOException e) {
            throw new CodegenException("Failed to create JAR bytes", e);
        }
    }

    private static Manifest createManifest(String mainClass) {
        Manifest manifest = new Manifest();
        Attributes attrs = manifest.getMainAttributes();
        attrs.put(Attributes.Name.MANIFEST_VERSION, "1.0");
        attrs.putValue("Created-By", "WarpForge Model Compiler " + ModelClassGenerator.GENERATOR_VERSION);
        attrs.putValue("Model-Class", mainClass);
        return manifest;
    }

    private static void addEntry(JarOutputStream jos, String path, byte[] content) throws IOException {
        JarEntry entry = new JarEntry(path);
        entry.setSize(content.length);
        jos.putNextEntry(entry);
        jos.write(content);
        jos.closeEntry();
    }

    private static String formatMetadataJson(ModelMetadata metadata) {
        // Simple JSON formatting without dependencies
        return "{\n" +
               "  \"name\": \"" + escapeJson(metadata.name()) + "\",\n" +
               "  \"sourceHash\": \"" + escapeJson(metadata.sourceHash()) + "\",\n" +
               "  \"generatedAt\": " + metadata.generatedAt() + ",\n" +
               "  \"generatorVersion\": \"" + escapeJson(metadata.generatorVersion()) + "\"\n" +
               "}\n";
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
