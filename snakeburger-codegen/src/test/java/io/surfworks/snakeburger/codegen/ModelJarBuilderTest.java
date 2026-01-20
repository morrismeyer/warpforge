package io.surfworks.snakeburger.codegen;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloParser;
import io.surfworks.warpforge.codegen.api.ModelMetadata;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.jar.Attributes;
import java.util.jar.JarEntry;
import java.util.jar.JarInputStream;
import java.util.jar.Manifest;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for ModelJarBuilder.
 *
 * <p>Verifies JAR structure, manifest attributes, metadata JSON,
 * and source MLIR embedding.
 */
class ModelJarBuilderTest {

    private static final String SIMPLE_ADD_MLIR = """
        module @main {
          func.func public @forward(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
            stablehlo.return %0 : tensor<2x2xf32>
          }
        }
        """;

    @TempDir
    Path tempDir;

    @Test
    void testBuildJarToFile() throws Exception {
        // Generate bytecode
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();
        String className = "io.surfworks.warpforge.generated.TestModel";

        var generated = ModelClassGenerator.generate(className, function, SIMPLE_ADD_MLIR);

        // Build JAR
        Path jarPath = tempDir.resolve("model.jar");
        ModelJarBuilder.build(jarPath, className, generated.bytecode(),
            generated.metadata(), SIMPLE_ADD_MLIR);

        // Verify JAR exists
        assertTrue(Files.exists(jarPath));
        assertTrue(Files.size(jarPath) > 0);

        // Verify JAR contents
        try (ZipFile zipFile = new ZipFile(jarPath.toFile())) {
            // Check class file exists
            String classPath = "io/surfworks/warpforge/generated/TestModel.class";
            ZipEntry classEntry = zipFile.getEntry(classPath);
            assertNotNull(classEntry, "Class file should exist in JAR");

            // Check metadata exists
            ZipEntry metadataEntry = zipFile.getEntry("META-INF/warpforge/model.json");
            assertNotNull(metadataEntry, "Metadata JSON should exist in JAR");

            // Check source.mlir exists
            ZipEntry sourceEntry = zipFile.getEntry("META-INF/warpforge/source.mlir");
            assertNotNull(sourceEntry, "Source MLIR should exist in JAR");

            // Check manifest
            ZipEntry manifestEntry = zipFile.getEntry("META-INF/MANIFEST.MF");
            assertNotNull(manifestEntry, "Manifest should exist in JAR");
        }
    }

    @Test
    void testBuildJarToBytes() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();
        String className = "io.surfworks.warpforge.generated.ByteModel";

        var generated = ModelClassGenerator.generate(className, function, SIMPLE_ADD_MLIR);

        // Build JAR to bytes
        byte[] jarBytes = ModelJarBuilder.buildToBytes(
            className, generated.bytecode(), generated.metadata(), SIMPLE_ADD_MLIR);

        assertNotNull(jarBytes);
        assertTrue(jarBytes.length > 0);

        // Parse JAR from bytes
        try (JarInputStream jis = new JarInputStream(new ByteArrayInputStream(jarBytes))) {
            Manifest manifest = jis.getManifest();
            assertNotNull(manifest);

            // Verify manifest attributes
            Attributes attrs = manifest.getMainAttributes();
            assertEquals("1.0", attrs.getValue(Attributes.Name.MANIFEST_VERSION));
            assertEquals(className, attrs.getValue("Model-Class"));
            assertTrue(attrs.getValue("Created-By").startsWith("WarpForge Model Compiler"));

            // Count entries
            int entryCount = 0;
            JarEntry entry;
            while ((entry = jis.getNextJarEntry()) != null) {
                entryCount++;
            }
            assertEquals(3, entryCount); // class + metadata.json + source.mlir
        }
    }

    @Test
    void testJarManifestAttributes() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();
        String className = "io.surfworks.warpforge.generated.ManifestTest";

        var generated = ModelClassGenerator.generate(className, function, SIMPLE_ADD_MLIR);

        byte[] jarBytes = ModelJarBuilder.buildToBytes(
            className, generated.bytecode(), generated.metadata(), SIMPLE_ADD_MLIR);

        try (JarInputStream jis = new JarInputStream(new ByteArrayInputStream(jarBytes))) {
            Manifest manifest = jis.getManifest();
            Attributes attrs = manifest.getMainAttributes();

            assertEquals("1.0", attrs.getValue(Attributes.Name.MANIFEST_VERSION));
            assertEquals(className, attrs.getValue("Model-Class"));
            assertNotNull(attrs.getValue("Created-By"));
            assertTrue(attrs.getValue("Created-By").contains("WarpForge"));
        }
    }

    @Test
    void testMetadataJsonContent() throws Exception {
        ModelMetadata metadata = new ModelMetadata(
            "test_model",
            "abc123def456abc123def456abc123def456abc123def456abc123def456abcd",
            1700000000000L,
            "1.0.0"
        );

        byte[] dummyBytecode = new byte[]{0x01, 0x02, 0x03};
        String className = "io.test.Model";

        byte[] jarBytes = ModelJarBuilder.buildToBytes(
            className, dummyBytecode, metadata, null);

        try (JarInputStream jis = new JarInputStream(new ByteArrayInputStream(jarBytes))) {
            JarEntry entry;
            while ((entry = jis.getNextJarEntry()) != null) {
                if (entry.getName().equals("META-INF/warpforge/model.json")) {
                    byte[] content = jis.readAllBytes();
                    String json = new String(content, StandardCharsets.UTF_8);

                    assertTrue(json.contains("\"name\": \"test_model\""));
                    assertTrue(json.contains("\"sourceHash\": \"abc123"));
                    assertTrue(json.contains("\"generatedAt\": 1700000000000"));
                    assertTrue(json.contains("\"generatorVersion\": \"1.0.0\""));
                    return;
                }
            }
        }
        throw new AssertionError("model.json not found in JAR");
    }

    @Test
    void testSourceMlirEmbedding() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();
        String className = "io.surfworks.warpforge.generated.SourceTest";

        var generated = ModelClassGenerator.generate(className, function, SIMPLE_ADD_MLIR);

        byte[] jarBytes = ModelJarBuilder.buildToBytes(
            className, generated.bytecode(), generated.metadata(), SIMPLE_ADD_MLIR);

        try (JarInputStream jis = new JarInputStream(new ByteArrayInputStream(jarBytes))) {
            JarEntry entry;
            while ((entry = jis.getNextJarEntry()) != null) {
                if (entry.getName().equals("META-INF/warpforge/source.mlir")) {
                    byte[] content = jis.readAllBytes();
                    String source = new String(content, StandardCharsets.UTF_8);

                    assertEquals(SIMPLE_ADD_MLIR, source);
                    return;
                }
            }
        }
        throw new AssertionError("source.mlir not found in JAR");
    }

    @Test
    void testSourceMlirOmittedWhenNull() throws Exception {
        ModelMetadata metadata = new ModelMetadata(
            "test", "hash123", System.currentTimeMillis(), "1.0.0");

        byte[] dummyBytecode = new byte[]{0x01, 0x02, 0x03};
        String className = "io.test.NoSourceModel";

        byte[] jarBytes = ModelJarBuilder.buildToBytes(
            className, dummyBytecode, metadata, null);

        try (JarInputStream jis = new JarInputStream(new ByteArrayInputStream(jarBytes))) {
            JarEntry entry;
            while ((entry = jis.getNextJarEntry()) != null) {
                if (entry.getName().equals("META-INF/warpforge/source.mlir")) {
                    throw new AssertionError("source.mlir should not exist when source is null");
                }
            }
        }
        // Test passes if no source.mlir entry found
    }

    @Test
    void testSourceMlirOmittedWhenBlank() throws Exception {
        ModelMetadata metadata = new ModelMetadata(
            "test", "hash123", System.currentTimeMillis(), "1.0.0");

        byte[] dummyBytecode = new byte[]{0x01, 0x02, 0x03};
        String className = "io.test.BlankSourceModel";

        byte[] jarBytes = ModelJarBuilder.buildToBytes(
            className, dummyBytecode, metadata, "   ");

        try (JarInputStream jis = new JarInputStream(new ByteArrayInputStream(jarBytes))) {
            JarEntry entry;
            while ((entry = jis.getNextJarEntry()) != null) {
                if (entry.getName().equals("META-INF/warpforge/source.mlir")) {
                    throw new AssertionError("source.mlir should not exist when source is blank");
                }
            }
        }
    }

    @Test
    void testClassFilePathMapping() throws Exception {
        // Test various class name formats
        String[] classNames = {
            "io.surfworks.warpforge.generated.Model",
            "com.example.TestModel",
            "Model"
        };

        String[] expectedPaths = {
            "io/surfworks/warpforge/generated/Model.class",
            "com/example/TestModel.class",
            "Model.class"
        };

        for (int i = 0; i < classNames.length; i++) {
            String className = classNames[i];
            String expectedPath = expectedPaths[i];

            ModelMetadata metadata = new ModelMetadata(
                "test", "hash", System.currentTimeMillis(), "1.0.0");

            byte[] dummyBytecode = new byte[]{0x01};

            byte[] jarBytes = ModelJarBuilder.buildToBytes(
                className, dummyBytecode, metadata, null);

            boolean foundClass = false;
            try (JarInputStream jis = new JarInputStream(new ByteArrayInputStream(jarBytes))) {
                JarEntry entry;
                while ((entry = jis.getNextJarEntry()) != null) {
                    if (entry.getName().equals(expectedPath)) {
                        foundClass = true;
                        break;
                    }
                }
            }
            assertTrue(foundClass, "Class should be at path: " + expectedPath);
        }
    }

    @Test
    void testBuildCreatesParentDirectories() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();
        String className = "io.surfworks.warpforge.generated.DirTest";

        var generated = ModelClassGenerator.generate(className, function, SIMPLE_ADD_MLIR);

        // Build JAR to nested path that doesn't exist
        Path jarPath = tempDir.resolve("nested/deep/path/model.jar");

        ModelJarBuilder.build(jarPath, className, generated.bytecode(),
            generated.metadata(), SIMPLE_ADD_MLIR);

        assertTrue(Files.exists(jarPath));
        assertTrue(Files.exists(jarPath.getParent()));
    }

    @Test
    void testMetadataJsonEscapesSpecialCharacters() throws Exception {
        // Metadata with characters that need JSON escaping
        ModelMetadata metadata = new ModelMetadata(
            "model\"with\"quotes",
            "hash\\with\\backslash",
            System.currentTimeMillis(),
            "1.0\n0"  // newline in version
        );

        byte[] dummyBytecode = new byte[]{0x01};
        String className = "io.test.EscapeTest";

        byte[] jarBytes = ModelJarBuilder.buildToBytes(
            className, dummyBytecode, metadata, null);

        try (JarInputStream jis = new JarInputStream(new ByteArrayInputStream(jarBytes))) {
            JarEntry entry;
            while ((entry = jis.getNextJarEntry()) != null) {
                if (entry.getName().equals("META-INF/warpforge/model.json")) {
                    byte[] content = jis.readAllBytes();
                    String json = new String(content, StandardCharsets.UTF_8);

                    // Verify JSON is properly escaped
                    assertTrue(json.contains("\\\""), "Quotes should be escaped");
                    assertTrue(json.contains("\\\\"), "Backslashes should be escaped");
                    assertTrue(json.contains("\\n"), "Newlines should be escaped");
                    return;
                }
            }
        }
        throw new AssertionError("model.json not found in JAR");
    }

    @Test
    void testBytecodePreserved() throws Exception {
        StableHloAst.Module module = StableHloParser.parse(SIMPLE_ADD_MLIR);
        StableHloAst.Function function = module.functions().getFirst();
        String className = "io.surfworks.warpforge.generated.BytecodeTest";

        var generated = ModelClassGenerator.generate(className, function, SIMPLE_ADD_MLIR);
        byte[] originalBytecode = generated.bytecode();

        byte[] jarBytes = ModelJarBuilder.buildToBytes(
            className, originalBytecode, generated.metadata(), SIMPLE_ADD_MLIR);

        // Extract bytecode from JAR and compare
        String classPath = "io/surfworks/warpforge/generated/BytecodeTest.class";

        try (JarInputStream jis = new JarInputStream(new ByteArrayInputStream(jarBytes))) {
            JarEntry entry;
            while ((entry = jis.getNextJarEntry()) != null) {
                if (entry.getName().equals(classPath)) {
                    byte[] extractedBytecode = jis.readAllBytes();
                    assertArrayEquals(originalBytecode, extractedBytecode);
                    return;
                }
            }
        }
        throw new AssertionError("Class file not found in JAR");
    }
}
