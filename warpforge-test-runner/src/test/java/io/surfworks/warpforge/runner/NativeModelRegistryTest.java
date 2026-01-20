package io.surfworks.warpforge.runner;

import io.surfworks.warpforge.codegen.api.CompiledModel;
import io.surfworks.warpforge.codegen.api.ModelMetadata;
import io.surfworks.warpforge.core.backend.Backend;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for NativeModelRegistry.
 *
 * <p>Note: Since NativeModelRegistry uses static state, tests use unique
 * model names (via UUID) to avoid cross-test interference.
 */
class NativeModelRegistryTest {

    /**
     * Creates a unique model name to avoid cross-test interference.
     */
    private static String uniqueName(String prefix) {
        return prefix + "_" + UUID.randomUUID().toString().substring(0, 8);
    }

    /**
     * Creates a mock CompiledModel for testing.
     */
    private static CompiledModel mockModel(String name) {
        return new CompiledModel() {
            @Override
            public List<Tensor> forward(List<Tensor> inputs, Backend backend) {
                return List.of();
            }

            @Override
            public int inputCount() {
                return 2;
            }

            @Override
            public int outputCount() {
                return 1;
            }

            @Override
            public ModelMetadata metadata() {
                return new ModelMetadata(name, "hash", System.currentTimeMillis(), "1.0.0");
            }
        };
    }

    // ========================
    // register() tests
    // ========================

    @Test
    void testRegisterAndGet() {
        String modelName = uniqueName("register_get");
        CompiledModel model = mockModel(modelName);

        NativeModelRegistry.register(modelName, model);

        Optional<CompiledModel> retrieved = NativeModelRegistry.get(modelName);
        assertTrue(retrieved.isPresent());
        assertEquals(model, retrieved.get());
    }

    @Test
    void testRegisterOverwrites() {
        String modelName = uniqueName("overwrite");
        CompiledModel model1 = mockModel(modelName + "_v1");
        CompiledModel model2 = mockModel(modelName + "_v2");

        NativeModelRegistry.register(modelName, model1);
        NativeModelRegistry.register(modelName, model2);

        Optional<CompiledModel> retrieved = NativeModelRegistry.get(modelName);
        assertTrue(retrieved.isPresent());
        assertEquals(model2, retrieved.get());  // Should be the second model
    }

    @Test
    void testRegisterMultipleModels() {
        String name1 = uniqueName("multi1");
        String name2 = uniqueName("multi2");
        String name3 = uniqueName("multi3");

        CompiledModel model1 = mockModel(name1);
        CompiledModel model2 = mockModel(name2);
        CompiledModel model3 = mockModel(name3);

        NativeModelRegistry.register(name1, model1);
        NativeModelRegistry.register(name2, model2);
        NativeModelRegistry.register(name3, model3);

        assertTrue(NativeModelRegistry.get(name1).isPresent());
        assertTrue(NativeModelRegistry.get(name2).isPresent());
        assertTrue(NativeModelRegistry.get(name3).isPresent());

        assertEquals(model1, NativeModelRegistry.get(name1).get());
        assertEquals(model2, NativeModelRegistry.get(name2).get());
        assertEquals(model3, NativeModelRegistry.get(name3).get());
    }

    // ========================
    // get() tests
    // ========================

    @Test
    void testGetNonexistent() {
        String modelName = uniqueName("nonexistent");
        Optional<CompiledModel> result = NativeModelRegistry.get(modelName);
        assertFalse(result.isPresent());
    }

    @Test
    void testGetIsCaseInsensitive() {
        String modelName = uniqueName("CaseTest");
        CompiledModel model = mockModel(modelName);

        NativeModelRegistry.register(modelName, model);

        // All case variations should work
        assertTrue(NativeModelRegistry.get(modelName.toLowerCase()).isPresent());
        assertTrue(NativeModelRegistry.get(modelName.toUpperCase()).isPresent());
    }

    @Test
    void testGetPreservesModelProperties() {
        String modelName = uniqueName("props");
        CompiledModel model = new CompiledModel() {
            @Override
            public List<Tensor> forward(List<Tensor> inputs, Backend backend) {
                return List.of();
            }

            @Override
            public int inputCount() {
                return 5;
            }

            @Override
            public int outputCount() {
                return 3;
            }

            @Override
            public ModelMetadata metadata() {
                return new ModelMetadata("custom", "abc123", 12345L, "2.0.0");
            }
        };

        NativeModelRegistry.register(modelName, model);

        CompiledModel retrieved = NativeModelRegistry.get(modelName).orElseThrow();
        assertEquals(5, retrieved.inputCount());
        assertEquals(3, retrieved.outputCount());
        assertEquals("custom", retrieved.metadata().name());
        assertEquals("abc123", retrieved.metadata().sourceHash());
        assertEquals(12345L, retrieved.metadata().generatedAt());
        assertEquals("2.0.0", retrieved.metadata().generatorVersion());
    }

    // ========================
    // contains() tests
    // ========================

    @Test
    void testContainsRegistered() {
        String modelName = uniqueName("contains_yes");
        CompiledModel model = mockModel(modelName);

        NativeModelRegistry.register(modelName, model);

        assertTrue(NativeModelRegistry.contains(modelName));
    }

    @Test
    void testContainsNotRegistered() {
        String modelName = uniqueName("contains_no");
        assertFalse(NativeModelRegistry.contains(modelName));
    }

    @Test
    void testContainsIsCaseInsensitive() {
        String modelName = uniqueName("ContainsCase");
        CompiledModel model = mockModel(modelName);

        NativeModelRegistry.register(modelName, model);

        assertTrue(NativeModelRegistry.contains(modelName.toLowerCase()));
        assertTrue(NativeModelRegistry.contains(modelName.toUpperCase()));
    }

    // ========================
    // listModels() tests
    // ========================

    @Test
    void testListModelsIncludesRegistered() {
        String modelName = uniqueName("listed");
        CompiledModel model = mockModel(modelName);

        NativeModelRegistry.register(modelName, model);

        Set<String> models = NativeModelRegistry.listModels();
        assertTrue(models.contains(modelName.toLowerCase())); // stored lowercase
    }

    @Test
    void testListModelsReturnsImmutableSet() {
        Set<String> models = NativeModelRegistry.listModels();

        // Attempting to modify should throw
        String newName = uniqueName("immutable");
        try {
            models.add(newName);
            throw new AssertionError("listModels() should return immutable set");
        } catch (UnsupportedOperationException e) {
            // Expected
        }
    }

    // ========================
    // size() tests
    // ========================

    @Test
    void testSizeIncreasesAfterRegister() {
        int initialSize = NativeModelRegistry.size();

        String modelName = uniqueName("size_test");
        NativeModelRegistry.register(modelName, mockModel(modelName));

        assertEquals(initialSize + 1, NativeModelRegistry.size());
    }

    @Test
    void testSizeDoesNotIncreaseOnOverwrite() {
        String modelName = uniqueName("size_overwrite");
        NativeModelRegistry.register(modelName, mockModel(modelName + "_v1"));
        int sizeAfterFirst = NativeModelRegistry.size();

        NativeModelRegistry.register(modelName, mockModel(modelName + "_v2"));

        assertEquals(sizeAfterFirst, NativeModelRegistry.size());
    }

    // ========================
    // isEmpty() tests
    // ========================

    @Test
    void testIsEmptyFalseAfterRegister() {
        // Register something to ensure not empty
        String modelName = uniqueName("not_empty");
        NativeModelRegistry.register(modelName, mockModel(modelName));

        assertFalse(NativeModelRegistry.isEmpty());
    }

    // ========================
    // Case sensitivity tests
    // ========================

    @Test
    void testRegisterNormalizesCaseToLower() {
        String mixedCase = "MixedCaseModel_" + UUID.randomUUID().toString().substring(0, 8);
        CompiledModel model = mockModel(mixedCase);

        NativeModelRegistry.register(mixedCase, model);

        // Should be retrievable with any case
        assertTrue(NativeModelRegistry.get(mixedCase).isPresent());
        assertTrue(NativeModelRegistry.get(mixedCase.toLowerCase()).isPresent());
        assertTrue(NativeModelRegistry.get(mixedCase.toUpperCase()).isPresent());

        // Model name in listModels should be lowercase
        Set<String> models = NativeModelRegistry.listModels();
        assertTrue(models.contains(mixedCase.toLowerCase()));
    }

    // ========================
    // Edge cases
    // ========================

    @Test
    void testRegisterWithEmptyName() {
        CompiledModel model = mockModel("empty_name");

        // Empty name should be allowed (stored as lowercase empty string)
        NativeModelRegistry.register("", model);
        assertTrue(NativeModelRegistry.contains(""));
    }

    @Test
    void testRegisterWithSpecialCharacters() {
        String specialName = "model-with_special.chars" + UUID.randomUUID().toString().substring(0, 4);
        CompiledModel model = mockModel(specialName);

        NativeModelRegistry.register(specialName, model);

        assertTrue(NativeModelRegistry.contains(specialName));
        assertTrue(NativeModelRegistry.get(specialName).isPresent());
    }
}
