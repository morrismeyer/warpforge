package io.surfworks.warpforge.data.benchmark;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BenchmarkConfigTest {

    @Test
    void testBuilderDefaults() {
        BenchmarkConfig config = BenchmarkConfig.builder("test-model").build();

        assertEquals("test-model", config.modelId());
        assertEquals("cpu", config.backend());
        assertNull(config.goldenDir());
        assertEquals(BenchmarkConfig.DEFAULT_WARMUP, config.warmupIterations());
        assertEquals(BenchmarkConfig.DEFAULT_MEASUREMENT, config.measurementIterations());
        assertEquals(BenchmarkConfig.DEFAULT_TOLERANCE, config.tolerance());
        assertTrue(config.validateOutputs());
        assertFalse(config.collectMemoryStats());
        assertEquals(BenchmarkConfig.DEFAULT_BATCH_SIZE, config.batchSize());
        assertEquals(BenchmarkConfig.DEFAULT_SEQUENCE_LENGTH, config.sequenceLength());
    }

    @Test
    void testBuilderCustomValues() {
        Path goldenDir = Path.of("/tmp/goldens");

        BenchmarkConfig config = BenchmarkConfig.builder("custom-model")
                .backend("nvidia")
                .goldenDir(goldenDir)
                .warmupIterations(10)
                .measurementIterations(50)
                .tolerance(1e-4)
                .validateOutputs(false)
                .collectMemoryStats(true)
                .batchSize(8)
                .sequenceLength(512)
                .build();

        assertEquals("custom-model", config.modelId());
        assertEquals("nvidia", config.backend());
        assertEquals(goldenDir, config.goldenDir());
        assertEquals(10, config.warmupIterations());
        assertEquals(50, config.measurementIterations());
        assertEquals(1e-4, config.tolerance());
        assertFalse(config.validateOutputs());
        assertTrue(config.collectMemoryStats());
        assertEquals(8, config.batchSize());
        assertEquals(512, config.sequenceLength());
    }

    @Test
    void testDefaultsFactory() {
        BenchmarkConfig config = BenchmarkConfig.defaults("default-model");

        assertEquals("default-model", config.modelId());
        assertEquals("cpu", config.backend());
    }

    @Test
    void testNullModelIdThrows() {
        assertThrows(NullPointerException.class, () ->
                BenchmarkConfig.builder(null));
    }

    @Test
    void testInvalidWarmupIterations() {
        assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder("model")
                        .warmupIterations(-1)
                        .build());
    }

    @Test
    void testInvalidMeasurementIterations() {
        assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder("model")
                        .measurementIterations(0)
                        .build());
    }

    @Test
    void testInvalidTolerance() {
        assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder("model")
                        .tolerance(0)
                        .build());

        assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder("model")
                        .tolerance(-1e-5)
                        .build());
    }

    @Test
    void testInvalidBatchSize() {
        assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder("model")
                        .batchSize(0)
                        .build());
    }

    @Test
    void testInvalidSequenceLength() {
        assertThrows(IllegalArgumentException.class, () ->
                BenchmarkConfig.builder("model")
                        .sequenceLength(0)
                        .build());
    }

    @Test
    void testZeroWarmupIsAllowed() {
        BenchmarkConfig config = BenchmarkConfig.builder("model")
                .warmupIterations(0)
                .build();

        assertEquals(0, config.warmupIterations());
    }
}
