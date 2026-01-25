package io.surfworks.warpforge.data.stablehlo;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;
import io.surfworks.warpforge.data.benchmark.BenchmarkConfig;
import io.surfworks.warpforge.data.benchmark.ModelBenchmark;
import io.surfworks.warpforge.data.stablehlo.StableHloModule.StableHloFunction;
import io.surfworks.warpforge.data.stablehlo.StableHloTypes.ScalarType;
import io.surfworks.warpforge.data.stablehlo.StableHloTypes.TensorType;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Benchmark implementation that loads and runs StableHLO models.
 *
 * <p>This allows benchmarking pre-exported StableHLO MLIR files and
 * validating their outputs against golden references.
 *
 * <p>Example usage:
 * <pre>{@code
 * StableHloBenchmark benchmark = StableHloBenchmark.builder("matmul-test")
 *     .mlirFile(Path.of("fixtures/stablehlo/matrix/matmul.mlir"))
 *     .build();
 *
 * BenchmarkConfig config = BenchmarkConfig.builder("matmul")
 *     .warmupIterations(5)
 *     .measurementIterations(100)
 *     .build();
 *
 * BenchmarkResult result = runner.run(benchmark, config);
 * }</pre>
 */
public final class StableHloBenchmark implements ModelBenchmark {

    private final String name;
    private final Path mlirFile;
    private final String mlirContent;
    private final boolean randomInputs;
    private final long seed;

    private StableHloModule module;
    private Arena arena;
    private Random random;

    private StableHloBenchmark(Builder builder) {
        this.name = builder.name;
        this.mlirFile = builder.mlirFile;
        this.mlirContent = builder.mlirContent;
        this.randomInputs = builder.randomInputs;
        this.seed = builder.seed;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public String modelId() {
        return mlirFile != null ? mlirFile.toString() : "inline-mlir";
    }

    @Override
    public void setup(BenchmarkConfig config) throws IOException {
        if (mlirFile != null) {
            this.module = StableHloModule.loadFrom(mlirFile);
        } else if (mlirContent != null) {
            this.module = StableHloModule.parse(mlirContent);
        } else {
            throw new IOException("No MLIR file or content provided");
        }

        this.arena = Arena.ofConfined();
        this.random = new Random(seed);
    }

    @Override
    public Map<String, TensorView> prepareInputs(BenchmarkConfig config) {
        StableHloFunction func = module.mainFunction();
        if (func == null) {
            throw new IllegalStateException("No main function found in module");
        }

        Map<String, TensorView> inputs = new HashMap<>();

        for (StableHloFunction.Argument arg : func.arguments()) {
            TensorType type = arg.type();
            TensorView view = createInputTensor(arg.name(), type, config.batchSize());
            inputs.put(arg.name(), view);
        }

        return inputs;
    }

    private TensorView createInputTensor(String name, TensorType type, int batchSize) {
        // Adjust first dimension to batch size if needed
        long[] shape = type.shape().clone();
        if (shape.length > 0 && batchSize > 0) {
            shape[0] = batchSize;
        }

        long elementCount = 1;
        for (long d : shape) {
            elementCount *= d;
        }

        DType dtype = toDType(type.elementType());
        long byteSize = elementCount * dtype.byteSize();

        MemorySegment segment = arena.allocate(byteSize);

        // Fill with data
        if (randomInputs) {
            fillRandom(segment, dtype, (int) elementCount);
        } else {
            fillZeros(segment, dtype, (int) elementCount);
        }

        TensorInfo info = new TensorInfo(name, dtype, shape, 0, byteSize);
        return new TensorView(segment, info);
    }

    private void fillRandom(MemorySegment segment, DType dtype, int count) {
        for (int i = 0; i < count; i++) {
            switch (dtype) {
                case F32 -> segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, random.nextFloat());
                case F64 -> segment.setAtIndex(ValueLayout.JAVA_DOUBLE, i, random.nextDouble());
                case F16, BF16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, i,
                        (short) Float.floatToFloat16(random.nextFloat()));
                case I32 -> segment.setAtIndex(ValueLayout.JAVA_INT, i, random.nextInt());
                case I64 -> segment.setAtIndex(ValueLayout.JAVA_LONG, i, random.nextLong());
                case I16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, i, (short) random.nextInt(Short.MAX_VALUE));
                case I8 -> segment.setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) random.nextInt(Byte.MAX_VALUE));
                default -> throw new UnsupportedOperationException("Unsupported dtype: " + dtype);
            }
        }
    }

    private void fillZeros(MemorySegment segment, DType dtype, int count) {
        segment.fill((byte) 0);
    }

    @Override
    public Map<String, TensorView> runInference(Map<String, TensorView> inputs) throws IOException {
        // This is a stub implementation - actual inference requires a backend
        // In practice, this would dispatch to warpforge-core's GraphExecutor
        // For benchmarking purposes, we simulate output tensor creation

        StableHloFunction func = module.mainFunction();
        Map<String, TensorView> outputs = new HashMap<>();

        int outputIndex = 0;
        for (TensorType returnType : func.returnTypes()) {
            String outputName = "output_" + outputIndex;
            long[] shape = returnType.shape().clone();

            // Adjust batch dimension to match input
            if (shape.length > 0 && !inputs.isEmpty()) {
                TensorView firstInput = inputs.values().iterator().next();
                if (firstInput.shape().length > 0) {
                    shape[0] = firstInput.shape()[0];
                }
            }

            long elementCount = 1;
            for (long d : shape) {
                elementCount *= d;
            }

            DType dtype = toDType(returnType.elementType());
            long byteSize = elementCount * dtype.byteSize();

            MemorySegment segment = arena.allocate(byteSize);
            TensorInfo info = new TensorInfo(outputName, dtype, shape, 0, byteSize);
            outputs.put(outputName, new TensorView(segment, info));

            outputIndex++;
        }

        return outputs;
    }

    @Override
    public List<String> outputsToValidate() {
        StableHloFunction func = module.mainFunction();
        if (func == null) return List.of();

        List<String> outputs = new ArrayList<>();
        for (int i = 0; i < func.returnTypes().size(); i++) {
            outputs.add("output_" + i);
        }
        return outputs;
    }

    @Override
    public void teardown() {
        if (arena != null) {
            arena.close();
            arena = null;
        }
    }

    /**
     * Get the loaded StableHLO module.
     */
    public StableHloModule module() {
        return module;
    }

    /**
     * Export the module to MLIR text.
     */
    public String toMlir() {
        return module != null ? module.toMlir() : "";
    }

    private static DType toDType(ScalarType scalarType) {
        return switch (scalarType) {
            case F16 -> DType.F16;
            case F32 -> DType.F32;
            case F64 -> DType.F64;
            case BF16 -> DType.BF16;
            case I8 -> DType.I8;
            case I16 -> DType.I16;
            case I32 -> DType.I32;
            case I64 -> DType.I64;
            case I1 -> DType.I8; // Store bool as i8
        };
    }

    /**
     * Create a builder for StableHloBenchmark.
     */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /**
     * Builder for StableHloBenchmark.
     */
    public static final class Builder {
        private final String name;
        private Path mlirFile;
        private String mlirContent;
        private boolean randomInputs = true;
        private long seed = 42;

        private Builder(String name) {
            this.name = name;
        }

        /**
         * Load MLIR from file.
         */
        public Builder mlirFile(Path path) {
            this.mlirFile = path;
            this.mlirContent = null;
            return this;
        }

        /**
         * Use inline MLIR content.
         */
        public Builder mlirContent(String mlir) {
            this.mlirContent = mlir;
            this.mlirFile = null;
            return this;
        }

        /**
         * Whether to fill inputs with random values (default: true).
         */
        public Builder randomInputs(boolean random) {
            this.randomInputs = random;
            return this;
        }

        /**
         * Random seed for reproducible inputs (default: 42).
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public StableHloBenchmark build() {
            if (mlirFile == null && mlirContent == null) {
                throw new IllegalStateException("Must provide either mlirFile or mlirContent");
            }
            return new StableHloBenchmark(this);
        }
    }
}
