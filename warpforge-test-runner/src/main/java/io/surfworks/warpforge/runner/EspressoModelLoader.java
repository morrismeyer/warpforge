package io.surfworks.warpforge.runner;

import io.surfworks.warpforge.codegen.api.CompiledModel;
import io.surfworks.warpforge.codegen.api.ModelMetadata;
import io.surfworks.warpforge.core.backend.Backend;
import io.surfworks.warpforge.core.tensor.Tensor;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.HostAccess;
import org.graalvm.polyglot.Value;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads and executes compiled models using Espresso (Java-on-Truffle).
 *
 * <p>This loader enables running model JARs within a GraalVM native-image
 * by using Espresso to provide a guest JVM environment. The model JAR
 * is loaded in the guest JVM while the Backend executes on the host.
 *
 * <h2>Architecture</h2>
 * <pre>
 * Native Image (Host)              Espresso (Guest)
 * ┌─────────────────────┐         ┌─────────────────────┐
 * │  EspressoModelLoader│◄───────►│  model.jar          │
 * │  Backend (CPU/GPU)  │         │  CompiledModel impl │
 * │  Tensor (host mem)  │         │  Operations         │
 * └─────────────────────┘         └─────────────────────┘
 * </pre>
 *
 * <p>The key insight is that operations are dispatched back to the host
 * Backend for execution, while the model's control flow runs in Espresso.
 */
public final class EspressoModelLoader implements AutoCloseable {

    private static final String JAVA_LANGUAGE = "java";
    private static final String DEFAULT_MODEL_CLASS = "io.surfworks.warpforge.generated.Model";

    private final Context context;
    private final Value modelInstance;
    private final int inputCount;
    private final int outputCount;
    private final ModelMetadata metadata;

    private EspressoModelLoader(Context context, Value modelInstance,
                                 int inputCount, int outputCount, ModelMetadata metadata) {
        this.context = context;
        this.modelInstance = modelInstance;
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.metadata = metadata;
    }

    /**
     * Load a model JAR using Espresso.
     *
     * @param jarPath   Path to the model JAR
     * @param className Fully qualified class name of the model
     * @return A loaded model ready for execution
     * @throws EspressoLoadException if loading fails
     */
    public static EspressoModelLoader load(Path jarPath, String className) throws EspressoLoadException {
        try {
            // Create Espresso context with the model JAR on classpath
            Context context = Context.newBuilder(JAVA_LANGUAGE)
                .allowAllAccess(true)
                .allowHostAccess(HostAccess.ALL)
                .option("java.Classpath", jarPath.toAbsolutePath().toString())
                .build();

            // Load the model class
            Value modelClass = context.getBindings(JAVA_LANGUAGE).getMember(className);
            if (modelClass == null || !modelClass.canInstantiate()) {
                throw new EspressoLoadException("Cannot load model class: " + className);
            }

            // Instantiate the model
            Value modelInstance = modelClass.newInstance();

            // Extract metadata
            int inputCount = modelInstance.invokeMember("inputCount").asInt();
            int outputCount = modelInstance.invokeMember("outputCount").asInt();

            Value metadataValue = modelInstance.invokeMember("metadata");
            ModelMetadata metadata = new ModelMetadata(
                metadataValue.invokeMember("name").asString(),
                metadataValue.invokeMember("sourceHash").asString(),
                metadataValue.invokeMember("generatedAt").asLong(),
                metadataValue.invokeMember("generatorVersion").asString()
            );

            return new EspressoModelLoader(context, modelInstance, inputCount, outputCount, metadata);

        } catch (Exception e) {
            throw new EspressoLoadException("Failed to load model via Espresso: " + jarPath, e);
        }
    }

    /**
     * Load a model JAR using the default class name.
     */
    public static EspressoModelLoader load(Path jarPath) throws EspressoLoadException {
        return load(jarPath, DEFAULT_MODEL_CLASS);
    }

    /**
     * Execute the model's forward pass.
     *
     * <p>This method bridges between the host Backend and the guest model.
     * Tensors are passed by reference (sharing the underlying memory).
     *
     * @param inputs  Input tensors (host-side)
     * @param backend Backend for operation execution (host-side)
     * @return Output tensors
     */
    public List<Tensor> forward(List<Tensor> inputs, Backend backend) {
        if (inputs.size() != inputCount) {
            throw new IllegalArgumentException(
                "Expected " + inputCount + " inputs, got " + inputs.size());
        }

        // Create a host-accessible wrapper for the backend
        BackendProxy proxy = new BackendProxy(backend);

        // Convert inputs to a guest-compatible list
        Value javaListClass = context.getBindings(JAVA_LANGUAGE).getMember("java.util.ArrayList");
        Value guestInputs = javaListClass.newInstance();
        for (Tensor input : inputs) {
            guestInputs.invokeMember("add", input);
        }

        // Invoke forward with the proxy backend
        Value result = modelInstance.invokeMember("forward", guestInputs, proxy);

        // Convert results back to host tensors
        List<Tensor> outputs = new ArrayList<>();
        int size = result.invokeMember("size").asInt();
        for (int i = 0; i < size; i++) {
            Value tensorValue = result.invokeMember("get", i);
            // The tensor should be a host object passed through
            if (tensorValue.isHostObject()) {
                outputs.add(tensorValue.asHostObject());
            } else {
                throw new IllegalStateException("Expected host Tensor object at index " + i);
            }
        }

        return outputs;
    }

    /**
     * Returns the number of input tensors this model expects.
     */
    public int inputCount() {
        return inputCount;
    }

    /**
     * Returns the number of output tensors this model produces.
     */
    public int outputCount() {
        return outputCount;
    }

    /**
     * Returns metadata about this model.
     */
    public ModelMetadata metadata() {
        return metadata;
    }

    /**
     * Creates a CompiledModel wrapper for API compatibility.
     */
    public CompiledModel asCompiledModel() {
        return new EspressoCompiledModel(this);
    }

    @Override
    public void close() {
        if (context != null) {
            context.close();
        }
    }

    /**
     * Proxy that exposes the host Backend to the guest model.
     * This allows the guest model to call backend.execute() which
     * runs on the host with full native performance.
     */
    public static class BackendProxy implements Backend {
        private final Backend delegate;

        public BackendProxy(Backend delegate) {
            this.delegate = delegate;
        }

        @Override
        public String name() {
            return delegate.name();
        }

        @Override
        public io.surfworks.warpforge.core.backend.BackendCapabilities capabilities() {
            return delegate.capabilities();
        }

        @Override
        public List<Tensor> execute(io.surfworks.snakeburger.stablehlo.StableHloAst.Operation op,
                                    List<Tensor> inputs) {
            return delegate.execute(op, inputs);
        }

        @Override
        public Tensor allocate(io.surfworks.warpforge.core.tensor.TensorSpec spec) {
            return delegate.allocate(spec);
        }

        @Override
        public boolean supports(io.surfworks.snakeburger.stablehlo.StableHloAst.Operation op) {
            return delegate.supports(op);
        }

        @Override
        public void close() {
            // Don't close the delegate - it's managed by the caller
        }
    }

    /**
     * Wrapper that implements CompiledModel using Espresso execution.
     */
    private static class EspressoCompiledModel implements CompiledModel {
        private final EspressoModelLoader loader;

        EspressoCompiledModel(EspressoModelLoader loader) {
            this.loader = loader;
        }

        @Override
        public List<Tensor> forward(List<Tensor> inputs, Backend backend) {
            return loader.forward(inputs, backend);
        }

        @Override
        public int inputCount() {
            return loader.inputCount();
        }

        @Override
        public int outputCount() {
            return loader.outputCount();
        }

        @Override
        public ModelMetadata metadata() {
            return loader.metadata();
        }
    }

    /**
     * Exception thrown when Espresso model loading fails.
     */
    public static class EspressoLoadException extends Exception {
        public EspressoLoadException(String message) {
            super(message);
        }

        public EspressoLoadException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
