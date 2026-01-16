package io.surfworks.warpforge.launch.job;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Definition of what to run - the "recipe" for a job.
 *
 * @param name          Human-readable job name
 * @param type          Type of job (FULL_PIPELINE, STABLEHLO_ONLY, PYTORCH_NATIVE)
 * @param modelSource   Path to Python source file or MLIR file
 * @param modelClass    Class name of nn.Module to trace (for PyTorch)
 * @param inputSpecs    Tensor input specifications
 * @param resources     Resource requirements
 * @param environment   Additional environment variables
 * @param timeout       Maximum execution time
 * @param seed          Random seed for reproducibility
 */
public record JobDefinition(
        String name,
        JobType type,
        Path modelSource,
        String modelClass,
        List<InputSpec> inputSpecs,
        ResourceRequirements resources,
        Map<String, String> environment,
        Duration timeout,
        long seed
) {

    /** Default timeout for jobs */
    public static final Duration DEFAULT_TIMEOUT = Duration.ofMinutes(30);

    /** Default seed for reproducibility */
    public static final long DEFAULT_SEED = 42L;

    public JobDefinition {
        Objects.requireNonNull(name, "name cannot be null");
        Objects.requireNonNull(type, "type cannot be null");
        Objects.requireNonNull(modelSource, "modelSource cannot be null");
        Objects.requireNonNull(inputSpecs, "inputSpecs cannot be null");
        Objects.requireNonNull(resources, "resources cannot be null");

        if (name.isBlank()) {
            throw new IllegalArgumentException("name cannot be blank");
        }
        if (inputSpecs.isEmpty()) {
            throw new IllegalArgumentException("inputSpecs cannot be empty");
        }

        // modelClass is required for FULL_PIPELINE and PYTORCH_NATIVE
        if ((type == JobType.FULL_PIPELINE || type == JobType.PYTORCH_NATIVE) &&
                (modelClass == null || modelClass.isBlank())) {
            throw new IllegalArgumentException("modelClass is required for " + type);
        }

        inputSpecs = List.copyOf(inputSpecs);
        environment = environment == null ? Map.of() : Map.copyOf(environment);
        timeout = timeout == null ? DEFAULT_TIMEOUT : timeout;
    }

    /**
     * Creates a builder for constructing JobDefinition instances.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Formats input specs as snakegrinder-compatible string.
     * Example: "[(1,8):f32,(1,16):f32]"
     */
    public String formatInputSpecs() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < inputSpecs.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(inputSpecs.get(i).toSpecString());
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Builder for JobDefinition.
     */
    public static class Builder {
        private String name;
        private JobType type = JobType.FULL_PIPELINE;
        private Path modelSource;
        private String modelClass;
        private List<InputSpec> inputSpecs;
        private ResourceRequirements resources = ResourceRequirements.cpuOnly(4, 4096);
        private Map<String, String> environment;
        private Duration timeout;
        private long seed = DEFAULT_SEED;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder type(JobType type) {
            this.type = type;
            return this;
        }

        public Builder modelSource(Path modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelSource(String modelSource) {
            this.modelSource = Path.of(modelSource);
            return this;
        }

        public Builder modelClass(String modelClass) {
            this.modelClass = modelClass;
            return this;
        }

        public Builder inputSpecs(List<InputSpec> inputSpecs) {
            this.inputSpecs = inputSpecs;
            return this;
        }

        public Builder inputSpecs(InputSpec... inputSpecs) {
            this.inputSpecs = List.of(inputSpecs);
            return this;
        }

        public Builder resources(ResourceRequirements resources) {
            this.resources = resources;
            return this;
        }

        public Builder environment(Map<String, String> environment) {
            this.environment = environment;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public JobDefinition build() {
            return new JobDefinition(
                    name, type, modelSource, modelClass, inputSpecs,
                    resources, environment, timeout, seed
            );
        }
    }
}
