package io.surfworks.warpforge.codegen.api;

/**
 * Metadata for a compiled model.
 * Captures provenance and versioning information.
 *
 * @param name             The model name (typically the function name from MLIR)
 * @param sourceHash       SHA-256 hash of the original MLIR source
 * @param generatedAt      Timestamp when the model was compiled (epoch millis)
 * @param generatorVersion Version of the code generator that produced this model
 */
public record ModelMetadata(
    String name,
    String sourceHash,
    long generatedAt,
    String generatorVersion
) {
    /**
     * Creates a ModelMetadata with the current timestamp.
     *
     * @param name             The model name
     * @param sourceHash       SHA-256 hash of the MLIR source
     * @param generatorVersion Version of the code generator
     * @return A new ModelMetadata instance
     */
    public static ModelMetadata create(String name, String sourceHash, String generatorVersion) {
        return new ModelMetadata(name, sourceHash, System.currentTimeMillis(), generatorVersion);
    }
}
