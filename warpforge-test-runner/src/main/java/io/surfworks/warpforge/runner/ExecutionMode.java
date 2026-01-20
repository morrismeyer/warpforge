package io.surfworks.warpforge.runner;

/**
 * Execution modes for running compiled WarpForge models.
 *
 * <p>Each mode represents a different deployment scenario:
 *
 * <ul>
 *   <li><b>JVM</b>: Standard JVM execution with URLClassLoader.
 *       Best for development and debugging.</li>
 *   <li><b>ESPRESSO</b>: Native-image with Espresso guest JVM.
 *       Best for hardware CI with fast startup.</li>
 *   <li><b>NATIVE</b>: Fully AOT-compiled model (future).
 *       Best for containerized deployment.</li>
 * </ul>
 */
public enum ExecutionMode {

    /**
     * JVM Mode: Load model JAR via standard class loading.
     *
     * <p>The test runner runs on a JVM (Babylon JDK 26 or GraalVM 25),
     * and the model JAR is loaded using URLClassLoader. This provides
     * the most straightforward debugging experience.
     *
     * <p>Pros: Easy debugging, no native-image build required
     * <p>Cons: JVM startup time, larger memory footprint
     */
    JVM,

    /**
     * Espresso Mode: Load model JAR via Espresso in native-image.
     *
     * <p>The test runner is compiled to native-image with Espresso
     * (Java-on-Truffle) enabled. Model JARs are loaded into a guest
     * JVM context while backends execute on the host.
     *
     * <p>Pros: Fast startup, small footprint, good for CI
     * <p>Cons: Requires native-image build with --language:java
     */
    ESPRESSO,

    /**
     * Native Mode: Fully AOT-compiled model (future work).
     *
     * <p>Both the test runner AND the model are compiled to native code.
     * This eliminates all JVM overhead but requires the model to be
     * known at native-image build time.
     *
     * <p>Pros: Fastest possible execution, smallest footprint
     * <p>Cons: Model must be compiled into the binary, not dynamic
     */
    NATIVE;

    /**
     * Parse execution mode from string (case-insensitive).
     */
    public static ExecutionMode fromString(String s) {
        return switch (s.toLowerCase()) {
            case "jvm" -> JVM;
            case "espresso" -> ESPRESSO;
            case "native" -> NATIVE;
            default -> throw new IllegalArgumentException("Unknown execution mode: " + s);
        };
    }

    /**
     * Check if this mode is currently supported.
     *
     * <p>All modes are now supported:
     * <ul>
     *   <li>JVM: Always works with JAR files</li>
     *   <li>ESPRESSO: Works on GraalVM JVM (native-image requires special setup)</li>
     *   <li>NATIVE: Works if models are pre-registered in NativeModelRegistry</li>
     * </ul>
     */
    public boolean isSupported() {
        return true; // All modes now have basic support
    }
}
