package io.surfworks.warpforge.codegen.api;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.jar.JarFile;

/**
 * Utility for loading compiled models from JAR files.
 *
 * <p>The expected JAR structure is:
 * <pre>
 * model.jar
 * ├── META-INF/
 * │   ├── MANIFEST.MF
 * │   └── warpforge/
 * │       ├── model.json          # Metadata (optional)
 * │       └── source.mlir         # Original MLIR (optional, for debugging)
 * └── io/surfworks/warpforge/generated/
 *     └── Model.class             # implements CompiledModel
 * </pre>
 */
public final class ModelLoader {

    private static final String DEFAULT_MODEL_CLASS = "io.surfworks.warpforge.generated.Model";
    private static final String METADATA_PATH = "META-INF/warpforge/model.json";
    private static final String SOURCE_PATH = "META-INF/warpforge/source.mlir";

    private ModelLoader() {} // Utility class

    /**
     * Load a compiled model from a JAR file.
     *
     * @param jarPath Path to the model JAR
     * @return The loaded CompiledModel
     * @throws ModelLoadException if loading fails
     */
    public static CompiledModel load(Path jarPath) throws ModelLoadException {
        return load(jarPath, DEFAULT_MODEL_CLASS);
    }

    /**
     * Load a compiled model from a JAR file with a custom class name.
     *
     * @param jarPath   Path to the model JAR
     * @param className Fully qualified class name of the model
     * @return The loaded CompiledModel
     * @throws ModelLoadException if loading fails
     */
    public static CompiledModel load(Path jarPath, String className) throws ModelLoadException {
        try {
            URL jarUrl = jarPath.toUri().toURL();
            URLClassLoader classLoader = new URLClassLoader(
                new URL[]{jarUrl},
                ModelLoader.class.getClassLoader()
            );

            Class<?> modelClass = classLoader.loadClass(className);

            if (!CompiledModel.class.isAssignableFrom(modelClass)) {
                throw new ModelLoadException(
                    "Class " + className + " does not implement CompiledModel");
            }

            return (CompiledModel) modelClass.getDeclaredConstructor().newInstance();

        } catch (IOException e) {
            throw new ModelLoadException("Failed to open JAR: " + jarPath, e);
        } catch (ClassNotFoundException e) {
            throw new ModelLoadException("Model class not found: " + className, e);
        } catch (NoSuchMethodException e) {
            throw new ModelLoadException("Model class missing no-arg constructor: " + className, e);
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException e) {
            throw new ModelLoadException("Failed to instantiate model: " + className, e);
        }
    }

    /**
     * Load a compiled model from bytes (for in-memory model loading).
     *
     * @param classBytes The bytecode of the model class
     * @param className  Fully qualified class name
     * @return The loaded CompiledModel
     * @throws ModelLoadException if loading fails
     */
    public static CompiledModel loadFromBytes(byte[] classBytes, String className) throws ModelLoadException {
        try {
            ByteArrayClassLoader classLoader = new ByteArrayClassLoader(
                className, classBytes, ModelLoader.class.getClassLoader());

            Class<?> modelClass = classLoader.loadClass(className);

            if (!CompiledModel.class.isAssignableFrom(modelClass)) {
                throw new ModelLoadException(
                    "Class " + className + " does not implement CompiledModel");
            }

            return (CompiledModel) modelClass.getDeclaredConstructor().newInstance();

        } catch (ClassNotFoundException e) {
            throw new ModelLoadException("Failed to define class: " + className, e);
        } catch (NoSuchMethodException e) {
            throw new ModelLoadException("Model class missing no-arg constructor: " + className, e);
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException e) {
            throw new ModelLoadException("Failed to instantiate model: " + className, e);
        }
    }

    /**
     * Read the original MLIR source from a model JAR (if present).
     *
     * @param jarPath Path to the model JAR
     * @return The MLIR source, or null if not present
     */
    public static String readSource(Path jarPath) {
        try (JarFile jar = new JarFile(jarPath.toFile())) {
            var entry = jar.getEntry(SOURCE_PATH);
            if (entry == null) {
                return null;
            }
            try (InputStream is = jar.getInputStream(entry)) {
                return new String(is.readAllBytes(), StandardCharsets.UTF_8);
            }
        } catch (IOException e) {
            return null;
        }
    }

    /**
     * ClassLoader that loads a single class from a byte array.
     */
    private static class ByteArrayClassLoader extends ClassLoader {
        private final String className;
        private final byte[] classBytes;

        ByteArrayClassLoader(String className, byte[] classBytes, ClassLoader parent) {
            super(parent);
            this.className = className;
            this.classBytes = classBytes;
        }

        @Override
        protected Class<?> findClass(String name) throws ClassNotFoundException {
            if (name.equals(className)) {
                return defineClass(name, classBytes, 0, classBytes.length);
            }
            throw new ClassNotFoundException(name);
        }
    }

    /**
     * Exception thrown when model loading fails.
     */
    public static class ModelLoadException extends Exception {
        public ModelLoadException(String message) {
            super(message);
        }

        public ModelLoadException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
