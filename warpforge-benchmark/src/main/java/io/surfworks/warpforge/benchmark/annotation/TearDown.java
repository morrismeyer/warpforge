package io.surfworks.warpforge.benchmark.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a method to be called after benchmark iterations.
 *
 * <p>Similar to JMH's {@code @TearDown}, this method is called to clean up state
 * after the benchmark completes. Use this to free GPU memory, close backends,
 * or release resources.
 *
 * <p>Example:
 * <pre>{@code
 * @TearDown
 * public void cleanup() {
 *     backend.close();
 * }
 * }</pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface TearDown {

    /**
     * When the teardown method should be called.
     */
    Setup.Level level() default Setup.Level.TRIAL;
}
