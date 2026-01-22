package io.surfworks.warpforge.benchmark.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a method to be called before benchmark iterations.
 *
 * <p>Similar to JMH's {@code @Setup}, this method is called to prepare state
 * before the benchmark runs. Use this to allocate tensors, initialize backends,
 * or prepare input data.
 *
 * <p>Example:
 * <pre>{@code
 * @Setup
 * public void setupTensors() {
 *     this.inputA = Tensor.random(DType.F32, new int[]{4096, 4096});
 *     this.inputB = Tensor.random(DType.F32, new int[]{4096, 4096});
 * }
 * }</pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Setup {

    /**
     * When the setup method should be called.
     */
    Level level() default Level.TRIAL;

    enum Level {
        /**
         * Called once before all iterations for all tiers.
         */
        TRIAL,

        /**
         * Called once before each tier's iterations.
         */
        TIER,

        /**
         * Called before each iteration (expensive, use sparingly).
         */
        ITERATION
    }
}
