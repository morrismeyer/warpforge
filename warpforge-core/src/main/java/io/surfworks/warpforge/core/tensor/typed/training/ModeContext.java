package io.surfworks.warpforge.core.tensor.typed.training;

import java.util.Objects;

/**
 * Thread-local context for training/inference mode selection.
 *
 * <p>ModeContext provides a scoped way to switch between training and inference
 * modes. The context is thread-local, so different threads can have different
 * modes simultaneously.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Default is Training mode
 * assert ModeContext.isTraining();
 *
 * // Switch to Inference for evaluation
 * try (var _ = new ModeContext(Inference.INSTANCE)) {
 *     assert !ModeContext.isTraining();
 *     var testOutput = model.forward(testInput);
 *     var accuracy = computeAccuracy(testOutput, testLabels);
 * }
 *
 * // Automatically restored to Training
 * assert ModeContext.isTraining();
 * }</pre>
 *
 * <p>Nesting is supported:
 * <pre>{@code
 * try (var _ = new ModeContext(Inference.INSTANCE)) {
 *     // Inference mode
 *     try (var _ = new ModeContext(Training.INSTANCE)) {
 *         // Back to Training mode temporarily
 *     }
 *     // Inference mode again
 * }
 * // Training mode (default)
 * }</pre>
 *
 * <p>Thread safety: Each thread has its own mode context. Creating a ModeContext
 * on one thread does not affect other threads.
 *
 * <p><strong>Warning:</strong> Failing to close a ModeContext (e.g., due to an
 * exception without try-with-resources) will leave the mode set incorrectly for
 * the current thread. Always use try-with-resources.
 */
public final class ModeContext implements AutoCloseable {

    private static final ThreadLocal<Mode> CURRENT = ThreadLocal.withInitial(() -> Training.INSTANCE);

    private final Mode previous;
    private boolean closed;

    /**
     * Creates a new mode context, setting the current mode for this thread.
     *
     * <p>The previous mode is saved and will be restored when this context is closed.
     *
     * @param mode the mode to set
     * @throws NullPointerException if mode is null
     */
    public ModeContext(Mode mode) {
        Objects.requireNonNull(mode, "mode cannot be null");
        this.previous = CURRENT.get();
        this.closed = false;
        CURRENT.set(mode);
    }

    /**
     * Returns the current mode for this thread.
     *
     * @return the current mode (Training or Inference)
     */
    public static Mode current() {
        return CURRENT.get();
    }

    /**
     * Returns true if the current mode is Training.
     *
     * @return true if in training mode
     */
    public static boolean isTraining() {
        return CURRENT.get().isTraining();
    }

    /**
     * Returns true if the current mode is Inference.
     *
     * @return true if in inference mode
     */
    public static boolean isInference() {
        return !CURRENT.get().isTraining();
    }

    /**
     * Convenience method to create an inference mode context.
     *
     * <p>Equivalent to {@code new ModeContext(Inference.INSTANCE)}.
     *
     * @return a new ModeContext set to Inference mode
     */
    public static ModeContext inference() {
        return new ModeContext(Inference.INSTANCE);
    }

    /**
     * Convenience method to create a training mode context.
     *
     * <p>Equivalent to {@code new ModeContext(Training.INSTANCE)}.
     *
     * @return a new ModeContext set to Training mode
     */
    public static ModeContext training() {
        return new ModeContext(Training.INSTANCE);
    }

    /**
     * Restores the previous mode for this thread.
     *
     * <p>This method is idempotent - calling it multiple times has no effect
     * after the first call.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            CURRENT.set(previous);
        }
    }

    /**
     * Returns true if this context has been closed.
     *
     * @return true if closed
     */
    public boolean isClosed() {
        return closed;
    }

    @Override
    public String toString() {
        if (closed) {
            return "ModeContext[CLOSED, previous=" + previous.modeName() + "]";
        }
        return String.format("ModeContext[current=%s, previous=%s]",
                CURRENT.get().modeName(), previous.modeName());
    }
}
