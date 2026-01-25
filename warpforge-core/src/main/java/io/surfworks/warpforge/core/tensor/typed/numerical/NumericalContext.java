package io.surfworks.warpforge.core.tensor.typed.numerical;

import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;

/**
 * Thread-local context for NaN/Inf checking with configurable policy.
 *
 * <p>NumericalContext provides a scoped way to enable numerical validation.
 * The context is thread-local, so different threads can have different policies.
 *
 * <p>Example usage:
 * <pre>{@code
 * // During debugging: fail fast on numerical issues
 * try (var _ = new NumericalContext(NaNPolicy.ERROR)) {
 *     var output = model.forward(input);
 *     NumericalContext.check(output, "forward");  // throws if NaN/Inf
 *
 *     var loss = computeLoss(output, labels);
 *     NumericalContext.check(loss, "loss");
 *
 *     backward(loss);
 *     NumericalContext.check(weights.grad(), "gradients");
 * }
 *
 * // Production: no overhead
 * try (var _ = new NumericalContext(NaNPolicy.IGNORE)) {
 *     var output = model.forward(input);
 *     // No checking - maximum performance
 * }
 * }</pre>
 *
 * <p>Thread safety: Each thread has its own policy. Creating a NumericalContext
 * on one thread does not affect other threads.
 */
public final class NumericalContext implements AutoCloseable {

    private static final Logger LOGGER = Logger.getLogger(NumericalContext.class.getName());

    private static final ThreadLocal<NaNPolicy> POLICY =
            ThreadLocal.withInitial(() -> NaNPolicy.IGNORE);

    private final NaNPolicy previous;
    private boolean closed;

    /**
     * Creates a new numerical context, setting the policy for this thread.
     *
     * <p>The previous policy is saved and will be restored when this context is closed.
     *
     * @param policy the NaN/Inf handling policy
     * @throws NullPointerException if policy is null
     */
    public NumericalContext(NaNPolicy policy) {
        Objects.requireNonNull(policy, "policy cannot be null");
        this.previous = POLICY.get();
        this.closed = false;
        POLICY.set(policy);
    }

    /**
     * Returns the current NaN policy for this thread.
     *
     * @return the current policy
     */
    public static NaNPolicy currentPolicy() {
        return POLICY.get();
    }

    /**
     * Checks a tensor for NaN or Inf values according to the current policy.
     *
     * <p>Behavior by policy:
     * <ul>
     *   <li>{@link NaNPolicy#ERROR} - Throws {@link NumericalException}</li>
     *   <li>{@link NaNPolicy#WARN} - Logs a warning</li>
     *   <li>{@link NaNPolicy#IGNORE} - Returns immediately (no checking)</li>
     * </ul>
     *
     * @param tensor the tensor to check
     * @param operation description of the operation that produced this tensor
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @throws NumericalException if NaN/Inf detected and policy is ERROR
     * @throws NullPointerException if tensor or operation is null
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    void check(TypedTensor<S, D, V> tensor, String operation) {
        NaNPolicy policy = POLICY.get();
        if (policy == NaNPolicy.IGNORE) {
            return;
        }

        Objects.requireNonNull(tensor, "tensor cannot be null");
        Objects.requireNonNull(operation, "operation cannot be null");

        CheckResult result = checkForInvalidValues(tensor);
        if (result != null) {
            handleInvalidValue(operation, result, policy);
        }
    }

    /**
     * Checks if a tensor contains any NaN or Inf values.
     *
     * <p>Unlike {@link #check}, this method always performs the check regardless
     * of the current policy. Use this for explicit validation outside the policy
     * framework.
     *
     * @param tensor the tensor to check
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return true if the tensor contains NaN or Inf values
     * @throws NullPointerException if tensor is null
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    boolean containsNaNOrInf(TypedTensor<S, D, V> tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return checkForInvalidValues(tensor) != null;
    }

    /**
     * Checks if a tensor contains any NaN values (not Inf).
     *
     * @param tensor the tensor to check
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return true if the tensor contains NaN values
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    boolean containsNaN(TypedTensor<S, D, V> tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        CheckResult result = checkForInvalidValues(tensor);
        return result != null && result.valueType == NumericalException.InvalidValueType.NAN;
    }

    /**
     * Checks if a tensor contains any Inf values (not NaN).
     *
     * @param tensor the tensor to check
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return true if the tensor contains Inf values
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    boolean containsInf(TypedTensor<S, D, V> tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        CheckResult result = checkForInvalidValues(tensor);
        return result != null && (result.valueType == NumericalException.InvalidValueType.POSITIVE_INF
                || result.valueType == NumericalException.InvalidValueType.NEGATIVE_INF);
    }

    private static record CheckResult(NumericalException.InvalidValueType valueType, long position) {}

    private static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    CheckResult checkForInvalidValues(TypedTensor<S, D, V> tensor) {
        DTypeTag dtype = tensor.dtypeType();
        long elementCount = tensor.elementCount();

        if (elementCount == 0) {
            return null;  // Empty tensor has no invalid values
        }

        // Check based on dtype
        if (dtype instanceof F32) {
            return checkFloatTensor(tensor);
        } else if (dtype instanceof F64) {
            return checkDoubleTensor(tensor);
        }
        // Integer types cannot have NaN/Inf
        return null;
    }

    private static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    CheckResult checkFloatTensor(TypedTensor<S, D, V> tensor) {
        float[] data = tensor.underlying().toFloatArray();
        for (int i = 0; i < data.length; i++) {
            float v = data[i];
            if (Float.isNaN(v)) {
                return new CheckResult(NumericalException.InvalidValueType.NAN, i);
            }
            if (v == Float.POSITIVE_INFINITY) {
                return new CheckResult(NumericalException.InvalidValueType.POSITIVE_INF, i);
            }
            if (v == Float.NEGATIVE_INFINITY) {
                return new CheckResult(NumericalException.InvalidValueType.NEGATIVE_INF, i);
            }
        }
        return null;
    }

    private static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    CheckResult checkDoubleTensor(TypedTensor<S, D, V> tensor) {
        double[] data = tensor.underlying().toDoubleArray();
        for (int i = 0; i < data.length; i++) {
            double v = data[i];
            if (Double.isNaN(v)) {
                return new CheckResult(NumericalException.InvalidValueType.NAN, i);
            }
            if (v == Double.POSITIVE_INFINITY) {
                return new CheckResult(NumericalException.InvalidValueType.POSITIVE_INF, i);
            }
            if (v == Double.NEGATIVE_INFINITY) {
                return new CheckResult(NumericalException.InvalidValueType.NEGATIVE_INF, i);
            }
        }
        return null;
    }

    private static void handleInvalidValue(String operation, CheckResult result, NaNPolicy policy) {
        if (policy == NaNPolicy.ERROR) {
            throw new NumericalException(operation, result.valueType, result.position);
        } else if (policy == NaNPolicy.WARN) {
            String valueDesc = switch (result.valueType) {
                case NAN -> "NaN";
                case POSITIVE_INF -> "+Inf";
                case NEGATIVE_INF -> "-Inf";
                case NAN_OR_INF -> "NaN/Inf";
            };
            LOGGER.log(Level.WARNING,
                    "{0} detected in {1} at position {2}",
                    new Object[]{valueDesc, operation, result.position});
        }
    }

    /**
     * Convenience method to create an ERROR policy context.
     *
     * @return a new NumericalContext with ERROR policy
     */
    public static NumericalContext errorOnInvalid() {
        return new NumericalContext(NaNPolicy.ERROR);
    }

    /**
     * Convenience method to create a WARN policy context.
     *
     * @return a new NumericalContext with WARN policy
     */
    public static NumericalContext warnOnInvalid() {
        return new NumericalContext(NaNPolicy.WARN);
    }

    /**
     * Returns true if this context has been closed.
     *
     * @return true if closed
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Restores the previous policy for this thread.
     *
     * <p>This method is idempotent - calling it multiple times has no effect
     * after the first call.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            POLICY.set(previous);
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "NumericalContext[CLOSED, previous=" + previous + "]";
        }
        return String.format("NumericalContext[current=%s, previous=%s]",
                POLICY.get(), previous);
    }
}
