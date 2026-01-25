package io.surfworks.warpforge.core.tensor.typed.numerical;

/**
 * Exception thrown when NaN or Inf values are detected in tensor data.
 *
 * <p>This exception is thrown by {@link NumericalContext#check} when the current
 * policy is {@link NaNPolicy#ERROR} and invalid numerical values are found.
 *
 * <p>The exception message includes:
 * <ul>
 *   <li>The operation that produced the invalid values</li>
 *   <li>The type of invalid value (NaN, +Inf, -Inf)</li>
 *   <li>The position where it was first detected (if available)</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * try (var _ = new NumericalContext(NaNPolicy.ERROR)) {
 *     var result = MatrixOps.matmul(a, b);
 *     NumericalContext.check(result, "matmul");
 * } catch (NumericalException e) {
 *     logger.error("Numerical instability detected", e);
 *     // Handle: reduce learning rate, clip gradients, etc.
 * }
 * }</pre>
 */
public class NumericalException extends RuntimeException {

    private final String operation;
    private final InvalidValueType valueType;
    private final long position;

    /**
     * Type of invalid numerical value detected.
     */
    public enum InvalidValueType {
        /** Not a Number */
        NAN,
        /** Positive infinity */
        POSITIVE_INF,
        /** Negative infinity */
        NEGATIVE_INF,
        /** Either NaN or Inf (unspecified) */
        NAN_OR_INF
    }

    /**
     * Creates a NumericalException with the specified details.
     *
     * @param message the exception message
     * @param operation the operation that produced invalid values
     * @param valueType the type of invalid value
     * @param position the position where the value was detected (-1 if unknown)
     */
    public NumericalException(String message, String operation, InvalidValueType valueType, long position) {
        super(message);
        this.operation = operation;
        this.valueType = valueType;
        this.position = position;
    }

    /**
     * Creates a NumericalException with a simple message.
     *
     * @param message the exception message
     */
    public NumericalException(String message) {
        super(message);
        this.operation = "unknown";
        this.valueType = InvalidValueType.NAN_OR_INF;
        this.position = -1;
    }

    /**
     * Creates a NumericalException for the specified operation.
     *
     * @param operation the operation that produced invalid values
     * @param valueType the type of invalid value
     * @param position the position where the value was detected
     */
    public NumericalException(String operation, InvalidValueType valueType, long position) {
        super(formatMessage(operation, valueType, position));
        this.operation = operation;
        this.valueType = valueType;
        this.position = position;
    }

    private static String formatMessage(String operation, InvalidValueType valueType, long position) {
        String valueDesc = switch (valueType) {
            case NAN -> "NaN";
            case POSITIVE_INF -> "+Inf";
            case NEGATIVE_INF -> "-Inf";
            case NAN_OR_INF -> "NaN/Inf";
        };
        if (position >= 0) {
            return String.format("%s detected in %s at position %d", valueDesc, operation, position);
        } else {
            return String.format("%s detected in %s", valueDesc, operation);
        }
    }

    /**
     * Returns the operation that produced invalid values.
     *
     * @return the operation name
     */
    public String operation() {
        return operation;
    }

    /**
     * Returns the type of invalid value that was detected.
     *
     * @return the invalid value type
     */
    public InvalidValueType valueType() {
        return valueType;
    }

    /**
     * Returns the position where the invalid value was first detected.
     *
     * @return the position (flat index), or -1 if unknown
     */
    public long position() {
        return position;
    }
}
