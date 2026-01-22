package io.surfworks.warpforge.io.collective;

/**
 * Exception thrown when a collective operation fails.
 */
public class CollectiveException extends RuntimeException {

    private final ErrorCode errorCode;
    private final int nativeErrorCode;

    public CollectiveException(String message) {
        super(message);
        this.errorCode = ErrorCode.UNKNOWN;
        this.nativeErrorCode = 0;
    }

    public CollectiveException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = ErrorCode.UNKNOWN;
        this.nativeErrorCode = 0;
    }

    public CollectiveException(String message, ErrorCode errorCode) {
        super(message);
        this.errorCode = errorCode;
        this.nativeErrorCode = 0;
    }

    public CollectiveException(String message, ErrorCode errorCode, int nativeErrorCode) {
        super(message);
        this.errorCode = errorCode;
        this.nativeErrorCode = nativeErrorCode;
    }

    public CollectiveException(String message, ErrorCode errorCode, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
        this.nativeErrorCode = 0;
    }

    public ErrorCode errorCode() {
        return errorCode;
    }

    public int nativeErrorCode() {
        return nativeErrorCode;
    }

    /**
     * Collective operation error codes.
     */
    public enum ErrorCode {
        /** Unknown or unclassified error */
        UNKNOWN,

        /** Not initialized */
        NOT_INITIALIZED,

        /** Already initialized */
        ALREADY_INITIALIZED,

        /** Invalid rank */
        INVALID_RANK,

        /** Invalid world size */
        INVALID_WORLD_SIZE,

        /** Tensor shape mismatch across ranks */
        SHAPE_MISMATCH,

        /** Tensor dtype mismatch across ranks */
        DTYPE_MISMATCH,

        /** Operation timed out */
        TIMEOUT,

        /** Communication failure */
        COMMUNICATION_ERROR,

        /** Out of resources */
        RESOURCE_EXHAUSTED,

        /** Operation not supported */
        NOT_SUPPORTED,

        /** Invalid state for operation */
        INVALID_STATE,

        /** Synchronization failure */
        SYNC_ERROR
    }

    public static CollectiveException notInitialized() {
        return new CollectiveException("Collective context not initialized", ErrorCode.NOT_INITIALIZED);
    }

    public static CollectiveException invalidRank(int rank, int worldSize) {
        return new CollectiveException(
                String.format("Invalid rank %d for world size %d", rank, worldSize),
                ErrorCode.INVALID_RANK);
    }

    public static CollectiveException shapeMismatch(String expected, String actual) {
        return new CollectiveException(
                String.format("Tensor shape mismatch: expected %s, got %s", expected, actual),
                ErrorCode.SHAPE_MISMATCH);
    }

    public static CollectiveException timeout(String operation, long timeoutMs) {
        return new CollectiveException(
                String.format("Collective operation '%s' timed out after %d ms", operation, timeoutMs),
                ErrorCode.TIMEOUT);
    }
}
