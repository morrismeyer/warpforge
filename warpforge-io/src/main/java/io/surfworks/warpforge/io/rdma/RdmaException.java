package io.surfworks.warpforge.io.rdma;

/**
 * Exception thrown when an RDMA operation fails.
 *
 * <p>This exception wraps errors from the underlying UCX/ibverbs libraries
 * and provides meaningful error messages for diagnosis.
 */
public class RdmaException extends RuntimeException {

    private final ErrorCode errorCode;
    private final int nativeErrorCode;

    public RdmaException(String message) {
        super(message);
        this.errorCode = ErrorCode.UNKNOWN;
        this.nativeErrorCode = 0;
    }

    public RdmaException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = ErrorCode.UNKNOWN;
        this.nativeErrorCode = 0;
    }

    public RdmaException(String message, ErrorCode errorCode) {
        super(message);
        this.errorCode = errorCode;
        this.nativeErrorCode = 0;
    }

    public RdmaException(String message, ErrorCode errorCode, int nativeErrorCode) {
        super(message);
        this.errorCode = errorCode;
        this.nativeErrorCode = nativeErrorCode;
    }

    public RdmaException(String message, ErrorCode errorCode, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
        this.nativeErrorCode = 0;
    }

    /**
     * Returns the error code category.
     */
    public ErrorCode errorCode() {
        return errorCode;
    }

    /**
     * Returns the native error code from UCX/ibverbs, or 0 if not applicable.
     */
    public int nativeErrorCode() {
        return nativeErrorCode;
    }

    /**
     * RDMA error code categories.
     */
    public enum ErrorCode {
        /** Unknown or unclassified error */
        UNKNOWN,

        /** No RDMA devices found on this system */
        NO_DEVICE,

        /** Specified device not found */
        DEVICE_NOT_FOUND,

        /** Memory registration failed */
        MEMORY_REGISTRATION_FAILED,

        /** Connection establishment failed */
        CONNECTION_FAILED,

        /** Connection was reset or lost */
        CONNECTION_RESET,

        /** Connection timeout */
        TIMEOUT,

        /** Operation was cancelled */
        CANCELLED,

        /** Remote peer rejected operation */
        REMOTE_ERROR,

        /** Invalid argument provided */
        INVALID_ARGUMENT,

        /** Requested feature not supported */
        NOT_SUPPORTED,

        /** Out of resources (memory, queue pairs, etc.) */
        RESOURCE_EXHAUSTED,

        /** Operation would block but non-blocking requested */
        WOULD_BLOCK,

        /** Endpoint is in wrong state for operation */
        INVALID_STATE,

        /** Access permission denied */
        ACCESS_DENIED,

        /** Address/port already in use */
        ADDRESS_IN_USE,

        /** Network unreachable */
        NETWORK_UNREACHABLE
    }

    /**
     * Creates an exception for no RDMA devices found.
     */
    public static RdmaException noDevice() {
        return new RdmaException("No RDMA devices found on this system", ErrorCode.NO_DEVICE);
    }

    /**
     * Creates an exception for device not found.
     */
    public static RdmaException deviceNotFound(String deviceName) {
        return new RdmaException("RDMA device not found: " + deviceName, ErrorCode.DEVICE_NOT_FOUND);
    }

    /**
     * Creates an exception for connection failure.
     */
    public static RdmaException connectionFailed(String address, int port, String reason) {
        return new RdmaException(
                String.format("Failed to connect to %s:%d: %s", address, port, reason),
                ErrorCode.CONNECTION_FAILED);
    }

    /**
     * Creates an exception for connection timeout.
     */
    public static RdmaException timeout(String operation) {
        return new RdmaException("Timeout during: " + operation, ErrorCode.TIMEOUT);
    }

    /**
     * Creates an exception for memory registration failure.
     */
    public static RdmaException memoryRegistrationFailed(long size, String reason) {
        return new RdmaException(
                String.format("Failed to register %d bytes for RDMA: %s", size, reason),
                ErrorCode.MEMORY_REGISTRATION_FAILED);
    }
}
