package io.surfworks.warpforge.license;

/**
 * Result of a license activation operation.
 *
 * @param success whether activation succeeded
 * @param license the license info if successful
 * @param error error message if failed
 * @param errorCode machine-readable error code
 */
public record ActivationResult(
    boolean success,
    LicenseInfo license,
    String error,
    ErrorCode errorCode
) {

    public enum ErrorCode {
        NONE,
        INVALID_KEY,
        ACTIVATION_LIMIT_REACHED,
        KEY_EXPIRED,
        KEY_DISABLED,
        NETWORK_ERROR,
        UNKNOWN
    }

    /**
     * Successful activation.
     */
    public static ActivationResult success(LicenseInfo license) {
        return new ActivationResult(true, license, null, ErrorCode.NONE);
    }

    /**
     * Failed activation with error message.
     */
    public static ActivationResult failure(String error) {
        return new ActivationResult(false, null, error, ErrorCode.UNKNOWN);
    }

    /**
     * Failed activation with error code.
     */
    public static ActivationResult failure(String error, ErrorCode code) {
        return new ActivationResult(false, null, error, code);
    }
}
