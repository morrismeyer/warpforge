package io.surfworks.warpforge.core.tensor.typed.device;

/**
 * Phantom type for device-agnostic tensors.
 *
 * <p>Used for generic code that works regardless of device placement.
 * This is an escape hatch for interoperability and should be used sparingly,
 * as it opts out of compile-time device checking.
 *
 * <p>Example:
 * <pre>{@code
 * // Generic function that works with any device
 * public static <V extends DeviceTag>
 * TypedTensor<Matrix, F32, V> normalize(TypedTensor<Matrix, F32, V> input) {
 *     // ... implementation
 * }
 *
 * // Loading from file where device isn't known
 * Tensor untyped = loadFromFile("weights.pt");
 * TypedTensor<Matrix, F32, AnyDevice> generic = TypedTensor.from(
 *     untyped, new Matrix(768, 512), F32.INSTANCE, AnyDevice.INSTANCE);
 * }</pre>
 */
public record AnyDevice() implements DeviceTag {

    /**
     * Singleton instance for any-device.
     */
    public static final AnyDevice INSTANCE = new AnyDevice();

    @Override
    public String deviceName() {
        return "any";
    }

    @Override
    public boolean isGpu() {
        return false; // Conservative: assume CPU semantics
    }

    @Override
    public int deviceIndex() {
        return 0;
    }

    @Override
    public String toString() {
        return "AnyDevice";
    }
}
