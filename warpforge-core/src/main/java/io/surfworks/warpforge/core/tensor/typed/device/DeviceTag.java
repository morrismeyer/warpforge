package io.surfworks.warpforge.core.tensor.typed.device;

/**
 * Phantom type representing tensor device placement at compile time.
 *
 * <p>This sealed interface provides type-level differentiation between
 * device locations, enabling compile-time checking to prevent cross-device
 * operations that would fail at runtime.
 *
 * <p>Example usage:
 * <pre>{@code
 * TypedTensor<Matrix, F32, Cpu> cpuTensor = ...;
 * TypedTensor<Matrix, F32, Nvidia> gpuTensor = ...;
 *
 * // This would NOT compile - device mismatch:
 * // TypedOps.add(cpuTensor, gpuTensor);
 *
 * // Explicit transfer required:
 * TypedTensor<Matrix, F32, Nvidia> transferred = cpuTensor.to(Nvidia.DEFAULT);
 * TypedOps.add(transferred, gpuTensor);  // OK
 * }</pre>
 *
 * @see Cpu for CPU/host memory
 * @see Nvidia for NVIDIA GPU memory
 * @see Amd for AMD GPU memory
 * @see AnyDevice for device-agnostic code
 */
public sealed interface DeviceTag permits Cpu, Nvidia, Amd, AnyDevice {

    /**
     * Returns the device name (e.g., "cpu", "nvidia:0", "amd:1").
     *
     * @return the device identifier string
     */
    String deviceName();

    /**
     * Returns true if this is a GPU device.
     *
     * @return true for Nvidia and Amd, false for Cpu and AnyDevice
     */
    default boolean isGpu() {
        return false;
    }

    /**
     * Returns the device index for multi-device setups.
     * Returns 0 for CPU and AnyDevice.
     *
     * @return the device index
     */
    default int deviceIndex() {
        return 0;
    }
}
