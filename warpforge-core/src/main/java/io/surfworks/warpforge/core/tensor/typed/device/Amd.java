package io.surfworks.warpforge.core.tensor.typed.device;

/**
 * Phantom type for AMD GPU device.
 *
 * <p>Tensors on AMD GPUs are in device memory and can leverage
 * ROCm/HIP for accelerated computation. Supports multiple GPUs via device index.
 *
 * <p>Example:
 * <pre>{@code
 * // Default GPU (device 0)
 * TypedTensor<Matrix, F32, Amd> gpu0 = TypedTensor.zeros(
 *     new Matrix(1000, 1000), F32.INSTANCE, Amd.DEFAULT);
 *
 * // Specific GPU
 * TypedTensor<Matrix, F32, Amd> gpu1 = TypedTensor.zeros(
 *     new Matrix(1000, 1000), F32.INSTANCE, new Amd(1));
 *
 * // Transfer from CPU
 * TypedTensor<Matrix, F32, Cpu> cpuTensor = ...;
 * TypedTensor<Matrix, F32, Amd> gpuTensor = cpuTensor.to(Amd.DEFAULT);
 * }</pre>
 *
 * @param deviceIndex the HIP device index (0, 1, 2, ...)
 */
public record Amd(int deviceIndex) implements DeviceTag {

    /**
     * Default AMD GPU (device 0).
     */
    public static final Amd DEFAULT = new Amd(0);

    /**
     * Creates an AMD device tag for the specified GPU.
     *
     * @param deviceIndex the HIP device index (must be non-negative)
     * @throws IllegalArgumentException if deviceIndex is negative
     */
    public Amd {
        if (deviceIndex < 0) {
            throw new IllegalArgumentException("Device index must be non-negative, got: " + deviceIndex);
        }
    }

    /**
     * Creates an AMD device tag for the specified GPU.
     *
     * @param index the HIP device index
     * @return the device tag
     */
    public static Amd device(int index) {
        return index == 0 ? DEFAULT : new Amd(index);
    }

    @Override
    public String deviceName() {
        return "amd:" + deviceIndex;
    }

    @Override
    public boolean isGpu() {
        return true;
    }

    @Override
    public String toString() {
        return "Amd[" + deviceIndex + "]";
    }
}
