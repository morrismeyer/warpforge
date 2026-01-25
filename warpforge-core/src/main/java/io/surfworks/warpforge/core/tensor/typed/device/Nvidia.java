package io.surfworks.warpforge.core.tensor.typed.device;

/**
 * Phantom type for NVIDIA GPU device.
 *
 * <p>Tensors on NVIDIA GPUs are in device memory and can leverage
 * CUDA for accelerated computation. Supports multiple GPUs via device index.
 *
 * <p>Example:
 * <pre>{@code
 * // Default GPU (device 0)
 * TypedTensor<Matrix, F32, Nvidia> gpu0 = TypedTensor.zeros(
 *     new Matrix(1000, 1000), F32.INSTANCE, Nvidia.DEFAULT);
 *
 * // Specific GPU
 * TypedTensor<Matrix, F32, Nvidia> gpu1 = TypedTensor.zeros(
 *     new Matrix(1000, 1000), F32.INSTANCE, new Nvidia(1));
 *
 * // Transfer from CPU
 * TypedTensor<Matrix, F32, Cpu> cpuTensor = ...;
 * TypedTensor<Matrix, F32, Nvidia> gpuTensor = cpuTensor.to(Nvidia.DEFAULT);
 * }</pre>
 *
 * @param deviceIndex the CUDA device index (0, 1, 2, ...)
 */
public record Nvidia(int deviceIndex) implements DeviceTag {

    /**
     * Default NVIDIA GPU (device 0).
     */
    public static final Nvidia DEFAULT = new Nvidia(0);

    /**
     * Creates an NVIDIA device tag for the specified GPU.
     *
     * @param deviceIndex the CUDA device index (must be non-negative)
     * @throws IllegalArgumentException if deviceIndex is negative
     */
    public Nvidia {
        if (deviceIndex < 0) {
            throw new IllegalArgumentException("Device index must be non-negative, got: " + deviceIndex);
        }
    }

    /**
     * Creates an NVIDIA device tag for the specified GPU.
     *
     * @param index the CUDA device index
     * @return the device tag
     */
    public static Nvidia device(int index) {
        return index == 0 ? DEFAULT : new Nvidia(index);
    }

    @Override
    public String deviceName() {
        return "nvidia:" + deviceIndex;
    }

    @Override
    public boolean isGpu() {
        return true;
    }

    @Override
    public String toString() {
        return "Nvidia[" + deviceIndex + "]";
    }
}
