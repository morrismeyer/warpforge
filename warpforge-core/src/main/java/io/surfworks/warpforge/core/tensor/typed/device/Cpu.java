package io.surfworks.warpforge.core.tensor.typed.device;

/**
 * Phantom type for CPU/host memory device.
 *
 * <p>Tensors on CPU are in system RAM and can be directly accessed
 * from Java code. CPU is the default device for most operations.
 *
 * <p>Example:
 * <pre>{@code
 * TypedTensor<Matrix, F32, Cpu> cpuTensor = TypedTensor.zeros(
 *     new Matrix(100, 100), F32.INSTANCE, Cpu.INSTANCE);
 *
 * // Direct element access works on CPU tensors
 * float value = cpuTensor.underlying().getFloat(0, 0);
 * }</pre>
 */
public record Cpu() implements DeviceTag {

    /**
     * Singleton instance for CPU device.
     */
    public static final Cpu INSTANCE = new Cpu();

    @Override
    public String deviceName() {
        return "cpu";
    }

    @Override
    public boolean isGpu() {
        return false;
    }

    @Override
    public int deviceIndex() {
        return 0;
    }

    @Override
    public String toString() {
        return "Cpu";
    }
}
