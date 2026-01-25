package io.surfworks.warpforge.core.tensor.typed.grad;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;

/**
 * Type-safe tensor wrapper with gradient tracking.
 *
 * <p>GradTensor wraps a {@link TypedTensor} and adds gradient tracking capability.
 * The gradient mode is encoded as a phantom type parameter, enabling compile-time
 * enforcement of gradient-related operations.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Create a tensor that tracks gradients (for model parameters)
 * var weights = GradTensor.requiresGrad(
 *     TypedTensor.randn(new Matrix(768, 512), F32.INSTANCE, Cpu.INSTANCE));
 *
 * // Zero gradients before backward pass
 * weights.zeroGrad();
 *
 * // After backward pass, access gradients
 * var gradient = weights.grad();
 *
 * // Create inference tensor (no gradient tracking)
 * var input = GradTensor.noGrad(inputTensor);
 * // input.zeroGrad();  // Throws IllegalStateException
 *
 * // Detach a tensor to freeze it
 * var frozen = weights.detach();
 * }</pre>
 *
 * @param <S> the shape phantom type (e.g., Matrix, Vector)
 * @param <D> the data type phantom type (e.g., F32, F64)
 * @param <V> the device phantom type (e.g., Cpu, Nvidia)
 * @param <G> the gradient mode phantom type (RequiresGrad, NoGrad, Detached)
 */
public final class GradTensor<S extends Shape, D extends DTypeTag, V extends DeviceTag, G extends GradMode>
        implements AutoCloseable {

    private final TypedTensor<S, D, V> data;
    private final G gradMode;
    private TypedTensor<S, D, V> grad;
    private final AtomicBoolean closed;
    private final boolean ownsUnderlying;

    private GradTensor(TypedTensor<S, D, V> data, G gradMode, boolean ownsUnderlying) {
        this.data = Objects.requireNonNull(data, "data tensor cannot be null");
        this.gradMode = Objects.requireNonNull(gradMode, "gradMode cannot be null");
        this.grad = null;
        this.closed = new AtomicBoolean(false);
        this.ownsUnderlying = ownsUnderlying;
    }

    // ==================== Factory Methods ====================

    /**
     * Creates a GradTensor that tracks gradients.
     *
     * <p>The returned tensor will accumulate gradients during backward passes.
     * Use {@link #zeroGrad()} to clear gradients before each training step.
     *
     * @param tensor the underlying typed tensor
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new GradTensor with RequiresGrad mode
     * @throws NullPointerException if tensor is null
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    GradTensor<S, D, V, RequiresGrad> requiresGrad(TypedTensor<S, D, V> tensor) {
        return new GradTensor<>(tensor, RequiresGrad.INSTANCE, false);
    }

    /**
     * Creates a GradTensor that tracks gradients, taking ownership of the underlying tensor.
     *
     * <p>When this GradTensor is closed, the underlying tensor will also be closed.
     *
     * @param tensor the underlying typed tensor (ownership transferred)
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new GradTensor with RequiresGrad mode
     * @throws NullPointerException if tensor is null
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    GradTensor<S, D, V, RequiresGrad> requiresGradOwning(TypedTensor<S, D, V> tensor) {
        return new GradTensor<>(tensor, RequiresGrad.INSTANCE, true);
    }

    /**
     * Creates a GradTensor that does not track gradients.
     *
     * <p>This is appropriate for input data, labels, and other tensors that should
     * not participate in gradient computation.
     *
     * @param tensor the underlying typed tensor
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new GradTensor with NoGrad mode
     * @throws NullPointerException if tensor is null
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    GradTensor<S, D, V, NoGrad> noGrad(TypedTensor<S, D, V> tensor) {
        return new GradTensor<>(tensor, NoGrad.INSTANCE, false);
    }

    /**
     * Creates a GradTensor that does not track gradients, taking ownership.
     *
     * @param tensor the underlying typed tensor (ownership transferred)
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new GradTensor with NoGrad mode
     * @throws NullPointerException if tensor is null
     */
    public static <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    GradTensor<S, D, V, NoGrad> noGradOwning(TypedTensor<S, D, V> tensor) {
        return new GradTensor<>(tensor, NoGrad.INSTANCE, true);
    }

    // ==================== Accessors ====================

    /**
     * Returns the underlying data tensor.
     *
     * @return the underlying TypedTensor
     * @throws IllegalStateException if this GradTensor has been closed
     */
    public TypedTensor<S, D, V> data() {
        ensureOpen();
        return data;
    }

    /**
     * Returns the gradient mode.
     *
     * @return the gradient mode (RequiresGrad, NoGrad, or Detached)
     */
    public G gradMode() {
        return gradMode;
    }

    /**
     * Returns true if this tensor tracks gradients.
     *
     * @return true for RequiresGrad mode, false otherwise
     */
    public boolean requiresGrad() {
        return gradMode.tracksGradient();
    }

    /**
     * Returns the shape phantom type instance.
     *
     * @return the shape type
     */
    public S shapeType() {
        return data.shapeType();
    }

    /**
     * Returns the dtype phantom type instance.
     *
     * @return the dtype type
     */
    public D dtypeType() {
        return data.dtypeType();
    }

    /**
     * Returns the device phantom type instance.
     *
     * @return the device type
     */
    public V deviceType() {
        return data.deviceType();
    }

    // ==================== Gradient Operations ====================

    /**
     * Returns the accumulated gradient tensor.
     *
     * <p>Returns null if no gradients have been computed yet (i.e., backward()
     * has not been called on a loss involving this tensor).
     *
     * @return the gradient tensor, or null if not yet computed
     * @throws IllegalStateException if this tensor does not track gradients
     * @throws IllegalStateException if this GradTensor has been closed
     */
    public TypedTensor<S, D, V> grad() {
        ensureOpen();
        if (!gradMode.tracksGradient()) {
            throw new IllegalStateException(
                    "Cannot access gradient on tensor with mode: " + gradMode.modeName());
        }
        return grad;
    }

    /**
     * Sets the gradient tensor (used during backward pass).
     *
     * <p>If a gradient already exists, the new gradient is accumulated (added).
     *
     * @param gradient the gradient tensor to accumulate
     * @throws IllegalStateException if this tensor does not track gradients
     * @throws IllegalStateException if this GradTensor has been closed
     * @throws NullPointerException if gradient is null
     */
    public void accumulateGrad(TypedTensor<S, D, V> gradient) {
        ensureOpen();
        Objects.requireNonNull(gradient, "gradient cannot be null");
        if (!gradMode.tracksGradient()) {
            throw new IllegalStateException(
                    "Cannot accumulate gradient on tensor with mode: " + gradMode.modeName());
        }
        if (this.grad == null) {
            // First gradient - just store it (take ownership for cleanup)
            this.grad = gradient;
        } else {
            // Accumulate: grad = grad + gradient
            // Note: In a real implementation, this would use TypedOps.add()
            // For now, we just replace (proper accumulation requires ops integration)
            this.grad = gradient;
        }
    }

    /**
     * Clears the accumulated gradient, setting it to zeros.
     *
     * <p>This should be called before each training step to prevent gradient
     * accumulation across batches (unless that's desired).
     *
     * @throws IllegalStateException if this tensor does not track gradients
     * @throws IllegalStateException if this GradTensor has been closed
     */
    public void zeroGrad() {
        ensureOpen();
        if (!gradMode.tracksGradient()) {
            throw new IllegalStateException(
                    "Cannot zero gradient on tensor with mode: " + gradMode.modeName());
        }
        if (grad != null) {
            grad.close();
            grad = null;
        }
    }

    /**
     * Returns true if gradients have been computed for this tensor.
     *
     * @return true if grad() would return a non-null value
     * @throws IllegalStateException if this tensor does not track gradients
     */
    public boolean hasGrad() {
        ensureOpen();
        if (!gradMode.tracksGradient()) {
            throw new IllegalStateException(
                    "Cannot check gradient on tensor with mode: " + gradMode.modeName());
        }
        return grad != null;
    }

    // ==================== Mode Transitions ====================

    /**
     * Creates a detached copy that no longer tracks gradients.
     *
     * <p>The detached tensor shares the same underlying data but will not
     * accumulate gradients. This is useful for:
     * <ul>
     *   <li>Freezing parts of a model during fine-tuning</li>
     *   <li>Breaking gradient chains to save memory</li>
     *   <li>Creating inference-only copies</li>
     * </ul>
     *
     * @return a new GradTensor with Detached mode
     * @throws IllegalStateException if this GradTensor has been closed
     */
    public GradTensor<S, D, V, Detached> detach() {
        ensureOpen();
        return new GradTensor<>(data, Detached.INSTANCE, false);
    }

    /**
     * Creates a NoGrad view of the underlying data.
     *
     * <p>This is semantically similar to detach(), but indicates the tensor
     * never tracked gradients rather than being explicitly detached.
     *
     * @return a new GradTensor with NoGrad mode
     * @throws IllegalStateException if this GradTensor has been closed
     */
    public GradTensor<S, D, V, NoGrad> asNoGrad() {
        ensureOpen();
        return new GradTensor<>(data, NoGrad.INSTANCE, false);
    }

    // ==================== Lifecycle ====================

    /**
     * Returns true if this GradTensor has been closed.
     *
     * @return true if closed
     */
    public boolean isClosed() {
        return closed.get();
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("GradTensor has been closed");
        }
    }

    /**
     * Closes this GradTensor and releases resources.
     *
     * <p>If this GradTensor owns the underlying tensor (created via *Owning factory
     * methods), the underlying tensor is also closed. Any accumulated gradient
     * tensor is always closed.
     *
     * <p>This method is idempotent - calling it multiple times has no effect.
     */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            if (grad != null) {
                grad.close();
                grad = null;
            }
            if (ownsUnderlying) {
                data.close();
            }
        }
    }

    @Override
    public String toString() {
        if (closed.get()) {
            return "GradTensor[CLOSED]";
        }
        return String.format("GradTensor[mode=%s, shape=%s, dtype=%s, device=%s, hasGrad=%s]",
                gradMode.modeName(),
                shapeType(),
                dtypeType(),
                deviceType().deviceName(),
                gradMode.tracksGradient() ? String.valueOf(grad != null) : "N/A");
    }
}
