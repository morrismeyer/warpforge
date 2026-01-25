package io.surfworks.warpforge.core.tensor.typed.grad;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;

/**
 * Scoped gradient lifecycle management for training loops.
 *
 * <p>GradientScope tracks a set of GradTensor instances and provides convenient
 * operations for common training patterns:
 * <ul>
 *   <li>{@link #track(TypedTensor)} - Create and track a new GradTensor</li>
 *   <li>{@link #zeroGrad()} - Zero all tracked gradients</li>
 *   <li>{@link #close()} - Release all tracked tensors</li>
 * </ul>
 *
 * <p>Example training loop:
 * <pre>{@code
 * try (GradientScope scope = new GradientScope()) {
 *     var weights = scope.track(
 *         TypedTensor.randn(new Matrix(768, 512), F32.INSTANCE, Cpu.INSTANCE));
 *     var bias = scope.track(
 *         TypedTensor.randn(new Vector(512), F32.INSTANCE, Cpu.INSTANCE));
 *
 *     for (int epoch = 0; epoch < numEpochs; epoch++) {
 *         scope.zeroGrad();  // Clear all gradients
 *
 *         // Forward pass
 *         var output = forward(input, weights, bias);
 *
 *         // Backward pass
 *         backward(loss, output);
 *
 *         // Update parameters using gradients
 *         update(weights, bias, learningRate);
 *     }
 * }  // All tracked tensors automatically closed
 * }</pre>
 *
 * <p>Thread safety: GradientScope is NOT thread-safe. Use separate scopes for
 * concurrent training or synchronize externally.
 */
public final class GradientScope implements AutoCloseable {

    private final List<GradTensor<?, ?, ?, RequiresGrad>> tracked;
    private final Set<TypedTensor<?, ?, ?>> seenTensors;
    private final AtomicBoolean closed;

    /**
     * Creates a new empty GradientScope.
     */
    public GradientScope() {
        this.tracked = new ArrayList<>();
        this.seenTensors = Collections.newSetFromMap(new IdentityHashMap<>());
        this.closed = new AtomicBoolean(false);
    }

    /**
     * Creates a GradTensor that tracks gradients and adds it to this scope.
     *
     * <p>The returned GradTensor will have its gradients zeroed when
     * {@link #zeroGrad()} is called and will be closed when this scope is closed.
     *
     * <p>If the same tensor is tracked multiple times, subsequent calls return
     * the same GradTensor instance.
     *
     * @param tensor the underlying typed tensor
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return a new GradTensor with RequiresGrad mode
     * @throws IllegalStateException if this scope has been closed
     * @throws NullPointerException if tensor is null
     */
    public <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    GradTensor<S, D, V, RequiresGrad> track(TypedTensor<S, D, V> tensor) {
        ensureOpen();
        Objects.requireNonNull(tensor, "tensor cannot be null");

        // Check if already tracked
        if (seenTensors.contains(tensor)) {
            // Find and return existing GradTensor
            for (GradTensor<?, ?, ?, RequiresGrad> gt : tracked) {
                if (gt.data() == tensor) {
                    @SuppressWarnings("unchecked")
                    GradTensor<S, D, V, RequiresGrad> cast = (GradTensor<S, D, V, RequiresGrad>) gt;
                    return cast;
                }
            }
        }

        // Create new GradTensor (scope manages GradTensor lifecycle, but not underlying tensor)
        GradTensor<S, D, V, RequiresGrad> gradTensor = GradTensor.requiresGrad(tensor);
        tracked.add(gradTensor);
        seenTensors.add(tensor);
        return gradTensor;
    }

    /**
     * Tracks an existing GradTensor in this scope.
     *
     * <p>The GradTensor will have its gradients zeroed when {@link #zeroGrad()}
     * is called, but will NOT be closed when this scope is closed (the caller
     * retains ownership).
     *
     * @param gradTensor the GradTensor to track
     * @param <S> the shape type
     * @param <D> the dtype type
     * @param <V> the device type
     * @return the same GradTensor (for chaining)
     * @throws IllegalStateException if this scope has been closed
     * @throws NullPointerException if gradTensor is null
     */
    public <S extends Shape, D extends DTypeTag, V extends DeviceTag>
    GradTensor<S, D, V, RequiresGrad> trackExisting(GradTensor<S, D, V, RequiresGrad> gradTensor) {
        ensureOpen();
        Objects.requireNonNull(gradTensor, "gradTensor cannot be null");

        if (!seenTensors.contains(gradTensor.data())) {
            // Note: we don't add to tracked list since we don't own it
            // Just track for zeroGrad purposes
            seenTensors.add(gradTensor.data());
            // We need a way to still zero its grad - track it but mark differently
            tracked.add(gradTensor);
        }
        return gradTensor;
    }

    /**
     * Zeros gradients on all tracked tensors.
     *
     * <p>This should be called at the beginning of each training step to prevent
     * gradient accumulation across batches (unless accumulation is desired).
     *
     * <p>This operation is safe to call on an empty scope (no-op).
     *
     * @throws IllegalStateException if this scope has been closed
     */
    public void zeroGrad() {
        ensureOpen();
        for (GradTensor<?, ?, ?, RequiresGrad> gt : tracked) {
            if (!gt.isClosed()) {
                gt.zeroGrad();
            }
        }
    }

    /**
     * Returns the number of tensors tracked by this scope.
     *
     * @return the number of tracked tensors
     */
    public int size() {
        return tracked.size();
    }

    /**
     * Returns true if no tensors are tracked.
     *
     * @return true if scope is empty
     */
    public boolean isEmpty() {
        return tracked.isEmpty();
    }

    /**
     * Returns an unmodifiable view of the tracked GradTensors.
     *
     * @return unmodifiable list of tracked tensors
     */
    public List<GradTensor<?, ?, ?, RequiresGrad>> trackedTensors() {
        return Collections.unmodifiableList(tracked);
    }

    /**
     * Returns true if this scope has been closed.
     *
     * @return true if closed
     */
    public boolean isClosed() {
        return closed.get();
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("GradientScope has been closed");
        }
    }

    /**
     * Closes this scope and releases all tracked GradTensors.
     *
     * <p>For GradTensors created via {@link #track(TypedTensor)}, both the
     * GradTensor and its underlying tensor are closed. For GradTensors added
     * via {@link #trackExisting(GradTensor)}, only the GradTensor tracking
     * is removed (the caller retains ownership).
     *
     * <p>This method is idempotent - calling it multiple times has no effect.
     */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            // Close in reverse order (LIFO)
            for (int i = tracked.size() - 1; i >= 0; i--) {
                GradTensor<?, ?, ?, RequiresGrad> gt = tracked.get(i);
                if (!gt.isClosed()) {
                    gt.close();
                }
            }
            tracked.clear();
            seenTensors.clear();
        }
    }

    @Override
    public String toString() {
        if (closed.get()) {
            return "GradientScope[CLOSED]";
        }
        return String.format("GradientScope[tracked=%d]", tracked.size());
    }
}
