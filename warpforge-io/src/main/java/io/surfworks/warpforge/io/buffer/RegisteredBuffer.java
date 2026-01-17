package io.surfworks.warpforge.io.buffer;

import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;
import io.surfworks.warpforge.io.rdma.RdmaApi;
import io.surfworks.warpforge.io.rdma.RdmaBuffer;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * A Tensor-compatible buffer that is RDMA-registered for zero-copy transfers.
 *
 * <p>This class bridges the WarpForge tensor system with RDMA networking,
 * enabling direct memory access between nodes without CPU-mediated copies.
 * The underlying MemorySegment is registered with the RDMA device and can
 * be used for both local tensor operations and remote RDMA transfers.
 *
 * <h2>Zero-Copy Path</h2>
 * <pre>
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  RegisteredBuffer                                               │
 *   │                                                                 │
 *   │  ┌─────────────────┐    ┌─────────────────┐                    │
 *   │  │     Tensor      │    │   RdmaBuffer    │                    │
 *   │  │  (computation)  │    │  (networking)   │                    │
 *   │  └────────┬────────┘    └────────┬────────┘                    │
 *   │           │                      │                             │
 *   │           └──────────┬───────────┘                             │
 *   │                      │                                         │
 *   │              ┌───────▼───────┐                                 │
 *   │              │ MemorySegment │  ← Same underlying memory       │
 *   │              │   (pinned)    │                                 │
 *   │              └───────────────┘                                 │
 *   └─────────────────────────────────────────────────────────────────┘
 * </pre>
 *
 * <h2>Usage Example</h2>
 * <pre>{@code
 * try (RdmaApi rdma = Rdma.load();
 *      RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, spec)) {
 *
 *     // Use as tensor for computation
 *     Tensor tensor = buffer.tensor();
 *     // ... fill tensor with gradients ...
 *
 *     // Use for RDMA transfer (zero-copy)
 *     endpoint.send(buffer.rdmaBuffer()).join();
 * }
 * }</pre>
 *
 * <h2>GPU Backend Integration</h2>
 * <p>The underlying MemorySegment can be passed directly to WarpForge GPU
 * backends that support FFM MemorySegment. This enables a zero-copy path
 * from GPU memory through RDMA to remote nodes.
 *
 * @see RdmaBuffer
 * @see Tensor
 */
public final class RegisteredBuffer implements AutoCloseable {

    private final Tensor tensor;
    private final RdmaBuffer rdmaBuffer;
    private final boolean ownsTensor;
    private volatile boolean closed = false;

    private RegisteredBuffer(Tensor tensor, RdmaBuffer rdmaBuffer, boolean ownsTensor) {
        this.tensor = tensor;
        this.rdmaBuffer = rdmaBuffer;
        this.ownsTensor = ownsTensor;
    }

    /**
     * Allocates a new RDMA-registered buffer with the specified tensor specification.
     *
     * <p>The buffer is initialized to zeros. The returned RegisteredBuffer owns
     * both the tensor and the RDMA registration.
     *
     * @param rdma RDMA API instance
     * @param spec tensor specification (dtype and shape)
     * @return newly allocated registered buffer
     */
    public static RegisteredBuffer allocate(RdmaApi rdma, TensorSpec spec) {
        Tensor tensor = Tensor.zeros(spec.dtype(), spec.shape());
        RdmaBuffer rdmaBuffer = rdma.registerMemory(tensor.data());
        return new RegisteredBuffer(tensor, rdmaBuffer, true);
    }

    /**
     * Allocates a new RDMA-registered buffer with specified dtype and shape.
     *
     * @param rdma RDMA API instance
     * @param dtype data type
     * @param shape tensor shape
     * @return newly allocated registered buffer
     */
    public static RegisteredBuffer allocate(RdmaApi rdma, io.surfworks.warpforge.core.tensor.ScalarType dtype, int... shape) {
        Tensor tensor = Tensor.zeros(dtype, shape);
        RdmaBuffer rdmaBuffer = rdma.registerMemory(tensor.data());
        return new RegisteredBuffer(tensor, rdmaBuffer, true);
    }

    /**
     * Wraps an existing tensor, registering its memory for RDMA.
     *
     * <p>The caller retains ownership of the tensor. When this RegisteredBuffer
     * is closed, only the RDMA registration is released; the tensor remains valid.
     *
     * @param rdma RDMA API instance
     * @param tensor tensor to wrap
     * @return registered buffer wrapping the tensor
     */
    public static RegisteredBuffer wrap(RdmaApi rdma, Tensor tensor) {
        RdmaBuffer rdmaBuffer = rdma.registerMemory(tensor.data());
        return new RegisteredBuffer(tensor, rdmaBuffer, false);
    }

    /**
     * Creates a registered buffer from raw memory.
     *
     * <p>This is useful when integrating with GPU backends that provide
     * their own memory allocation.
     *
     * @param rdma RDMA API instance
     * @param segment memory segment to register
     * @param dtype data type for tensor view
     * @param shape shape for tensor view
     * @return registered buffer wrapping the segment
     */
    public static RegisteredBuffer fromSegment(RdmaApi rdma, MemorySegment segment,
                                                io.surfworks.warpforge.core.tensor.ScalarType dtype, int... shape) {
        TensorSpec spec = TensorSpec.of(dtype, shape);
        Tensor tensor = Tensor.fromMemorySegment(segment, spec);
        RdmaBuffer rdmaBuffer = rdma.registerMemory(segment);
        return new RegisteredBuffer(tensor, rdmaBuffer, false);
    }

    /**
     * Returns the tensor view of this buffer.
     *
     * <p>The tensor can be used for computation. Modifications are visible
     * in the underlying memory and will be transferred via RDMA operations.
     *
     * @return tensor backed by this buffer's memory
     */
    public Tensor tensor() {
        checkNotClosed();
        return tensor;
    }

    /**
     * Returns the RDMA buffer for network operations.
     *
     * @return RDMA-registered buffer
     */
    public RdmaBuffer rdmaBuffer() {
        checkNotClosed();
        return rdmaBuffer;
    }

    /**
     * Returns the underlying memory segment.
     *
     * <p>This segment can be passed directly to WarpForge GPU backends.
     *
     * @return underlying memory segment
     */
    public MemorySegment segment() {
        checkNotClosed();
        return tensor.data();
    }

    /**
     * Returns the remote key for RDMA one-sided operations.
     *
     * @return remote memory key
     */
    public long remoteKey() {
        checkNotClosed();
        return rdmaBuffer.remoteKey();
    }

    /**
     * Returns the virtual address of this buffer.
     *
     * @return buffer address for RDMA operations
     */
    public long address() {
        checkNotClosed();
        return rdmaBuffer.address();
    }

    /**
     * Returns the size of this buffer in bytes.
     *
     * @return buffer size
     */
    public long byteSize() {
        return tensor.spec().byteSize();
    }

    /**
     * Returns whether this buffer is still valid.
     *
     * @return true if buffer can be used
     */
    public boolean isValid() {
        return !closed && rdmaBuffer.isValid();
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // Always close RDMA registration
        rdmaBuffer.close();

        // Only close tensor if we own it
        if (ownsTensor) {
            tensor.close();
        }
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("RegisteredBuffer has been closed");
        }
    }

    @Override
    public String toString() {
        return String.format("RegisteredBuffer[%s, %d bytes, addr=0x%x, rkey=0x%x]",
                tensor.spec(), byteSize(), address(), remoteKey());
    }
}
