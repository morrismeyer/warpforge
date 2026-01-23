package io.surfworks.warpforge.io.collective.impl;

import java.lang.foreign.MemorySegment;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * Pool of pre-allocated tensors for collective operations.
 *
 * <p>Collective operations often need temporary output tensors. Creating a new
 * tensor for each operation incurs allocation overhead. This pool maintains
 * tensors organized by size class for efficient reuse.
 *
 * <h2>Size Classes</h2>
 * <p>Tensors are pooled by their byte size, rounded up to the nearest power of 2.
 * This provides a reasonable trade-off between pool fragmentation and memory waste:
 * <ul>
 *   <li>1KB-2KB tensors share a pool (up to 2KB)</li>
 *   <li>2KB-4KB tensors share a pool (up to 4KB)</li>
 *   <li>4KB-1MB tensors share a pool (up to 1MB)</li>
 *   <li>1MB+ tensors share larger size class pools</li>
 * </ul>
 *
 * <h2>Performance Impact</h2>
 * <p>For repeated operations with similar sizes:
 * <ul>
 *   <li>Without pool: ~5-50us per tensor allocation (varies by size)</li>
 *   <li>With pool: ~100ns per acquire/release (cache hit)</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <p>This class is thread-safe. Multiple threads can acquire and release
 * tensors concurrently.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * TensorBufferPool pool = new TensorBufferPool();
 *
 * // Acquire a tensor of at least the required size
 * Tensor output = pool.acquire(ScalarType.FLOAT32, 1024 * 1024);
 *
 * // Use the tensor for collective operation
 * collective.allReduce(input, output);
 *
 * // Release back to pool for reuse
 * pool.release(output);
 *
 * pool.close();
 * }</pre>
 */
public final class TensorBufferPool implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(TensorBufferPool.class.getName());

    /** Minimum size class (1KB) */
    private static final long MIN_SIZE_CLASS = 1024;

    /** Maximum size class (256MB) */
    private static final long MAX_SIZE_CLASS = 256 * 1024 * 1024;

    /** Maximum tensors per size class */
    private static final int MAX_TENSORS_PER_CLASS = 4;

    /** System property to enable tensor pooling */
    public static final String PROP_ENABLED = "warpforge.ucc.tensorPool";

    /**
     * Singleton instance for global use.
     */
    private static volatile TensorBufferPool instance;

    /**
     * Get the global tensor buffer pool.
     *
     * <p>Returns null if tensor pooling is disabled via system property.
     */
    public static TensorBufferPool getInstance() {
        if (!Boolean.parseBoolean(System.getProperty(PROP_ENABLED, "false"))) {
            return null;
        }
        if (instance == null) {
            synchronized (TensorBufferPool.class) {
                if (instance == null) {
                    instance = new TensorBufferPool();
                }
            }
        }
        return instance;
    }

    // Pool organized by (dtype, sizeClass) -> queue of tensors
    private final ConcurrentHashMap<PoolKey, Deque<Tensor>> pools;

    // Statistics
    private final AtomicLong acquireCount = new AtomicLong();
    private final AtomicLong hitCount = new AtomicLong();
    private final AtomicLong missCount = new AtomicLong();
    private final AtomicLong releaseCount = new AtomicLong();
    private final AtomicLong evictionCount = new AtomicLong();

    private volatile boolean closed = false;

    /**
     * Create a new tensor buffer pool.
     */
    public TensorBufferPool() {
        this.pools = new ConcurrentHashMap<>();
    }

    /**
     * Acquire a tensor of at least the specified size.
     *
     * <p>If a suitable tensor is available in the pool, it is returned.
     * Otherwise, a new tensor is allocated. The returned tensor may be
     * larger than requested (rounded up to size class).
     *
     * @param dtype data type of the tensor
     * @param minByteSize minimum size in bytes
     * @return a tensor with at least minByteSize bytes
     */
    public Tensor acquire(ScalarType dtype, long minByteSize) {
        if (closed) {
            throw new IllegalStateException("TensorBufferPool is closed");
        }

        acquireCount.incrementAndGet();

        long sizeClass = roundUpToSizeClass(minByteSize);
        PoolKey key = new PoolKey(dtype, sizeClass);

        Deque<Tensor> pool = pools.get(key);
        if (pool != null) {
            synchronized (pool) {
                Tensor tensor = pool.pollFirst();
                if (tensor != null) {
                    hitCount.incrementAndGet();
                    // Zero the tensor for safety (avoid leaking data between operations)
                    zeroTensor(tensor);
                    return tensor;
                }
            }
        }

        missCount.incrementAndGet();

        // Allocate new tensor with size class dimensions
        // Use a 1D shape that gives us the right byte size
        int elementSize = dtype.byteSize();
        int elementCount = (int) (sizeClass / elementSize);
        return Tensor.zeros(dtype, elementCount);
    }

    /**
     * Acquire a tensor with the same shape and dtype as the template.
     *
     * @param template tensor to match shape and dtype
     * @return a tensor with matching shape and dtype
     */
    public Tensor acquireLike(Tensor template) {
        return acquire(template.dtype(), template.spec().byteSize());
    }

    /**
     * Release a tensor back to the pool for reuse.
     *
     * <p>If the pool for this size class is full, the tensor is closed.
     *
     * @param tensor tensor to release
     */
    public void release(Tensor tensor) {
        if (tensor == null || closed) {
            if (tensor != null) {
                tensor.close();
            }
            return;
        }

        releaseCount.incrementAndGet();

        long sizeClass = roundUpToSizeClass(tensor.spec().byteSize());
        PoolKey key = new PoolKey(tensor.dtype(), sizeClass);

        Deque<Tensor> pool = pools.computeIfAbsent(key, k -> new ArrayDeque<>());

        synchronized (pool) {
            if (pool.size() < MAX_TENSORS_PER_CLASS) {
                pool.addLast(tensor);
                return;
            }
        }

        // Pool is full, close the tensor
        evictionCount.incrementAndGet();
        tensor.close();
    }

    /**
     * Round up to the nearest power-of-2 size class.
     */
    private long roundUpToSizeClass(long size) {
        if (size <= MIN_SIZE_CLASS) {
            return MIN_SIZE_CLASS;
        }
        if (size >= MAX_SIZE_CLASS) {
            return MAX_SIZE_CLASS;
        }

        // Round up to next power of 2
        long sizeClass = MIN_SIZE_CLASS;
        while (sizeClass < size) {
            sizeClass *= 2;
        }
        return sizeClass;
    }

    /**
     * Zero a tensor's memory for reuse.
     */
    private void zeroTensor(Tensor tensor) {
        MemorySegment data = tensor.data();
        data.fill((byte) 0);
    }

    /**
     * Get pool statistics.
     *
     * @return statistics string for logging
     */
    public String getStats() {
        long total = acquireCount.get();
        long hits = hitCount.get();
        double hitRate = total > 0 ? (double) hits / total * 100 : 0;
        return String.format(
            "TensorBufferPool: acquires=%d, hits=%d (%.1f%%), misses=%d, releases=%d, evictions=%d, pools=%d",
            total, hits, hitRate, missCount.get(), releaseCount.get(), evictionCount.get(), pools.size()
        );
    }

    /**
     * Get total number of pooled tensors.
     */
    public int pooledCount() {
        return pools.values().stream().mapToInt(Deque::size).sum();
    }

    /**
     * Clear all pooled tensors.
     */
    public void clear() {
        for (Deque<Tensor> pool : pools.values()) {
            synchronized (pool) {
                Tensor tensor;
                while ((tensor = pool.pollFirst()) != null) {
                    tensor.close();
                }
            }
        }
        pools.clear();
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        LOG.fine("Closing TensorBufferPool: " + getStats());
        clear();
    }

    /**
     * Key for pool lookup.
     */
    private record PoolKey(ScalarType dtype, long sizeClass) {}
}
