package io.surfworks.warpforge.io.collective.impl;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * Page-aligned memory allocation for optimal RDMA performance.
 *
 * <p>RDMA hardware works most efficiently with page-aligned memory (4KB alignment
 * on most systems, 64KB on some HPC systems). This allocator ensures all memory
 * is properly aligned and tracks allocations for cleanup.
 *
 * <h2>Performance Impact</h2>
 * <p>Page-aligned memory provides:
 * <ul>
 *   <li>Efficient DMA transfers (no partial page handling)</li>
 *   <li>Better memory registration performance</li>
 *   <li>Reduced cache pollution from alignment padding</li>
 * </ul>
 *
 * <h2>Alignment Levels</h2>
 * <table>
 *   <tr><th>Alignment</th><th>Use Case</th></tr>
 *   <tr><td>64 bytes</td><td>Cache line (CPU optimization)</td></tr>
 *   <tr><td>4096 bytes</td><td>Standard page (RDMA standard)</td></tr>
 *   <tr><td>2MB</td><td>Huge page (HPC systems)</td></tr>
 * </table>
 *
 * <h2>Thread Safety</h2>
 * <p>This class is thread-safe. Multiple threads can allocate and free
 * memory concurrently.
 */
public final class AlignedMemoryAllocator implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(AlignedMemoryAllocator.class.getName());

    /** Cache line alignment (64 bytes) */
    public static final long CACHE_LINE_ALIGNMENT = 64;

    /** Standard page alignment (4KB) */
    public static final long PAGE_ALIGNMENT = 4096;

    /** Large page alignment (2MB) */
    public static final long HUGE_PAGE_ALIGNMENT = 2 * 1024 * 1024;

    /** Default alignment for RDMA operations */
    public static final long DEFAULT_ALIGNMENT = PAGE_ALIGNMENT;

    /**
     * Singleton instance for global use.
     */
    private static volatile AlignedMemoryAllocator instance;

    /**
     * Get the global allocator instance.
     */
    public static AlignedMemoryAllocator getInstance() {
        if (instance == null) {
            synchronized (AlignedMemoryAllocator.class) {
                if (instance == null) {
                    instance = new AlignedMemoryAllocator();
                }
            }
        }
        return instance;
    }

    private final Arena arena;
    private final ConcurrentHashMap<Long, MemorySegment> allocations;
    private final AtomicLong totalAllocated = new AtomicLong();
    private final AtomicLong allocationCount = new AtomicLong();

    private volatile boolean closed = false;

    /**
     * Create a new aligned memory allocator.
     */
    public AlignedMemoryAllocator() {
        this.arena = Arena.ofShared();
        this.allocations = new ConcurrentHashMap<>();
    }

    /**
     * Allocate page-aligned memory.
     *
     * @param size size in bytes (will be rounded up to page boundary)
     * @return page-aligned memory segment
     */
    public MemorySegment allocate(long size) {
        return allocate(size, DEFAULT_ALIGNMENT);
    }

    /**
     * Allocate memory with specified alignment.
     *
     * @param size size in bytes
     * @param alignment alignment in bytes (must be power of 2)
     * @return aligned memory segment
     */
    public MemorySegment allocate(long size, long alignment) {
        if (closed) {
            throw new IllegalStateException("Allocator is closed");
        }
        if (size <= 0) {
            throw new IllegalArgumentException("Size must be positive: " + size);
        }
        if (!isPowerOfTwo(alignment)) {
            throw new IllegalArgumentException("Alignment must be power of 2: " + alignment);
        }

        // Round size up to alignment boundary
        long alignedSize = roundUp(size, alignment);

        MemorySegment segment = arena.allocate(alignedSize, alignment);

        allocations.put(segment.address(), segment);
        totalAllocated.addAndGet(alignedSize);
        allocationCount.incrementAndGet();

        return segment;
    }

    /**
     * Allocate and zero-fill memory.
     *
     * @param size size in bytes
     * @return zero-filled, page-aligned memory segment
     */
    public MemorySegment allocateZeroed(long size) {
        return allocateZeroed(size, DEFAULT_ALIGNMENT);
    }

    /**
     * Allocate and zero-fill memory with specified alignment.
     *
     * @param size size in bytes
     * @param alignment alignment in bytes
     * @return zero-filled, aligned memory segment
     */
    public MemorySegment allocateZeroed(long size, long alignment) {
        MemorySegment segment = allocate(size, alignment);
        segment.fill((byte) 0);
        return segment;
    }

    /**
     * Free a previously allocated segment.
     *
     * <p>Note: With shared arena, memory is not actually freed until
     * the allocator is closed. This method just removes tracking.
     *
     * @param segment segment to free
     */
    public void free(MemorySegment segment) {
        if (segment != null) {
            allocations.remove(segment.address());
        }
    }

    /**
     * Check if a memory segment is properly aligned.
     *
     * @param segment segment to check
     * @param alignment required alignment
     * @return true if segment address is aligned
     */
    public static boolean isAligned(MemorySegment segment, long alignment) {
        return segment != null && (segment.address() % alignment) == 0;
    }

    /**
     * Check if a memory segment is page-aligned.
     *
     * @param segment segment to check
     * @return true if segment is page-aligned
     */
    public static boolean isPageAligned(MemorySegment segment) {
        return isAligned(segment, PAGE_ALIGNMENT);
    }

    /**
     * Round up to alignment boundary.
     */
    private static long roundUp(long value, long alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    /**
     * Check if value is power of 2.
     */
    private static boolean isPowerOfTwo(long value) {
        return value > 0 && (value & (value - 1)) == 0;
    }

    /**
     * Get statistics about allocations.
     */
    public String getStats() {
        return String.format("AlignedMemoryAllocator: %d allocations, %d bytes total, %d tracked",
            allocationCount.get(), totalAllocated.get(), allocations.size());
    }

    /**
     * Get total bytes allocated.
     */
    public long getTotalAllocated() {
        return totalAllocated.get();
    }

    /**
     * Get number of active allocations.
     */
    public int getActiveCount() {
        return allocations.size();
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        LOG.fine("Closing AlignedMemoryAllocator: " + getStats());

        allocations.clear();

        try {
            arena.close();
        } catch (Exception e) {
            LOG.warning("Error closing arena: " + e.getMessage());
        }
    }
}
