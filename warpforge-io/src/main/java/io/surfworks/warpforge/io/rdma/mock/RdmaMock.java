package io.surfworks.warpforge.io.rdma.mock;

import io.surfworks.warpforge.io.rdma.*;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Mock RDMA implementation for testing and development.
 *
 * <p>This implementation simulates RDMA operations without actual hardware.
 * It's used on systems without RDMA capability (e.g., macOS, Linux without
 * InfiniBand) and for unit testing.
 *
 * <h2>Behavior</h2>
 * <ul>
 *   <li>Memory registration: Creates mock buffers wrapping the segment</li>
 *   <li>Connections: Uses loopback or simulated network</li>
 *   <li>Operations: Copy data locally instead of RDMA transfer</li>
 *   <li>Statistics: Accurately tracks operation counts and bytes</li>
 * </ul>
 */
public class RdmaMock implements RdmaApi {

    private final RdmaConfig config;
    private final Map<Long, MockRdmaBuffer> registeredBuffers = new ConcurrentHashMap<>();
    private final AtomicLong bufferIdGenerator = new AtomicLong(1);
    private final AtomicLong totalBytesSent = new AtomicLong();
    private final AtomicLong totalBytesReceived = new AtomicLong();
    private final AtomicLong totalOperations = new AtomicLong();
    private final AtomicLong activeEndpoints = new AtomicLong();

    private volatile boolean closed = false;

    public RdmaMock(RdmaConfig config) {
        this.config = config;
    }

    @Override
    public String backendName() {
        return "mock";
    }

    @Override
    public RdmaConfig config() {
        return config;
    }

    @Override
    public List<RdmaDevice> devices() {
        // Return a simulated device
        return List.of(new RdmaDevice(
                "mock0",
                "WarpForge",
                1,
                4096,
                100.0,
                true,
                false
        ));
    }

    @Override
    public RdmaDevice currentDevice() {
        return devices().get(0);
    }

    @Override
    public void selectDevice(String deviceName) {
        if (!"mock0".equals(deviceName)) {
            throw RdmaException.deviceNotFound(deviceName);
        }
    }

    @Override
    public RdmaBuffer registerMemory(MemorySegment segment) {
        return registerMemory(segment, MemoryFlags.DEFAULT);
    }

    @Override
    public RdmaBuffer registerMemory(MemorySegment segment, int flags) {
        checkNotClosed();

        long id = bufferIdGenerator.getAndIncrement();
        MockRdmaBuffer buffer = new MockRdmaBuffer(id, segment, flags, this);
        registeredBuffers.put(id, buffer);
        return buffer;
    }

    @Override
    public void unregisterMemory(RdmaBuffer buffer) {
        if (buffer instanceof MockRdmaBuffer mockBuffer) {
            registeredBuffers.remove(mockBuffer.id());
            mockBuffer.invalidate();
        }
    }

    @Override
    public RdmaEndpoint connect(String remoteAddress, int port) {
        return connect(remoteAddress, port, 30000);
    }

    @Override
    public RdmaEndpoint connect(String remoteAddress, int port, long timeoutMillis) {
        checkNotClosed();
        activeEndpoints.incrementAndGet();
        return new MockRdmaEndpoint(remoteAddress, port, this);
    }

    @Override
    public RdmaListener listen(int port) {
        checkNotClosed();
        return new MockRdmaListener(port, this);
    }

    @Override
    public boolean supportsGpuDirect() {
        return false;
    }

    @Override
    public boolean supportsAtomics() {
        return true;
    }

    @Override
    public int maxInlineSize() {
        return config.maxInlineData();
    }

    @Override
    public long maxMemoryRegionSize() {
        return Long.MAX_VALUE;
    }

    @Override
    public RdmaStats stats() {
        return new RdmaStats(
                totalBytesSent.get(),
                totalBytesReceived.get(),
                totalOperations.get(),
                activeEndpoints.get(),
                registeredBuffers.values().stream().mapToLong(b -> b.byteSize()).sum(),
                registeredBuffers.size()
        );
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // Invalidate all registered buffers
        registeredBuffers.values().forEach(MockRdmaBuffer::invalidate);
        registeredBuffers.clear();
    }

    private void checkNotClosed() {
        if (closed) {
            throw new RdmaException("RDMA context has been closed", RdmaException.ErrorCode.INVALID_STATE);
        }
    }

    // Internal methods for mock components
    void recordSend(long bytes) {
        totalBytesSent.addAndGet(bytes);
        totalOperations.incrementAndGet();
    }

    void recordReceive(long bytes) {
        totalBytesReceived.addAndGet(bytes);
        totalOperations.incrementAndGet();
    }

    void recordOperation() {
        totalOperations.incrementAndGet();
    }

    void endpointClosed() {
        activeEndpoints.decrementAndGet();
    }

    MockRdmaBuffer getBuffer(long id) {
        return registeredBuffers.get(id);
    }
}
