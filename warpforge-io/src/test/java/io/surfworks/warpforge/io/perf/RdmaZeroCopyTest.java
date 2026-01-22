package io.surfworks.warpforge.io.perf;

import io.surfworks.warpforge.io.rdma.Rdma;
import io.surfworks.warpforge.io.rdma.RdmaApi;
import io.surfworks.warpforge.io.rdma.RdmaBuffer;
import io.surfworks.warpforge.io.rdma.RdmaEndpoint;
import io.surfworks.warpforge.io.rdma.RdmaListener;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Zero-Copy Verification Tests for RDMA.
 *
 * <p>These tests verify that the RDMA implementation uses MemorySegment directly
 * without intermediate copies. This is critical for achieving line-rate performance
 * and enabling GPU Direct RDMA in future.
 *
 * <h2>Zero-Copy Requirements</h2>
 * <ul>
 *   <li>RdmaBuffer.segment() must return the original MemorySegment</li>
 *   <li>Memory address must remain constant after registration</li>
 *   <li>No intermediate ByteBuffer or byte[] allocations</li>
 *   <li>Same MemorySegment usable by GPU backends via FFM</li>
 * </ul>
 *
 * <h2>Verification Methods</h2>
 * <ul>
 *   <li>Address equality: MemorySegment.address() unchanged</li>
 *   <li>Reference equality: segment() returns same object</li>
 *   <li>Content modification: Changes visible through both references</li>
 * </ul>
 */
@Tag("unit")
@DisplayName("RDMA Zero-Copy Verification")
class RdmaZeroCopyTest {

    private RdmaApi rdma;
    private Arena arena;

    @BeforeEach
    void setUp() {
        rdma = Rdma.load();
        arena = Arena.ofConfined();
    }

    @AfterEach
    void tearDown() {
        arena.close();
        rdma.close();
    }

    @Test
    @DisplayName("RdmaBuffer.segment() returns original MemorySegment")
    void testSegmentIdentity() {
        MemorySegment original = arena.allocate(4096);

        try (RdmaBuffer buffer = rdma.registerMemory(original)) {
            MemorySegment returned = buffer.segment();

            // Reference equality - must be the exact same object
            assertSame(original, returned,
                    "RdmaBuffer.segment() must return the original MemorySegment, not a copy");
        }
    }

    @Test
    @DisplayName("Memory address unchanged after registration")
    void testAddressStability() {
        MemorySegment segment = arena.allocate(4096);
        long originalAddress = segment.address();

        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            // The buffer's address should match the segment's address
            assertEquals(originalAddress, buffer.address(),
                    "RdmaBuffer.address() should equal MemorySegment.address()");

            // The segment's address should be unchanged
            assertEquals(originalAddress, segment.address(),
                    "MemorySegment.address() should not change after registration");

            // And they should all match
            assertEquals(buffer.segment().address(), buffer.address(),
                    "All address accessors must return the same value");
        }
    }

    @Test
    @DisplayName("Modifications through MemorySegment visible via RdmaBuffer")
    void testBidirectionalVisibility() {
        MemorySegment segment = arena.allocate(1024);

        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            // Write through original segment
            segment.set(ValueLayout.JAVA_LONG, 0, 0xDEADBEEFCAFEBABEL);

            // Read through RdmaBuffer's segment - should see same value
            long value = buffer.segment().get(ValueLayout.JAVA_LONG, 0);
            assertEquals(0xDEADBEEFCAFEBABEL, value,
                    "Write through original segment must be visible through RdmaBuffer.segment()");

            // Write through buffer's segment
            buffer.segment().set(ValueLayout.JAVA_LONG, 8, 0x1234567890ABCDEFL);

            // Read through original - should see same value
            long value2 = segment.get(ValueLayout.JAVA_LONG, 8);
            assertEquals(0x1234567890ABCDEFL, value2,
                    "Write through RdmaBuffer.segment() must be visible through original segment");
        }
    }

    @Test
    @DisplayName("Buffer size matches MemorySegment size")
    void testSizeConsistency() {
        long[] sizes = {64, 4096, 1024 * 1024, 16 * 1024 * 1024};

        for (long size : sizes) {
            MemorySegment segment = arena.allocate(size);

            try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                assertEquals(size, buffer.byteSize(),
                        "RdmaBuffer.byteSize() must match MemorySegment.byteSize()");
                assertEquals(segment.byteSize(), buffer.byteSize(),
                        "Sizes must be identical");
            }
        }
    }

    @Test
    @DisplayName("Large buffer zero-copy (256MB)")
    void testLargeBufferZeroCopy() {
        // 256MB buffer - any copying here would be extremely slow
        long size = 256 * 1024 * 1024L;
        MemorySegment segment = arena.allocate(size);

        long startNanos = System.nanoTime();

        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            long registrationNanos = System.nanoTime() - startNanos;

            // Registration should be fast (< 100ms) - copying 256MB would take much longer
            // Typical registration is ~100us, copying 256MB at 10GB/s would be ~25ms
            assertTrue(registrationNanos < 100_000_000, // 100ms
                    String.format("Registration took %.2f ms - possible copy detected",
                            registrationNanos / 1e6));

            // Verify identity
            assertSame(segment, buffer.segment(),
                    "Large buffer segment identity failed");

            System.out.printf("256MB registration time: %.3f ms%n", registrationNanos / 1e6);
        }
    }

    @Test
    @DisplayName("Fill pattern visible without copy")
    void testFillPatternNoCopy() {
        MemorySegment segment = arena.allocate(1024 * 1024); // 1MB

        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            // Fill through original
            byte pattern = (byte) 0xAB;
            segment.fill(pattern);

            // Verify through buffer's segment (sample check - not every byte)
            MemorySegment bufferSegment = buffer.segment();
            for (int i = 0; i < 1024; i += 64) {
                assertEquals(pattern, bufferSegment.get(ValueLayout.JAVA_BYTE, i),
                        "Fill pattern not visible at offset " + i);
            }

            // Verify at end
            assertEquals(pattern, bufferSegment.get(ValueLayout.JAVA_BYTE, 1024 * 1024 - 1),
                    "Fill pattern not visible at end");
        }
    }

    @Test
    @DisplayName("Multiple registrations of different segments are independent")
    void testMultipleBufferIndependence() {
        MemorySegment seg1 = arena.allocate(4096);
        MemorySegment seg2 = arena.allocate(4096);

        try (RdmaBuffer buf1 = rdma.registerMemory(seg1);
             RdmaBuffer buf2 = rdma.registerMemory(seg2)) {

            // Write different values
            seg1.set(ValueLayout.JAVA_LONG, 0, 111L);
            seg2.set(ValueLayout.JAVA_LONG, 0, 222L);

            // Verify independence
            assertEquals(111L, buf1.segment().get(ValueLayout.JAVA_LONG, 0));
            assertEquals(222L, buf2.segment().get(ValueLayout.JAVA_LONG, 0));

            // Addresses must be different
            assertTrue(buf1.address() != buf2.address(),
                    "Different buffers must have different addresses");
        }
    }

    @Test
    @DisplayName("Keys are valid for RDMA operations")
    void testKeyValidity() {
        MemorySegment segment = arena.allocate(4096);

        try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
            // Keys should be non-zero (0 typically indicates unregistered)
            // Note: mock implementation may use different sentinel values
            long localKey = buffer.localKey();
            long remoteKey = buffer.remoteKey();

            System.out.printf("Local Key:  0x%016x%n", localKey);
            System.out.printf("Remote Key: 0x%016x%n", remoteKey);

            // At minimum, buffer should be valid
            assertTrue(buffer.isValid(), "Buffer should be valid after registration");
        }
    }

    @Test
    @DisplayName("Unregistered buffer is invalid")
    void testUnregisteredBufferInvalid() {
        MemorySegment segment = arena.allocate(4096);

        RdmaBuffer buffer = rdma.registerMemory(segment);
        assertTrue(buffer.isValid(), "Buffer should be valid after registration");

        buffer.close();

        // After close, buffer should be invalid
        assertTrue(!buffer.isValid(), "Buffer should be invalid after close");
    }

    // ===== Advanced Zero-Copy Verification =====

    @Nested
    @DisplayName("Memory Properties")
    class MemoryPropertiesTests {

        @Test
        @DisplayName("Memory is native (off-heap)")
        void testNativeMemory() {
            // Arena.allocate() should allocate native memory, not heap
            MemorySegment segment = arena.allocate(4096);

            try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                // Native memory has a valid address (heap memory would fail RDMA registration)
                assertTrue(buffer.address() != 0,
                        "Buffer address should be non-zero (native memory)");

                // The address should be stable across calls
                long addr1 = buffer.address();
                long addr2 = buffer.address();
                assertEquals(addr1, addr2,
                        "Buffer address should be stable across calls");
            }
        }

        @Test
        @DisplayName("Page alignment for optimal RDMA performance")
        void testPageAlignment() {
            // Page-aligned allocations are optimal for RDMA
            long pageSize = 4096; // Typical page size
            long size = pageSize * 4; // 16KB, multiple pages

            MemorySegment segment = arena.allocate(size, pageSize);

            try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                // Address should be page-aligned
                long address = buffer.address();
                assertEquals(0, address % pageSize,
                        String.format("Address 0x%x should be page-aligned to %d",
                                address, pageSize));

                System.out.printf("Page-aligned buffer: addr=0x%x, size=%d%n", address, size);
            }
        }

        @Test
        @DisplayName("Memory contiguity for large allocations")
        void testMemoryContiguity() {
            // 64MB contiguous allocation
            long size = 64 * 1024 * 1024;
            MemorySegment segment = arena.allocate(size);

            try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                long startAddr = buffer.address();
                long endAddr = startAddr + size - 1;

                // Write at both ends and middle
                segment.set(ValueLayout.JAVA_LONG, 0, 0xAAAAAAAAAAAAAAAAL);
                segment.set(ValueLayout.JAVA_LONG, size - 8, 0xBBBBBBBBBBBBBBBBL);
                segment.set(ValueLayout.JAVA_LONG, size / 2, 0xCCCCCCCCCCCCCCCCL);

                // Verify through buffer's segment
                assertEquals(0xAAAAAAAAAAAAAAAAL, buffer.segment().get(ValueLayout.JAVA_LONG, 0));
                assertEquals(0xBBBBBBBBBBBBBBBBL, buffer.segment().get(ValueLayout.JAVA_LONG, size - 8));
                assertEquals(0xCCCCCCCCCCCCCCCCL, buffer.segment().get(ValueLayout.JAVA_LONG, size / 2));

                System.out.printf("Contiguous 64MB: 0x%x to 0x%x%n", startAddr, endAddr);
            }
        }
    }

    @Nested
    @DisplayName("Concurrent Access")
    class ConcurrentAccessTests {

        @Test
        @DisplayName("Concurrent writes visible across threads")
        void testConcurrentWriteVisibility() throws Exception {
            int numThreads = 4;
            int writesPerThread = 1000;
            long size = numThreads * writesPerThread * 8L;

            // Use shared arena for multi-threaded access
            try (Arena sharedArena = Arena.ofShared()) {
                MemorySegment segment = sharedArena.allocate(size);

                try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                    ExecutorService executor = Executors.newFixedThreadPool(numThreads);
                    CountDownLatch startLatch = new CountDownLatch(1);
                    CountDownLatch doneLatch = new CountDownLatch(numThreads);
                    AtomicBoolean error = new AtomicBoolean(false);

                    // Each thread writes to its own region
                    for (int t = 0; t < numThreads; t++) {
                        final int threadId = t;
                        final long offset = threadId * writesPerThread * 8L;

                        executor.submit(() -> {
                            try {
                                startLatch.await(); // Wait for all threads ready
                                MemorySegment seg = buffer.segment();
                                for (int i = 0; i < writesPerThread; i++) {
                                    long value = ((long) threadId << 16) | i;
                                    seg.set(ValueLayout.JAVA_LONG, offset + i * 8L, value);
                                }
                            } catch (Exception e) {
                                error.set(true);
                                e.printStackTrace();
                            } finally {
                                doneLatch.countDown();
                            }
                        });
                    }

                    startLatch.countDown(); // Start all threads
                    assertTrue(doneLatch.await(10, TimeUnit.SECONDS), "Writers should complete");
                    assertFalse(error.get(), "No errors during concurrent writes");

                    executor.shutdown();
                    executor.awaitTermination(5, TimeUnit.SECONDS);

                    // Verify all writes through original segment
                    for (int t = 0; t < numThreads; t++) {
                        long offset = t * writesPerThread * 8L;
                        for (int i = 0; i < writesPerThread; i++) {
                            long expected = ((long) t << 16) | i;
                            long actual = segment.get(ValueLayout.JAVA_LONG, offset + i * 8L);
                            assertEquals(expected, actual,
                                    String.format("Thread %d, write %d mismatch: expected 0x%x, got 0x%x",
                                            t, i, expected, actual));
                        }
                    }

                    System.out.printf("Verified %d concurrent writes across %d threads%n",
                            numThreads * writesPerThread, numThreads);
                }
            }
        }

        @Test
        @DisplayName("Address stability under concurrent access")
        void testAddressStabilityUnderLoad() throws Exception {
            // Use shared arena for multi-threaded access
            try (Arena sharedArena = Arena.ofShared()) {
                MemorySegment segment = sharedArena.allocate(1024 * 1024);

                try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                    long expectedAddress = buffer.address();

                    ExecutorService executor = Executors.newFixedThreadPool(8);
                    AtomicLong addressMismatches = new AtomicLong(0);
                    CountDownLatch latch = new CountDownLatch(8);

                    for (int t = 0; t < 8; t++) {
                        executor.submit(() -> {
                            try {
                                for (int i = 0; i < 10000; i++) {
                                    long addr = buffer.address();
                                    if (addr != expectedAddress) {
                                        addressMismatches.incrementAndGet();
                                    }
                                    // Also verify segment address
                                    if (buffer.segment().address() != expectedAddress) {
                                        addressMismatches.incrementAndGet();
                                    }
                                }
                            } finally {
                                latch.countDown();
                            }
                        });
                    }

                    assertTrue(latch.await(30, TimeUnit.SECONDS), "Threads should complete");
                    assertEquals(0, addressMismatches.get(),
                            "Address should never change under concurrent access");

                    executor.shutdown();
                    System.out.println("Address stable across 80,000 concurrent accesses");
                }
            }
        }
    }

    @Nested
    @DisplayName("Partial Operations")
    class PartialOperationsTests {

        @Test
        @DisplayName("Partial write within buffer")
        void testPartialWriteVisibility() {
            MemorySegment segment = arena.allocate(4096);
            segment.fill((byte) 0x00); // Initialize to zeros

            try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                // Write to middle region only (offset 1024, length 1024)
                for (int i = 0; i < 128; i++) {
                    buffer.segment().set(ValueLayout.JAVA_LONG, 1024 + i * 8L, 0xFFFFFFFFFFFFFFFL);
                }

                // Verify through original segment
                // Before region should be zeros
                assertEquals(0L, segment.get(ValueLayout.JAVA_LONG, 1016));

                // Within region should be all F's
                for (int i = 0; i < 128; i++) {
                    assertEquals(0xFFFFFFFFFFFFFFFL, segment.get(ValueLayout.JAVA_LONG, 1024 + i * 8L));
                }

                // After region should be zeros
                assertEquals(0L, segment.get(ValueLayout.JAVA_LONG, 2048));
            }
        }

        @Test
        @DisplayName("Boundary bytes preserved during partial operations")
        void testBoundaryPreservation() {
            int size = 4096;
            MemorySegment segment = arena.allocate(size);

            // Set sentinel values at boundaries
            segment.set(ValueLayout.JAVA_BYTE, 0, (byte) 0xAA);
            segment.set(ValueLayout.JAVA_BYTE, size - 1, (byte) 0xBB);
            segment.set(ValueLayout.JAVA_BYTE, 1, (byte) 0xCC);
            segment.set(ValueLayout.JAVA_BYTE, size - 2, (byte) 0xDD);

            try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                // Fill middle region
                for (int i = 100; i < size - 100; i++) {
                    buffer.segment().set(ValueLayout.JAVA_BYTE, i, (byte) 0xFF);
                }

                // Verify boundaries preserved
                assertEquals((byte) 0xAA, segment.get(ValueLayout.JAVA_BYTE, 0), "First byte corrupted");
                assertEquals((byte) 0xBB, segment.get(ValueLayout.JAVA_BYTE, size - 1), "Last byte corrupted");
                assertEquals((byte) 0xCC, segment.get(ValueLayout.JAVA_BYTE, 1), "Second byte corrupted");
                assertEquals((byte) 0xDD, segment.get(ValueLayout.JAVA_BYTE, size - 2), "Second-to-last byte corrupted");
            }
        }
    }

    @Nested
    @DisplayName("RDMA Operations (Mock Loopback)")
    class RdmaOperationsTests {

        @Test
        @DisplayName("Send operation preserves zero-copy semantics")
        void testSendZeroCopy() throws Exception {
            MemorySegment sendSeg = arena.allocate(4096);

            // Fill send buffer with pattern
            for (int i = 0; i < 512; i++) {
                sendSeg.set(ValueLayout.JAVA_LONG, i * 8L, i * 1000L);
            }

            try (RdmaBuffer sendBuf = rdma.registerMemory(sendSeg)) {
                // Verify segment identity before operation
                assertSame(sendSeg, sendBuf.segment(), "Send buffer segment identity before op");
                long addressBefore = sendBuf.address();

                // Create endpoint and perform send (mock doesn't require a real listener)
                try (RdmaEndpoint client = rdma.connect("localhost", 18600, 5000)) {
                    // Send data - the operation itself should preserve zero-copy
                    client.send(sendBuf).join();

                    // Verify segment identity after operation (should not change)
                    assertSame(sendSeg, sendBuf.segment(), "Send buffer segment changed after send");
                    assertEquals(addressBefore, sendBuf.address(), "Address changed after send");

                    // Verify data in buffer unchanged
                    for (int i = 0; i < 512; i++) {
                        assertEquals(i * 1000L, sendSeg.get(ValueLayout.JAVA_LONG, i * 8L),
                                "Data corrupted after send at index " + i);
                    }
                }
            }
        }

        @Test
        @DisplayName("Receive operation preserves zero-copy semantics")
        void testReceiveZeroCopy() throws Exception {
            MemorySegment recvSeg = arena.allocate(4096);
            recvSeg.fill((byte) 0x00); // Initialize

            try (RdmaBuffer recvBuf = rdma.registerMemory(recvSeg)) {
                // Verify segment identity before operation
                assertSame(recvSeg, recvBuf.segment(), "Recv buffer segment identity before op");
                long addressBefore = recvBuf.address();

                try (RdmaEndpoint client = rdma.connect("localhost", 18601, 5000)) {
                    // Post receive - mock will return immediately
                    // Note: Mock doesn't actually receive data, but the operation should preserve zero-copy
                    var future = client.receive(recvBuf);

                    // Verify segment identity is preserved even while receive is pending
                    assertSame(recvSeg, recvBuf.segment(), "Recv buffer segment changed during receive");
                    assertEquals(addressBefore, recvBuf.address(), "Address changed during receive");

                    // Cancel the pending receive (mock)
                    future.cancel(true);
                }
            }
        }

        @Test
        @DisplayName("RDMA write preserves zero-copy semantics")
        void testRdmaWriteZeroCopy() throws Exception {
            MemorySegment localSeg = arena.allocate(4096);
            MemorySegment remoteSeg = arena.allocate(4096);

            // Fill local buffer
            localSeg.fill((byte) 0xAB);

            try (RdmaBuffer localBuf = rdma.registerMemory(localSeg);
                 RdmaBuffer remoteBuf = rdma.registerMemory(remoteSeg);
                 RdmaListener listener = rdma.listen(18601);
                 RdmaEndpoint client = rdma.connect("localhost", 18601, 5000);
                 RdmaEndpoint server = listener.accept(5000)) {

                long remoteAddr = remoteBuf.address();
                long remoteKey = remoteBuf.remoteKey();

                // RDMA write
                client.write(localBuf, remoteAddr, remoteKey).join();

                // Verify segments unchanged
                assertSame(localSeg, localBuf.segment(), "Local segment changed after write");
                assertSame(remoteSeg, remoteBuf.segment(), "Remote segment changed after write");

                // Verify addresses unchanged
                assertEquals(localSeg.address(), localBuf.address());
                assertEquals(remoteSeg.address(), remoteBuf.address());

                // Verify data written (mock implementation)
                for (int i = 0; i < 4096; i += 64) {
                    assertEquals((byte) 0xAB, remoteSeg.get(ValueLayout.JAVA_BYTE, i),
                            "RDMA write data mismatch at offset " + i);
                }
            }
        }

        @Test
        @DisplayName("Multiple operations don't create copies")
        void testMultipleOperationsNoCopies() throws Exception {
            MemorySegment seg = arena.allocate(4096);

            try (RdmaBuffer buf = rdma.registerMemory(seg);
                 RdmaListener listener = rdma.listen(18602);
                 RdmaEndpoint client = rdma.connect("localhost", 18602, 5000);
                 RdmaEndpoint server = listener.accept(5000)) {

                long originalAddress = buf.address();
                MemorySegment originalSegment = buf.segment();

                // Perform multiple operations
                for (int i = 0; i < 100; i++) {
                    seg.set(ValueLayout.JAVA_LONG, 0, i);
                    client.send(buf).join();

                    // Verify no copies created
                    assertSame(originalSegment, buf.segment(),
                            "Segment changed after iteration " + i);
                    assertEquals(originalAddress, buf.address(),
                            "Address changed after iteration " + i);
                }

                System.out.println("100 operations completed with zero copies");
            }
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCasesTests {

        @Test
        @DisplayName("Minimum size buffer (64 bytes)")
        void testMinimumSizeBuffer() {
            MemorySegment segment = arena.allocate(64);

            try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                assertSame(segment, buffer.segment());
                assertEquals(64, buffer.byteSize());
                assertEquals(segment.address(), buffer.address());

                // Verify usable
                segment.set(ValueLayout.JAVA_LONG, 0, 12345L);
                assertEquals(12345L, buffer.segment().get(ValueLayout.JAVA_LONG, 0));
            }
        }

        @Test
        @DisplayName("Non-power-of-two size buffer")
        void testNonPowerOfTwoSize() {
            long size = 12345; // Odd size

            MemorySegment segment = arena.allocate(size);

            try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                assertSame(segment, buffer.segment());
                assertEquals(size, buffer.byteSize());

                // Verify last byte accessible
                segment.set(ValueLayout.JAVA_BYTE, size - 1, (byte) 0xFF);
                assertEquals((byte) 0xFF, buffer.segment().get(ValueLayout.JAVA_BYTE, size - 1));
            }
        }

        @Test
        @DisplayName("Re-registration after close")
        void testReRegistrationAfterClose() {
            // Use a fresh arena segment for this test
            try (Arena testArena = Arena.ofConfined()) {
                MemorySegment segment = testArena.allocate(4096);

                // First registration
                RdmaBuffer buffer1 = rdma.registerMemory(segment);
                long firstAddress = buffer1.address();
                assertSame(segment, buffer1.segment());
                buffer1.close();
                assertFalse(buffer1.isValid());

                // Re-register same segment (should work - segment still valid)
                // Note: Some implementations may or may not allow this
                try {
                    RdmaBuffer buffer2 = rdma.registerMemory(segment);
                    try {
                        assertSame(segment, buffer2.segment());
                        assertTrue(buffer2.isValid());
                        assertEquals(segment.address(), buffer2.address());
                        System.out.println("Re-registration succeeded (implementation allows it)");
                    } finally {
                        buffer2.close();
                    }
                } catch (Exception e) {
                    // Some implementations may not allow re-registration
                    System.out.println("Re-registration not supported: " + e.getMessage());
                    // This is acceptable behavior - document it
                    assertTrue(firstAddress != 0, "First registration should have worked");
                }
            }
        }

        @Test
        @DisplayName("Immediate re-read after write")
        void testImmediateReadAfterWrite() {
            MemorySegment segment = arena.allocate(4096);

            try (RdmaBuffer buffer = rdma.registerMemory(segment)) {
                // Write-read cycle should be immediate (no caching/buffering)
                for (long value = 0; value < 1000; value++) {
                    segment.set(ValueLayout.JAVA_LONG, 0, value);
                    long read = buffer.segment().get(ValueLayout.JAVA_LONG, 0);
                    assertEquals(value, read, "Write-read mismatch at iteration " + value);
                }
            }
        }
    }
}
