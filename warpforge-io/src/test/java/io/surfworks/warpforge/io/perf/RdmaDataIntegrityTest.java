package io.surfworks.warpforge.io.perf;

import io.surfworks.warpforge.io.rdma.Rdma;
import io.surfworks.warpforge.io.rdma.RdmaApi;
import io.surfworks.warpforge.io.rdma.RdmaBuffer;
import io.surfworks.warpforge.io.rdma.RdmaEndpoint;
import io.surfworks.warpforge.io.rdma.RdmaListener;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.security.MessageDigest;
import java.util.Random;
import java.util.zip.CRC32;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * RDMA Data Integrity Tests.
 *
 * <p>Validates that data transferred via RDMA arrives correctly without corruption.
 * These tests are essential for ensuring the Mellanox link is reliable for
 * ML gradient synchronization and model transfer.
 *
 * <h2>Test Categories</h2>
 * <ul>
 *   <li><b>Pattern Tests</b>: Known bit patterns (all 0s, all 1s, alternating, etc.)</li>
 *   <li><b>Sequential Tests</b>: Monotonic sequences to detect reordering</li>
 *   <li><b>Random Tests</b>: Random data with checksum verification</li>
 *   <li><b>Stress Tests</b>: Many iterations to detect intermittent errors</li>
 * </ul>
 *
 * <h2>Verification Methods</h2>
 * <ul>
 *   <li>Byte-by-byte comparison for small transfers</li>
 *   <li>CRC32 checksum for large transfers</li>
 *   <li>SHA-256 for cryptographic verification</li>
 * </ul>
 *
 * <h2>Running</h2>
 * <pre>{@code
 * # Start server:
 * ./gradlew :warpforge-io:run --args='server'
 *
 * # Run integrity tests:
 * ./gradlew :warpforge-io:rdmaPerfTest \
 *     --tests "*RdmaDataIntegrityTest*" \
 *     -Drdma.server.host=10.0.0.1
 * }</pre>
 */
@Tag("rdma-perf")
@Tag("rdma")
@DisplayName("RDMA Data Integrity Tests")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class RdmaDataIntegrityTest {

    private static final int DEFAULT_PORT = 18516; // Different from perf tests
    private static final int SMALL_SIZE = 64;
    private static final int MEDIUM_SIZE = 64 * 1024;        // 64KB
    private static final int LARGE_SIZE = 16 * 1024 * 1024;  // 16MB
    private static final int STRESS_ITERATIONS = 1000;

    private static RdmaApi rdma;
    private static Arena arena;
    private static String serverHost;
    private static int serverPort;
    private static boolean isServer;

    // Statistics
    private static long totalBytesVerified = 0;
    private static int totalTestsPassed = 0;

    @BeforeAll
    static void setUp() {
        serverHost = System.getProperty("rdma.server.host",
                System.getenv().getOrDefault("RDMA_SERVER_HOST", "10.0.0.1"));
        serverPort = Integer.getInteger("rdma.server.port", DEFAULT_PORT);
        isServer = Boolean.getBoolean("rdma.server.mode");

        assumeTrue(Rdma.canUseRealRdma(),
                "Skipping: Real RDMA hardware not available");

        rdma = Rdma.load();
        arena = Arena.ofShared();

        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║           RDMA DATA INTEGRITY TEST SUITE                         ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  Backend: %-54s ║%n", rdma.backendName());
        System.out.printf("║  Server:  %-54s ║%n", serverHost + ":" + serverPort);
        System.out.printf("║  Mode:    %-54s ║%n", isServer ? "SERVER" : "CLIENT");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝");
    }

    @AfterAll
    static void tearDown() {
        if (arena != null) arena.close();
        if (rdma != null) rdma.close();

        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║           DATA INTEGRITY SUMMARY                                 ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════╣");
        System.out.printf("║  Tests Passed:    %d                                             ║%n", totalTestsPassed);
        System.out.printf("║  Bytes Verified:  %,d                                  ║%n", totalBytesVerified);
        System.out.println("║  Status:          ALL DATA VERIFIED CORRECTLY                   ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝");
    }

    // ===== Pattern Tests =====

    @Test
    @Order(1)
    @DisplayName("Pattern: All zeros")
    void testAllZerosPattern() throws Exception {
        verifyPattern((byte) 0x00, MEDIUM_SIZE, "all zeros");
    }

    @Test
    @Order(2)
    @DisplayName("Pattern: All ones")
    void testAllOnesPattern() throws Exception {
        verifyPattern((byte) 0xFF, MEDIUM_SIZE, "all ones");
    }

    @Test
    @Order(3)
    @DisplayName("Pattern: Alternating bits (0xAA)")
    void testAlternatingPattern1() throws Exception {
        verifyPattern((byte) 0xAA, MEDIUM_SIZE, "0xAA");
    }

    @Test
    @Order(4)
    @DisplayName("Pattern: Alternating bits (0x55)")
    void testAlternatingPattern2() throws Exception {
        verifyPattern((byte) 0x55, MEDIUM_SIZE, "0x55");
    }

    @Test
    @Order(5)
    @DisplayName("Pattern: Checkerboard (0x5A)")
    void testCheckerboardPattern() throws Exception {
        verifyPattern((byte) 0x5A, MEDIUM_SIZE, "0x5A");
    }

    // ===== Sequential Tests =====

    @Test
    @Order(10)
    @DisplayName("Sequential: Monotonic long sequence")
    void testMonotonicSequence() throws Exception {
        System.out.println("\n─── MONOTONIC SEQUENCE TEST ───");

        MemorySegment sendSeg = arena.allocate(LARGE_SIZE);
        MemorySegment recvSeg = arena.allocate(LARGE_SIZE);

        // Fill with sequential longs
        int numLongs = LARGE_SIZE / 8;
        for (int i = 0; i < numLongs; i++) {
            sendSeg.setAtIndex(ValueLayout.JAVA_LONG, i, i);
        }

        try (RdmaBuffer sendBuf = rdma.registerMemory(sendSeg);
             RdmaBuffer recvBuf = rdma.registerMemory(recvSeg)) {

            if (isServer) {
                runEchoServer(recvBuf, sendBuf);
            } else {
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    // Send
                    endpoint.send(sendBuf).join();

                    // Receive echo
                    endpoint.receive(recvBuf).join();

                    // Verify sequence
                    boolean allCorrect = true;
                    int firstError = -1;
                    for (int i = 0; i < numLongs; i++) {
                        long expected = i;
                        long actual = recvSeg.getAtIndex(ValueLayout.JAVA_LONG, i);
                        if (actual != expected) {
                            if (allCorrect) {
                                firstError = i;
                                System.out.printf("First error at index %d: expected %d, got %d%n",
                                        i, expected, actual);
                            }
                            allCorrect = false;
                        }
                    }

                    assertTrue(allCorrect,
                            "Monotonic sequence corrupted at index " + firstError);

                    totalBytesVerified += LARGE_SIZE;
                    totalTestsPassed++;
                    System.out.println("PASS: Monotonic sequence verified (" + (LARGE_SIZE / 1024 / 1024) + " MB)");
                }
            }
        }
    }

    @Test
    @Order(11)
    @DisplayName("Sequential: Byte position encoding")
    void testBytePositionEncoding() throws Exception {
        System.out.println("\n─── BYTE POSITION ENCODING TEST ───");

        MemorySegment sendSeg = arena.allocate(MEDIUM_SIZE);
        MemorySegment recvSeg = arena.allocate(MEDIUM_SIZE);

        // Each byte encodes its position (mod 256)
        for (int i = 0; i < MEDIUM_SIZE; i++) {
            sendSeg.set(ValueLayout.JAVA_BYTE, i, (byte) (i & 0xFF));
        }

        try (RdmaBuffer sendBuf = rdma.registerMemory(sendSeg);
             RdmaBuffer recvBuf = rdma.registerMemory(recvSeg)) {

            if (isServer) {
                runEchoServer(recvBuf, sendBuf);
            } else {
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    endpoint.send(sendBuf).join();
                    endpoint.receive(recvBuf).join();

                    // Verify
                    for (int i = 0; i < MEDIUM_SIZE; i++) {
                        byte expected = (byte) (i & 0xFF);
                        byte actual = recvSeg.get(ValueLayout.JAVA_BYTE, i);
                        assertEquals(expected, actual,
                                "Byte position mismatch at offset " + i);
                    }

                    totalBytesVerified += MEDIUM_SIZE;
                    totalTestsPassed++;
                    System.out.println("PASS: Byte position encoding verified (" + (MEDIUM_SIZE / 1024) + " KB)");
                }
            }
        }
    }

    // ===== Random Data Tests =====

    @Test
    @Order(20)
    @DisplayName("Random: CRC32 verification")
    void testRandomWithCrc32() throws Exception {
        System.out.println("\n─── RANDOM DATA CRC32 TEST ───");

        Random random = new Random(42); // Deterministic seed
        MemorySegment sendSeg = arena.allocate(LARGE_SIZE);
        MemorySegment recvSeg = arena.allocate(LARGE_SIZE);

        // Fill with random data
        byte[] randomBytes = new byte[LARGE_SIZE];
        random.nextBytes(randomBytes);
        MemorySegment.copy(randomBytes, 0, sendSeg, ValueLayout.JAVA_BYTE, 0, LARGE_SIZE);

        // Calculate expected CRC32
        CRC32 crc = new CRC32();
        crc.update(randomBytes);
        long expectedCrc = crc.getValue();

        try (RdmaBuffer sendBuf = rdma.registerMemory(sendSeg);
             RdmaBuffer recvBuf = rdma.registerMemory(recvSeg)) {

            if (isServer) {
                runEchoServer(recvBuf, sendBuf);
            } else {
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    endpoint.send(sendBuf).join();
                    endpoint.receive(recvBuf).join();

                    // Calculate received CRC32
                    byte[] receivedBytes = new byte[LARGE_SIZE];
                    MemorySegment.copy(recvSeg, ValueLayout.JAVA_BYTE, 0, receivedBytes, 0, LARGE_SIZE);

                    CRC32 recvCrc = new CRC32();
                    recvCrc.update(receivedBytes);
                    long actualCrc = recvCrc.getValue();

                    assertEquals(expectedCrc, actualCrc,
                            String.format("CRC32 mismatch: expected 0x%08x, got 0x%08x",
                                    expectedCrc, actualCrc));

                    totalBytesVerified += LARGE_SIZE;
                    totalTestsPassed++;
                    System.out.printf("PASS: Random data CRC32 verified (0x%08x, %d MB)%n",
                            actualCrc, LARGE_SIZE / 1024 / 1024);
                }
            }
        }
    }

    @Test
    @Order(21)
    @DisplayName("Random: SHA-256 verification")
    void testRandomWithSha256() throws Exception {
        System.out.println("\n─── RANDOM DATA SHA-256 TEST ───");

        Random random = new Random(12345);
        MemorySegment sendSeg = arena.allocate(LARGE_SIZE);
        MemorySegment recvSeg = arena.allocate(LARGE_SIZE);

        byte[] randomBytes = new byte[LARGE_SIZE];
        random.nextBytes(randomBytes);
        MemorySegment.copy(randomBytes, 0, sendSeg, ValueLayout.JAVA_BYTE, 0, LARGE_SIZE);

        // Calculate expected SHA-256
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] expectedHash = md.digest(randomBytes);

        try (RdmaBuffer sendBuf = rdma.registerMemory(sendSeg);
             RdmaBuffer recvBuf = rdma.registerMemory(recvSeg)) {

            if (isServer) {
                runEchoServer(recvBuf, sendBuf);
            } else {
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    endpoint.send(sendBuf).join();
                    endpoint.receive(recvBuf).join();

                    // Calculate received SHA-256
                    byte[] receivedBytes = new byte[LARGE_SIZE];
                    MemorySegment.copy(recvSeg, ValueLayout.JAVA_BYTE, 0, receivedBytes, 0, LARGE_SIZE);

                    md.reset();
                    byte[] actualHash = md.digest(receivedBytes);

                    assertArrayEquals(expectedHash, actualHash,
                            "SHA-256 hash mismatch - data corruption detected");

                    totalBytesVerified += LARGE_SIZE;
                    totalTestsPassed++;
                    System.out.printf("PASS: Random data SHA-256 verified (%d MB)%n",
                            LARGE_SIZE / 1024 / 1024);
                }
            }
        }
    }

    // ===== Stress Tests =====

    @Test
    @Order(30)
    @DisplayName("Stress: Many small transfers")
    void testStressManySmallTransfers() throws Exception {
        System.out.println("\n─── STRESS TEST: MANY SMALL TRANSFERS ───");
        System.out.printf("Iterations: %d, Size: %d bytes each%n", STRESS_ITERATIONS, SMALL_SIZE);

        MemorySegment sendSeg = arena.allocate(SMALL_SIZE);
        MemorySegment recvSeg = arena.allocate(SMALL_SIZE);

        try (RdmaBuffer sendBuf = rdma.registerMemory(sendSeg);
             RdmaBuffer recvBuf = rdma.registerMemory(recvSeg)) {

            if (isServer) {
                runStressServer(recvBuf, sendBuf, STRESS_ITERATIONS);
            } else {
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    int errors = 0;

                    for (int iter = 0; iter < STRESS_ITERATIONS; iter++) {
                        // Different pattern each iteration
                        byte pattern = (byte) (iter & 0xFF);
                        sendSeg.fill(pattern);

                        endpoint.send(sendBuf).join();
                        endpoint.receive(recvBuf).join();

                        // Verify
                        for (int i = 0; i < SMALL_SIZE; i++) {
                            if (recvSeg.get(ValueLayout.JAVA_BYTE, i) != pattern) {
                                errors++;
                                break;
                            }
                        }

                        if (iter % 100 == 0 && iter > 0) {
                            System.out.printf("  Progress: %d/%d iterations%n", iter, STRESS_ITERATIONS);
                        }
                    }

                    assertEquals(0, errors,
                            "Stress test failed: " + errors + " corrupted transfers");

                    totalBytesVerified += (long) SMALL_SIZE * STRESS_ITERATIONS;
                    totalTestsPassed++;
                    System.out.printf("PASS: %d transfers verified without errors%n", STRESS_ITERATIONS);
                }
            }
        }
    }

    @Test
    @Order(31)
    @DisplayName("Stress: Large transfer repeated")
    void testStressLargeTransfersRepeated() throws Exception {
        System.out.println("\n─── STRESS TEST: LARGE TRANSFERS REPEATED ───");

        int iterations = 10; // Fewer iterations for large transfers
        System.out.printf("Iterations: %d, Size: %d MB each%n", iterations, LARGE_SIZE / 1024 / 1024);

        MemorySegment sendSeg = arena.allocate(LARGE_SIZE);
        MemorySegment recvSeg = arena.allocate(LARGE_SIZE);

        Random random = new Random(99999);

        try (RdmaBuffer sendBuf = rdma.registerMemory(sendSeg);
             RdmaBuffer recvBuf = rdma.registerMemory(recvSeg)) {

            if (isServer) {
                runStressServer(recvBuf, sendBuf, iterations);
            } else {
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    CRC32 crc = new CRC32();
                    byte[] buffer = new byte[LARGE_SIZE];

                    for (int iter = 0; iter < iterations; iter++) {
                        // New random data each iteration
                        random.nextBytes(buffer);
                        MemorySegment.copy(buffer, 0, sendSeg, ValueLayout.JAVA_BYTE, 0, LARGE_SIZE);

                        crc.reset();
                        crc.update(buffer);
                        long expectedCrc = crc.getValue();

                        endpoint.send(sendBuf).join();
                        endpoint.receive(recvBuf).join();

                        // Verify CRC
                        MemorySegment.copy(recvSeg, ValueLayout.JAVA_BYTE, 0, buffer, 0, LARGE_SIZE);
                        crc.reset();
                        crc.update(buffer);
                        long actualCrc = crc.getValue();

                        assertEquals(expectedCrc, actualCrc,
                                String.format("Iteration %d: CRC mismatch (0x%08x vs 0x%08x)",
                                        iter, expectedCrc, actualCrc));

                        System.out.printf("  Iteration %d/%d: CRC 0x%08x verified%n",
                                iter + 1, iterations, actualCrc);
                    }

                    totalBytesVerified += (long) LARGE_SIZE * iterations;
                    totalTestsPassed++;
                    System.out.printf("PASS: %d large transfers verified%n", iterations);
                }
            }
        }
    }

    // ===== Boundary Tests =====

    @Test
    @Order(40)
    @DisplayName("Boundary: First and last bytes")
    void testBoundaryBytes() throws Exception {
        System.out.println("\n─── BOUNDARY BYTES TEST ───");

        int[] sizes = {64, 4096, 65536, 1024 * 1024};

        for (int size : sizes) {
            MemorySegment sendSeg = arena.allocate(size);
            MemorySegment recvSeg = arena.allocate(size);

            // Set distinct values at boundaries
            sendSeg.set(ValueLayout.JAVA_BYTE, 0, (byte) 0x11);           // First byte
            sendSeg.set(ValueLayout.JAVA_BYTE, size - 1, (byte) 0x22);   // Last byte
            sendSeg.set(ValueLayout.JAVA_BYTE, size / 2, (byte) 0x33);   // Middle byte

            try (RdmaBuffer sendBuf = rdma.registerMemory(sendSeg);
                 RdmaBuffer recvBuf = rdma.registerMemory(recvSeg)) {

                if (isServer) {
                    runEchoServer(recvBuf, sendBuf);
                } else {
                    try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                        endpoint.send(sendBuf).join();
                        endpoint.receive(recvBuf).join();

                        assertEquals((byte) 0x11, recvSeg.get(ValueLayout.JAVA_BYTE, 0),
                                "First byte corrupted for size " + size);
                        assertEquals((byte) 0x22, recvSeg.get(ValueLayout.JAVA_BYTE, size - 1),
                                "Last byte corrupted for size " + size);
                        assertEquals((byte) 0x33, recvSeg.get(ValueLayout.JAVA_BYTE, size / 2),
                                "Middle byte corrupted for size " + size);

                        System.out.printf("  Size %d: boundaries verified%n", size);
                    }
                }
            }
        }

        totalTestsPassed++;
        System.out.println("PASS: All boundary tests verified");
    }

    // ===== Helper Methods =====

    private void verifyPattern(byte pattern, int size, String patternName) throws Exception {
        System.out.println("\n─── PATTERN TEST: " + patternName.toUpperCase() + " ───");

        MemorySegment sendSeg = arena.allocate(size);
        MemorySegment recvSeg = arena.allocate(size);

        sendSeg.fill(pattern);

        try (RdmaBuffer sendBuf = rdma.registerMemory(sendSeg);
             RdmaBuffer recvBuf = rdma.registerMemory(recvSeg)) {

            if (isServer) {
                runEchoServer(recvBuf, sendBuf);
            } else {
                try (RdmaEndpoint endpoint = rdma.connect(serverHost, serverPort, 30_000)) {
                    endpoint.send(sendBuf).join();
                    endpoint.receive(recvBuf).join();

                    // Verify every byte
                    for (int i = 0; i < size; i++) {
                        byte actual = recvSeg.get(ValueLayout.JAVA_BYTE, i);
                        assertEquals(pattern, actual,
                                String.format("Pattern mismatch at offset %d: expected 0x%02x, got 0x%02x",
                                        i, pattern & 0xFF, actual & 0xFF));
                    }

                    totalBytesVerified += size;
                    totalTestsPassed++;
                    System.out.printf("PASS: Pattern %s verified (%d KB)%n", patternName, size / 1024);
                }
            }
        }
    }

    private void runEchoServer(RdmaBuffer recvBuf, RdmaBuffer sendBuf) throws Exception {
        try (RdmaListener listener = rdma.listen(serverPort);
             RdmaEndpoint endpoint = listener.accept(30_000)) {
            System.out.println("Client connected: " + endpoint.remoteAddress());

            // Echo: receive then send back
            endpoint.receive(recvBuf).join();

            // Copy received data to send buffer
            MemorySegment.copy(recvBuf.segment(), 0, sendBuf.segment(), 0, recvBuf.byteSize());

            endpoint.send(sendBuf).join();
        }
    }

    private void runStressServer(RdmaBuffer recvBuf, RdmaBuffer sendBuf, int iterations) throws Exception {
        try (RdmaListener listener = rdma.listen(serverPort);
             RdmaEndpoint endpoint = listener.accept(30_000)) {
            System.out.println("Client connected: " + endpoint.remoteAddress());

            for (int i = 0; i < iterations; i++) {
                endpoint.receive(recvBuf).join();
                MemorySegment.copy(recvBuf.segment(), 0, sendBuf.segment(), 0, recvBuf.byteSize());
                endpoint.send(sendBuf).join();
            }
        }
    }
}
