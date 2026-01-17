package io.surfworks.warpforge.backend.nvidia;

import io.surfworks.warpforge.core.backend.GpuBackendCapabilities;
import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;
import org.junit.jupiter.api.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for NvidiaBackend.
 * These tests run without actual CUDA hardware using stub implementations.
 */
@DisplayName("NVIDIA Backend Tests")
class NvidiaBackendTest {

    private NvidiaBackend backend;

    @BeforeEach
    void setUp() {
        backend = new NvidiaBackend(0);
    }

    @AfterEach
    void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    @Test
    @DisplayName("Backend reports correct name")
    void testName() {
        assertEquals("nvidia", backend.name());
    }

    @Test
    @DisplayName("Backend reports device index")
    void testDeviceIndex() {
        assertEquals(0, backend.deviceIndex());

        try (NvidiaBackend backend1 = new NvidiaBackend(1)) {
            assertEquals(1, backend1.deviceIndex());
        }
    }

    @Test
    @DisplayName("GPU capabilities are populated")
    void testGpuCapabilities() {
        GpuBackendCapabilities caps = backend.gpuCapabilities();

        assertNotNull(caps);
        assertTrue(caps.deviceMemoryBytes() > 0);
        assertTrue(caps.computeUnits() > 0);
        assertTrue(caps.supportsFp16());
        assertTrue(caps.supportsBf16());
    }

    @Test
    @DisplayName("Base capabilities include GPU dtypes")
    void testBaseCapabilities() {
        var caps = backend.capabilities();

        assertNotNull(caps);
        assertTrue(caps.supportedDtypes().contains(ScalarType.F32));
        assertTrue(caps.supportedDtypes().contains(ScalarType.F64));
        assertTrue(caps.supportedDtypes().contains(ScalarType.F16));
        assertTrue(caps.supportedDtypes().contains(ScalarType.BF16));
        assertTrue(caps.supportsAsync());
    }

    @Test
    @DisplayName("Allocate device tensor")
    void testAllocateDevice() {
        TensorSpec spec = TensorSpec.of(ScalarType.F32, 64, 128);

        try (Tensor tensor = backend.allocateDevice(spec)) {
            assertNotNull(tensor);
            assertEquals(64 * 128, tensor.elementCount());
            assertEquals(ScalarType.F32, tensor.dtype());
        }
    }

    @Test
    @DisplayName("Copy to/from device")
    void testCopyToFromDevice() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
        try (Tensor hostTensor = Tensor.fromFloatArray(data, 4);
             Tensor deviceTensor = backend.copyToDevice(hostTensor);
             Tensor backToHost = backend.copyToHost(deviceTensor)) {

            assertArrayEquals(data, backToHost.toFloatArray(), 0.0001f);
        }
    }

    @Test
    @DisplayName("Allocate pinned memory")
    void testAllocatePinned() {
        TensorSpec spec = TensorSpec.of(ScalarType.F32, 1024);

        try (Tensor pinned = backend.allocatePinned(spec)) {
            assertNotNull(pinned);
            assertEquals(1024, pinned.elementCount());
            assertNotNull(pinned.data());
        }
    }

    @Test
    @DisplayName("Register memory segment for RDMA")
    void testRegisterForRdma() {
        TensorSpec spec = TensorSpec.of(ScalarType.F32, 256);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(spec.byteSize());

            // Fill with test data
            for (int i = 0; i < 256; i++) {
                segment.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, i * 0.5f);
            }

            Tensor registered = backend.registerForRdma(segment, spec);

            assertNotNull(registered);
            assertEquals(256, registered.elementCount());
            assertEquals(0.0f, registered.getFloatFlat(0), 0.0001f);
            assertEquals(127.5f, registered.getFloatFlat(255), 0.0001f);
        }
    }

    @Test
    @DisplayName("Stream creation and destruction")
    void testStreamManagement() {
        long stream = backend.createStream();
        assertTrue(stream != 0);

        // Should not throw
        backend.synchronizeStream(stream);
        backend.destroyStream(stream);
    }

    @Test
    @DisplayName("Device synchronization")
    void testDeviceSync() {
        // Should not throw
        backend.synchronizeDevice();
    }

    @Test
    @DisplayName("Memory info reports values")
    void testMemoryInfo() {
        long total = backend.totalDeviceMemory();
        long free = backend.freeDeviceMemory();
        long used = backend.usedDeviceMemory();

        assertTrue(total > 0);
        assertTrue(free >= 0);
        assertTrue(used >= 0);
        assertEquals(total, free + used);
    }

    @Test
    @DisplayName("Allocation tracks memory usage")
    void testMemoryTracking() {
        long usedBefore = backend.usedDeviceMemory();

        TensorSpec spec = TensorSpec.of(ScalarType.F32, 1024); // 4KB
        try (Tensor tensor = backend.allocateDevice(spec)) {
            long usedAfter = backend.usedDeviceMemory();
            assertEquals(usedBefore + spec.byteSize(), usedAfter);
        }
    }

    @Test
    @DisplayName("Backend throws after close")
    void testClosedBackend() {
        backend.close();

        assertThrows(IllegalStateException.class, () ->
            backend.allocate(TensorSpec.of(ScalarType.F32, 10)));
    }

    @Test
    @DisplayName("Double close is safe")
    void testDoubleClose() {
        backend.close();
        backend.close(); // Should not throw
    }

    @Test
    @Tag("nvidia")
    @DisplayName("CUDA availability check")
    void testCudaAvailability() {
        // This test only runs when nvidia tag is included
        // It checks actual CUDA presence
        boolean available = NvidiaBackend.isCudaAvailable();
        int deviceCount = NvidiaBackend.getDeviceCount();

        if (available) {
            assertTrue(deviceCount > 0, "CUDA available but no devices found");
        } else {
            assertEquals(0, deviceCount);
        }
    }
}
