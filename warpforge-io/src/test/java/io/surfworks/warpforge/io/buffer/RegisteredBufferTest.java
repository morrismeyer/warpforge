package io.surfworks.warpforge.io.buffer;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;
import io.surfworks.warpforge.io.rdma.Rdma;
import io.surfworks.warpforge.io.rdma.RdmaApi;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for RegisteredBuffer class.
 */
@Tag("unit")
@DisplayName("RegisteredBuffer Unit Tests")
class RegisteredBufferTest {

    private RdmaApi rdma;

    @BeforeEach
    void setUp() {
        rdma = Rdma.loadMock();
    }

    @AfterEach
    void tearDown() {
        if (rdma != null) rdma.close();
    }

    @Test
    @DisplayName("Should allocate buffer with TensorSpec")
    void testAllocateWithTensorSpec() {
        TensorSpec spec = TensorSpec.of(ScalarType.F32, 4, 4);

        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, spec)) {
            assertNotNull(buffer);
            assertNotNull(buffer.tensor());
            assertNotNull(buffer.rdmaBuffer());
            assertEquals(64, buffer.byteSize()); // 4*4*4 bytes for F32
            assertTrue(buffer.isValid());
        }
    }

    @Test
    @DisplayName("Should allocate buffer with ScalarType and shape")
    void testAllocateWithScalarTypeAndShape() {
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F64, 2, 3)) {
            assertNotNull(buffer);
            assertEquals(48, buffer.byteSize()); // 2*3*8 bytes for F64
            assertEquals(ScalarType.F64, buffer.tensor().dtype());
            assertArrayEquals(new int[]{2, 3}, buffer.tensor().shape());
        }
    }

    @Test
    @DisplayName("Should wrap existing tensor")
    void testWrapTensor() {
        try (Tensor tensor = Tensor.zeros(ScalarType.F32, 8);
             RegisteredBuffer buffer = RegisteredBuffer.wrap(rdma, tensor)) {

            assertNotNull(buffer);
            assertSame(tensor, buffer.tensor());
            assertEquals(32, buffer.byteSize()); // 8*4 bytes
            assertTrue(buffer.isValid());
        }
    }

    @Test
    @DisplayName("Should provide remote key from RDMA buffer")
    void testRemoteKey() {
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F32, 4)) {
            long remoteKey = buffer.remoteKey();
            assertEquals(buffer.rdmaBuffer().remoteKey(), remoteKey);
        }
    }

    @Test
    @DisplayName("Should provide address from RDMA buffer")
    void testAddress() {
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F32, 4)) {
            long address = buffer.address();
            assertEquals(buffer.rdmaBuffer().address(), address);
        }
    }

    @Test
    @DisplayName("Should provide memory segment")
    void testSegment() {
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F32, 4)) {
            assertNotNull(buffer.segment());
            assertEquals(16, buffer.segment().byteSize()); // 4*4 bytes
        }
    }

    @Test
    @DisplayName("Should invalidate on close")
    void testCloseInvalidates() {
        RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F32, 4);
        assertTrue(buffer.isValid());

        buffer.close();

        assertFalse(buffer.isValid());
    }

    @Test
    @DisplayName("Should have meaningful toString")
    void testToString() {
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F32, 4, 4)) {
            String str = buffer.toString();
            assertNotNull(str);
            assertTrue(str.contains("RegisteredBuffer"));
            assertTrue(str.contains("F32") || str.contains("float"));
        }
    }

    @Test
    @DisplayName("Should handle 1D tensor")
    void testOneDimensionalTensor() {
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.I32, 10)) {
            assertEquals(40, buffer.byteSize()); // 10*4 bytes
            assertArrayEquals(new int[]{10}, buffer.tensor().shape());
        }
    }

    @Test
    @DisplayName("Should handle 3D tensor")
    void testThreeDimensionalTensor() {
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F32, 2, 3, 4)) {
            assertEquals(96, buffer.byteSize()); // 2*3*4*4 bytes
            assertArrayEquals(new int[]{2, 3, 4}, buffer.tensor().shape());
        }
    }

    @Test
    @DisplayName("Should support different scalar types")
    void testDifferentScalarTypes() {
        // F16 (2 bytes)
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F16, 4)) {
            assertEquals(8, buffer.byteSize());
        }

        // F32 (4 bytes)
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F32, 4)) {
            assertEquals(16, buffer.byteSize());
        }

        // F64 (8 bytes)
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F64, 4)) {
            assertEquals(32, buffer.byteSize());
        }

        // I8 (1 byte)
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.I8, 4)) {
            assertEquals(4, buffer.byteSize());
        }

        // I32 (4 bytes)
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.I32, 4)) {
            assertEquals(16, buffer.byteSize());
        }

        // I64 (8 bytes)
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.I64, 4)) {
            assertEquals(32, buffer.byteSize());
        }
    }

    @Test
    @DisplayName("Should share data between tensor and segment")
    void testDataSharing() {
        try (RegisteredBuffer buffer = RegisteredBuffer.allocate(rdma, ScalarType.F32, 4)) {
            // Write through tensor
            float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
            buffer.tensor().copyFrom(data);

            // Read through segment using MemorySegment.copy (avoids byte order issues)
            float[] readBack = new float[4];
            MemorySegment.copy(buffer.segment(), ValueLayout.JAVA_FLOAT, 0, readBack, 0, 4);

            assertArrayEquals(data, readBack);
        }
    }
}
