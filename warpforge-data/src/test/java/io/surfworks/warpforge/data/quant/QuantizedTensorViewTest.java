package io.surfworks.warpforge.data.quant;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class QuantizedTensorViewTest {

    @Nested
    class BasicTests {

        @Test
        void testShape() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(16);

                TensorInfo info = new TensorInfo("test", DType.F32, new long[]{2, 2}, 0, 16);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.F32);

                assertEquals(2, view.shape().length);
                assertEquals(2, view.shape()[0]);
                assertEquals(2, view.shape()[1]);
            }
        }

        @Test
        void testElementCount() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(48);

                TensorInfo info = new TensorInfo("test", DType.F32, new long[]{2, 3, 4}, 0, 48);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.F32);

                assertEquals(24, view.elementCount());
            }
        }

        @Test
        void testQuantizationType() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(16);

                TensorInfo info = new TensorInfo("test", DType.I8, new long[]{16}, 0, 16);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.Q8_0);

                assertEquals(QuantizationType.Q8_0, view.quantizationType());
            }
        }
    }

    @Nested
    class F32AccessTests {

        @Test
        void testGetFloatFlat() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(16);
                data.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1.0f);
                data.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 2.0f);
                data.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 3.0f);
                data.setAtIndex(ValueLayout.JAVA_FLOAT, 3, 4.0f);

                TensorInfo info = new TensorInfo("test", DType.F32, new long[]{4}, 0, 16);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.F32);

                assertEquals(1.0f, view.getFloatFlat(0), 1e-6f);
                assertEquals(2.0f, view.getFloatFlat(1), 1e-6f);
                assertEquals(3.0f, view.getFloatFlat(2), 1e-6f);
                assertEquals(4.0f, view.getFloatFlat(3), 1e-6f);
            }
        }

        @Test
        void testGetFloatWithIndices() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(24);
                // 2x3 matrix row-major
                float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
                for (int i = 0; i < 6; i++) {
                    data.setAtIndex(ValueLayout.JAVA_FLOAT, i, values[i]);
                }

                TensorInfo info = new TensorInfo("test", DType.F32, new long[]{2, 3}, 0, 24);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.F32);

                assertEquals(1.0f, view.getFloat(0, 0), 1e-6f);
                assertEquals(2.0f, view.getFloat(0, 1), 1e-6f);
                assertEquals(3.0f, view.getFloat(0, 2), 1e-6f);
                assertEquals(4.0f, view.getFloat(1, 0), 1e-6f);
                assertEquals(5.0f, view.getFloat(1, 1), 1e-6f);
                assertEquals(6.0f, view.getFloat(1, 2), 1e-6f);
            }
        }
    }

    @Nested
    class DequantizationTests {

        @Test
        void testDequantizeF32() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(16);
                float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
                for (int i = 0; i < 4; i++) {
                    data.setAtIndex(ValueLayout.JAVA_FLOAT, i, values[i]);
                }

                TensorInfo info = new TensorInfo("test", DType.F32, new long[]{4}, 0, 16);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.F32);

                TensorView dequantized = view.dequantize(arena);

                assertEquals(4, dequantized.info().elementCount());
                assertEquals(DType.F32, dequantized.info().dtype());
                assertEquals(1.0f, dequantized.getFloatFlat(0), 1e-6f);
                assertEquals(4.0f, dequantized.getFloatFlat(3), 1e-6f);
            }
        }

        @Test
        void testDequantizeToArray() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(16);
                float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
                for (int i = 0; i < 4; i++) {
                    data.setAtIndex(ValueLayout.JAVA_FLOAT, i, values[i]);
                }

                TensorInfo info = new TensorInfo("test", DType.F32, new long[]{4}, 0, 16);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.F32);

                float[] result = view.dequantizeToArray();

                assertEquals(4, result.length);
                assertEquals(1.0f, result[0], 1e-6f);
                assertEquals(4.0f, result[3], 1e-6f);
            }
        }

        @Test
        void testDequantizeF16() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(4);
                // 1.0f in FP16
                data.setAtIndex(ValueLayout.JAVA_SHORT, 0, (short) 0x3C00);
                // 2.0f in FP16
                data.setAtIndex(ValueLayout.JAVA_SHORT, 1, (short) 0x4000);

                TensorInfo info = new TensorInfo("test", DType.F16, new long[]{2}, 0, 4);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.F16);

                float[] result = view.dequantizeToArray();

                assertEquals(1.0f, result[0], 1e-3f);
                assertEquals(2.0f, result[1], 1e-3f);
            }
        }
    }

    @Nested
    class CompressionRatioTests {

        @Test
        void testCompressionRatioF32() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(16);
                TensorInfo info = new TensorInfo("test", DType.F32, new long[]{4}, 0, 16);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.F32);

                assertEquals(1.0, view.compressionRatio(), 1e-6);
            }
        }

        @Test
        void testCompressionRatioQ4() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(18);  // One Q4_0 block
                TensorInfo info = new TensorInfo("test", DType.I8, new long[]{32}, 0, 18);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.Q4_0);

                assertEquals(8.0, view.compressionRatio(), 1e-6);
            }
        }

        @Test
        void testSizeBytes() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment data = arena.allocate(34);  // One Q8_0 block (2 + 32)
                TensorInfo info = new TensorInfo("test", DType.I8, new long[]{32}, 0, 34);
                QuantizedTensorView view = QuantizedTensorView.of(data, info, QuantizationType.Q8_0);

                assertEquals(34, view.quantizedSizeBytes());
                assertEquals(128, view.dequantizedSizeBytes());  // 32 * 4 bytes
            }
        }
    }
}
