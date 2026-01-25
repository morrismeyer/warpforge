package io.surfworks.warpforge.core.tensor;

import io.surfworks.warpforge.data.DType;
import io.surfworks.warpforge.data.TensorInfo;
import io.surfworks.warpforge.data.TensorView;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Bridge between warpforge-data tensor types and warpforge-core tensor types.
 *
 * <p>Provides conversions between:
 * <ul>
 *   <li>{@link DType} ↔ {@link ScalarType}</li>
 *   <li>{@link TensorInfo} → {@link TensorSpec}</li>
 *   <li>{@link TensorView} → {@link Tensor}</li>
 * </ul>
 *
 * <p>This bridge enables loading tensors from SafeTensors/GGUF files via warpforge-data
 * and using them in warpforge-core computations.
 */
public final class TensorBridge {

    private TensorBridge() {
        // Utility class
    }

    // =========================================================================
    // DType ↔ ScalarType Conversion
    // =========================================================================

    /**
     * Convert a warpforge-data DType to a warpforge-core ScalarType.
     *
     * @param dtype The DType from warpforge-data
     * @return The corresponding ScalarType
     * @throws IllegalArgumentException if the DType is not supported (e.g., quantized types)
     */
    public static ScalarType toScalarType(DType dtype) {
        return switch (dtype) {
            // Standard IEEE 754
            case F32 -> ScalarType.F32;
            case F64 -> ScalarType.F64;
            case F16 -> ScalarType.F16;
            case BF16 -> ScalarType.BF16;

            // P3109 FP8 types
            case F8_E5M2 -> ScalarType.F8_E5M2;
            case F8_E4M3 -> ScalarType.F8_E4M3;
            case F8_E4M3FN -> ScalarType.F8_E4M3FN;
            case F8_E8M0 -> ScalarType.F8_E8M0;

            // P3109 FP6 types
            case F6_E3M2 -> ScalarType.F6_E3M2;
            case F6_E2M3 -> ScalarType.F6_E2M3;

            // P3109 FP4 types
            case F4_E2M1 -> ScalarType.F4_E2M1;
            case F4_E1M2 -> ScalarType.F4_E1M2;

            // Integer types (signed)
            case I8 -> ScalarType.I8;
            case I16 -> ScalarType.I16;
            case I32 -> ScalarType.I32;
            case I64 -> ScalarType.I64;

            // Unsigned integers map to signed (best effort)
            case U8 -> ScalarType.I8;
            case U16 -> ScalarType.I16;
            case U32 -> ScalarType.I32;
            case U64 -> ScalarType.I64;

            // Boolean
            case BOOL -> ScalarType.BOOL;

            // Quantized types are not directly supported in ScalarType
            case Q4_0, Q4_1, Q4_K_M, Q5_0, Q5_1, Q5_K_M, Q8_0, Q8_K ->
                throw new IllegalArgumentException(
                    "Quantized type " + dtype + " requires dequantization before use in warpforge-core. " +
                    "Use TensorBridge.dequantize() first.");
        };
    }

    /**
     * Convert a warpforge-core ScalarType to a warpforge-data DType.
     *
     * @param scalarType The ScalarType from warpforge-core
     * @return The corresponding DType
     */
    public static DType toDType(ScalarType scalarType) {
        return switch (scalarType) {
            // Standard IEEE 754
            case F32 -> DType.F32;
            case F64 -> DType.F64;
            case F16 -> DType.F16;
            case BF16 -> DType.BF16;

            // P3109 FP8 types
            case F8_E5M2 -> DType.F8_E5M2;
            case F8_E4M3 -> DType.F8_E4M3;
            case F8_E4M3FN -> DType.F8_E4M3FN;
            case F8_E8M0 -> DType.F8_E8M0;

            // P3109 FP6 types
            case F6_E3M2 -> DType.F6_E3M2;
            case F6_E2M3 -> DType.F6_E2M3;

            // P3109 FP4 types
            case F4_E2M1 -> DType.F4_E2M1;
            case F4_E1M2 -> DType.F4_E1M2;

            // Integer types
            case I1, BOOL -> DType.BOOL;
            case I8 -> DType.I8;
            case I16 -> DType.I16;
            case I32 -> DType.I32;
            case I64 -> DType.I64;
        };
    }

    // =========================================================================
    // TensorInfo → TensorSpec Conversion
    // =========================================================================

    /**
     * Convert TensorInfo to TensorSpec.
     *
     * <p>Note: TensorInfo uses long[] for shape while TensorSpec uses int[].
     * This method validates that dimensions fit in int range.
     *
     * @param info The TensorInfo from warpforge-data
     * @return The corresponding TensorSpec
     * @throws IllegalArgumentException if shape dimensions exceed Integer.MAX_VALUE
     */
    public static TensorSpec toTensorSpec(TensorInfo info) {
        ScalarType scalarType = toScalarType(info.dtype());
        int[] shape = toLongArrayToIntArray(info.shape());
        return TensorSpec.of(scalarType, shape);
    }

    /**
     * Convert TensorView's info to TensorSpec.
     */
    public static TensorSpec toTensorSpec(TensorView view) {
        return toTensorSpec(view.info());
    }

    // =========================================================================
    // TensorView → Tensor Conversion
    // =========================================================================

    /**
     * Create a Tensor from a TensorView with zero-copy semantics.
     *
     * <p>The returned Tensor shares the underlying memory with the TensorView.
     * The Tensor does NOT own the memory, so the source (SafeTensors, GGUF, etc.)
     * must remain open for the Tensor to be valid.
     *
     * <p>This is the most efficient option when:
     * <ul>
     *   <li>You only need to read the tensor data</li>
     *   <li>The source file will remain open during processing</li>
     *   <li>You want to minimize memory usage</li>
     * </ul>
     *
     * @param view The TensorView from warpforge-data
     * @return A Tensor wrapping the same memory (zero-copy)
     */
    public static Tensor toTensorZeroCopy(TensorView view) {
        TensorSpec spec = toTensorSpec(view);
        return Tensor.fromMemorySegment(view.data(), spec);
    }

    /**
     * Create a Tensor from a TensorView by copying the data.
     *
     * <p>The returned Tensor owns its memory and is independent of the TensorView.
     * The source can be closed after this call.
     *
     * <p>Use this when:
     * <ul>
     *   <li>You need the tensor to outlive the source file</li>
     *   <li>You need to modify the tensor data</li>
     *   <li>You want to close the source file early</li>
     * </ul>
     *
     * @param view The TensorView from warpforge-data
     * @return A new Tensor with copied data (owns its memory)
     */
    public static Tensor toTensorCopy(TensorView view) {
        TensorSpec spec = toTensorSpec(view);
        Arena arena = Arena.ofConfined();
        MemorySegment newData = arena.allocate(spec.byteSize());
        MemorySegment.copy(view.data(), 0, newData, 0, view.byteSize());
        return Tensor.fromMemorySegment(newData, spec, arena);
    }

    /**
     * Create a Tensor from a TensorView using a shared Arena.
     *
     * <p>The returned Tensor uses the provided Arena for memory allocation.
     * The data is copied. The Tensor does not own the Arena.
     *
     * @param view  The TensorView from warpforge-data
     * @param arena The Arena to use for memory allocation
     * @return A new Tensor with copied data using the shared Arena
     */
    public static Tensor toTensor(TensorView view, Arena arena) {
        TensorSpec spec = toTensorSpec(view);
        MemorySegment newData = arena.allocate(spec.byteSize());
        MemorySegment.copy(view.data(), 0, newData, 0, view.byteSize());
        return Tensor.fromMemorySegment(newData, spec);
    }

    /**
     * Create a Tensor from a TensorView, converting to F32 if needed.
     *
     * <p>For F16, BF16, and FP8 types, this performs the conversion to F32.
     * For F32 input, this is equivalent to toTensorCopy().
     *
     * @param view The TensorView from warpforge-data
     * @return A new F32 Tensor with converted data
     */
    public static Tensor toTensorAsFloat32(TensorView view) {
        DType sourceDtype = view.dtype();

        // If already F32, just copy
        if (sourceDtype == DType.F32) {
            return toTensorCopy(view);
        }

        // Convert to F32
        int[] shape = toLongArrayToIntArray(view.shape());
        TensorSpec spec = TensorSpec.of(ScalarType.F32, shape);
        long elementCount = spec.elementCount();

        Arena arena = Arena.ofConfined();
        MemorySegment newData = arena.allocate(spec.byteSize());

        // Use TensorView's getFloatFlat which handles F16/BF16/FP8 conversion
        for (long i = 0; i < elementCount; i++) {
            float value = view.getFloatFlat(i);
            newData.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, i, value);
        }

        return Tensor.fromMemorySegment(newData, spec, arena);
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /**
     * Check if a DType can be directly converted to ScalarType without dequantization.
     */
    public static boolean isDirectlyConvertible(DType dtype) {
        return !dtype.isBlockQuantized();
    }

    /**
     * Convert long[] shape to int[] shape, validating that dimensions fit.
     */
    private static int[] toLongArrayToIntArray(long[] longArray) {
        int[] intArray = new int[longArray.length];
        for (int i = 0; i < longArray.length; i++) {
            if (longArray[i] > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                    "Dimension " + i + " exceeds Integer.MAX_VALUE: " + longArray[i]);
            }
            intArray[i] = (int) longArray[i];
        }
        return intArray;
    }
}
