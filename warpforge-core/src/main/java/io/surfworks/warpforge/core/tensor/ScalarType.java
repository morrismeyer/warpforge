package io.surfworks.warpforge.core.tensor;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.core.formats.FormatParameters;

import java.lang.foreign.ValueLayout;

/**
 * Scalar element types for tensors.
 * Maps from StableHLO AST types and provides byte size information.
 *
 * <p>Includes support for P3109 small floating-point formats:
 * <ul>
 *   <li>FP8: E5M2, E4M3, E4M3FN, E8M0</li>
 *   <li>FP6: E3M2, E2M3</li>
 *   <li>FP4: E2M1, E1M2</li>
 * </ul>
 */
public enum ScalarType {
    // Standard IEEE 754 types
    F16(2, false, true, null),
    F32(4, false, true, null),
    F64(8, false, true, null),
    BF16(2, false, true, null),

    // P3109 FP8 types
    F8_E5M2(1, false, true, FormatParameters.FP8_E5M2),
    F8_E4M3(1, false, true, FormatParameters.FP8_E4M3),
    F8_E4M3FN(1, false, true, FormatParameters.FP8_E4M3FN),
    F8_E8M0(1, false, true, FormatParameters.FP8_E8M0),

    // P3109 FP6 types (packed: 4 values in 3 bytes, but we report 1 for element ops)
    F6_E3M2(1, false, true, FormatParameters.FP6_E3M2),
    F6_E2M3(1, false, true, FormatParameters.FP6_E2M3),

    // P3109 FP4 types (packed: 2 values per byte, but we report 1 for element ops)
    F4_E2M1(1, false, true, FormatParameters.FP4_E2M1),
    F4_E1M2(1, false, true, FormatParameters.FP4_E1M2),

    // Integer types
    I1(1, true, false, null),
    I8(1, true, false, null),
    I16(2, true, false, null),
    I32(4, true, false, null),
    I64(8, true, false, null),
    BOOL(1, false, false, null);

    private final int byteSize;
    private final boolean isInteger;
    private final boolean isFloating;
    private final FormatParameters formatParams;

    ScalarType(int byteSize, boolean isInteger, boolean isFloating, FormatParameters formatParams) {
        this.byteSize = byteSize;
        this.isInteger = isInteger;
        this.isFloating = isFloating;
        this.formatParams = formatParams;
    }

    public int byteSize() {
        return byteSize;
    }

    public boolean isInteger() {
        return isInteger;
    }

    public boolean isFloating() {
        return isFloating;
    }

    /**
     * Check if this is a P3109 small floating-point type.
     */
    public boolean isP3109Format() {
        return formatParams != null;
    }

    /**
     * Get the P3109 format parameters, if applicable.
     * @return FormatParameters or null if not a P3109 type
     */
    public FormatParameters formatParameters() {
        return formatParams;
    }

    /**
     * Get the actual bit width for this type.
     * For packed formats (FP4, FP6), this returns the true bit width.
     */
    public int bitWidth() {
        if (formatParams != null) {
            return formatParams.bitWidth();
        }
        return byteSize * 8;
    }

    /**
     * Calculate the packed byte size for count elements.
     * Handles sub-byte formats (FP4, FP6) correctly.
     */
    public long packedByteSize(long count) {
        if (formatParams == null) {
            return count * byteSize;
        }
        // Use ceiling division for sub-byte formats
        long totalBits = count * formatParams.bitWidth();
        return (totalBits + 7) / 8;
    }

    /**
     * Get the corresponding ValueLayout for FFM operations.
     */
    public ValueLayout valueLayout() {
        return switch (this) {
            case F32 -> ValueLayout.JAVA_FLOAT;
            case F64 -> ValueLayout.JAVA_DOUBLE;
            case I8 -> ValueLayout.JAVA_BYTE;
            case I16 -> ValueLayout.JAVA_SHORT;
            case I32 -> ValueLayout.JAVA_INT;
            case I64 -> ValueLayout.JAVA_LONG;
            case I1, BOOL -> ValueLayout.JAVA_BYTE;
            case F16, BF16 -> ValueLayout.JAVA_SHORT; // Half precision stored as short
            // P3109 types use JAVA_BYTE for element access
            case F8_E5M2, F8_E4M3, F8_E4M3FN, F8_E8M0 -> ValueLayout.JAVA_BYTE;
            case F6_E3M2, F6_E2M3 -> ValueLayout.JAVA_BYTE;
            case F4_E2M1, F4_E1M2 -> ValueLayout.JAVA_BYTE;
        };
    }

    /**
     * Convert from StableHLO AST ScalarType.
     */
    public static ScalarType fromAst(StableHloAst.ScalarType astType) {
        return switch (astType.name()) {
            case "f16" -> F16;
            case "f32" -> F32;
            case "f64" -> F64;
            case "bf16" -> BF16;
            case "f8_e5m2" -> F8_E5M2;
            case "f8_e4m3" -> F8_E4M3;
            case "f8_e4m3fn" -> F8_E4M3FN;
            case "i1" -> I1;
            case "i8" -> I8;
            case "i16" -> I16;
            case "i32" -> I32;
            case "i64" -> I64;
            default -> throw new IllegalArgumentException("Unknown scalar type: " + astType.name());
        };
    }

    /**
     * Convert to StableHLO AST ScalarType.
     *
     * <p>Note: P3109 small formats (FP4, FP6, FP8) that don't have direct
     * StableHLO representation are mapped using StableHloAst.ScalarType.of()
     * which creates a custom ScalarType with the appropriate name.
     */
    public StableHloAst.ScalarType toAst() {
        return switch (this) {
            case F16 -> StableHloAst.ScalarType.F16;
            case F32 -> StableHloAst.ScalarType.F32;
            case F64 -> StableHloAst.ScalarType.F64;
            case BF16 -> StableHloAst.ScalarType.BF16;
            case I1 -> StableHloAst.ScalarType.I1;
            case I8 -> StableHloAst.ScalarType.I8;
            case I16 -> StableHloAst.ScalarType.I16;
            case I32 -> StableHloAst.ScalarType.I32;
            case I64 -> StableHloAst.ScalarType.I64;
            case BOOL -> StableHloAst.ScalarType.I1;
            // P3109 FP8 types - use StableHloAst.ScalarType.of() for custom names
            case F8_E5M2 -> StableHloAst.ScalarType.of("f8_e5m2");
            case F8_E4M3 -> StableHloAst.ScalarType.of("f8_e4m3");
            case F8_E4M3FN -> StableHloAst.ScalarType.of("f8_e4m3fn");
            case F8_E8M0 -> StableHloAst.ScalarType.of("f8_e8m0");
            // P3109 FP6/FP4 types
            case F6_E3M2 -> StableHloAst.ScalarType.of("f6_e3m2");
            case F6_E2M3 -> StableHloAst.ScalarType.of("f6_e2m3");
            case F4_E2M1 -> StableHloAst.ScalarType.of("f4_e2m1");
            case F4_E1M2 -> StableHloAst.ScalarType.of("f4_e1m2");
        };
    }

    /**
     * Parse from NumPy dtype string (e.g., "<f4", ">f8", "<i4").
     * Also supports ML extension dtype strings for FP8/FP4 formats.
     */
    public static ScalarType fromNpyDtype(String dtype) {
        // Strip byte order prefix if present
        String typeStr = dtype;
        if (dtype.startsWith("<") || dtype.startsWith(">") || dtype.startsWith("|") || dtype.startsWith("=")) {
            typeStr = dtype.substring(1);
        }

        return switch (typeStr) {
            case "f2" -> F16;
            case "f4" -> F32;
            case "f8" -> F64;
            case "i1" -> I8;
            case "i2" -> I16;
            case "i4" -> I32;
            case "i8" -> I64;
            case "u1" -> I8;  // Unsigned byte treated as signed for now
            case "b1", "?" -> BOOL;
            // ML extension dtypes for FP8/FP4
            case "e5m2", "float8_e5m2" -> F8_E5M2;
            case "e4m3", "float8_e4m3" -> F8_E4M3;
            case "e4m3fn", "float8_e4m3fn" -> F8_E4M3FN;
            case "e8m0", "float8_e8m0" -> F8_E8M0;
            case "e3m2", "float6_e3m2" -> F6_E3M2;
            case "e2m3", "float6_e2m3" -> F6_E2M3;
            case "e2m1", "float4_e2m1" -> F4_E2M1;
            case "e1m2", "float4_e1m2" -> F4_E1M2;
            default -> throw new IllegalArgumentException("Unknown NumPy dtype: " + dtype);
        };
    }

    /**
     * Convert to NumPy dtype string (little-endian).
     */
    public String toNpyDtype() {
        return switch (this) {
            case F16 -> "<f2";
            case F32 -> "<f4";
            case F64 -> "<f8";
            case BF16 -> "<f2"; // BF16 not directly supported in NumPy
            case I1, BOOL -> "|b1";
            case I8 -> "|i1";
            case I16 -> "<i2";
            case I32 -> "<i4";
            case I64 -> "<i8";
            // ML extension dtypes (using ml_dtypes naming convention)
            case F8_E5M2 -> "|float8_e5m2";
            case F8_E4M3 -> "|float8_e4m3";
            case F8_E4M3FN -> "|float8_e4m3fn";
            case F8_E8M0 -> "|float8_e8m0";
            case F6_E3M2 -> "|float6_e3m2";
            case F6_E2M3 -> "|float6_e2m3";
            case F4_E2M1 -> "|float4_e2m1";
            case F4_E1M2 -> "|float4_e1m2";
        };
    }

    /**
     * Get a ScalarType from P3109 format parameters.
     *
     * @param params The format parameters
     * @return The corresponding ScalarType, or null if no match
     */
    public static ScalarType fromFormatParameters(FormatParameters params) {
        if (params == null) return null;
        for (ScalarType type : values()) {
            if (params.equals(type.formatParams)) {
                return type;
            }
        }
        return null;
    }
}
