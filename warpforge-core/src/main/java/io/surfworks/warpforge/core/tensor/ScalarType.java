package io.surfworks.warpforge.core.tensor;

import io.surfworks.snakeburger.stablehlo.StableHloAst;

import java.lang.foreign.ValueLayout;

/**
 * Scalar element types for tensors.
 * Maps from StableHLO AST types and provides byte size information.
 */
public enum ScalarType {
    F16(2, false, true),
    F32(4, false, true),
    F64(8, false, true),
    BF16(2, false, true),
    I1(1, true, false),
    I8(1, true, false),
    I16(2, true, false),
    I32(4, true, false),
    I64(8, true, false),
    BOOL(1, false, false);

    private final int byteSize;
    private final boolean isInteger;
    private final boolean isFloating;

    ScalarType(int byteSize, boolean isInteger, boolean isFloating) {
        this.byteSize = byteSize;
        this.isInteger = isInteger;
        this.isFloating = isFloating;
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
        };
    }

    /**
     * Parse from NumPy dtype string (e.g., "<f4", ">f8", "<i4").
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
        };
    }
}
