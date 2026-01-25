package io.surfworks.warpforge.data.stablehlo;

import java.util.List;

/**
 * StableHLO type definitions for warpforge-data.
 *
 * <p>This is a standalone representation that doesn't depend on snakeburger-core
 * (which requires Babylon JDK). It provides the minimal type system needed for
 * benchmark model representation and StableHLO MLIR emission.
 */
public final class StableHloTypes {

    private StableHloTypes() {}

    /**
     * Base interface for MLIR types.
     */
    public sealed interface Type permits ScalarType, TensorType, FunctionType {
        /**
         * Convert to MLIR text representation.
         */
        String toMlir();
    }

    /**
     * Scalar element types: f32, f64, i32, i64, bf16, etc.
     */
    public enum ScalarType implements Type {
        F16("f16"),
        F32("f32"),
        F64("f64"),
        BF16("bf16"),
        I1("i1"),
        I8("i8"),
        I16("i16"),
        I32("i32"),
        I64("i64");

        private final String mlirName;

        ScalarType(String mlirName) {
            this.mlirName = mlirName;
        }

        @Override
        public String toMlir() {
            return mlirName;
        }

        /**
         * Parse scalar type from MLIR string.
         */
        public static ScalarType fromMlir(String s) {
            return switch (s) {
                case "f16" -> F16;
                case "f32" -> F32;
                case "f64" -> F64;
                case "bf16" -> BF16;
                case "i1" -> I1;
                case "i8" -> I8;
                case "i16" -> I16;
                case "i32" -> I32;
                case "i64" -> I64;
                default -> throw new IllegalArgumentException("Unknown scalar type: " + s);
            };
        }

        /**
         * Whether this is a floating-point type.
         */
        public boolean isFloatingPoint() {
            return this == F16 || this == F32 || this == F64 || this == BF16;
        }

        /**
         * Whether this is an integer type.
         */
        public boolean isInteger() {
            return this == I1 || this == I8 || this == I16 || this == I32 || this == I64;
        }

        /**
         * Byte width of this type.
         */
        public int byteWidth() {
            return switch (this) {
                case I1 -> 1; // Stored as 1 byte
                case I8 -> 1;
                case F16, BF16, I16 -> 2;
                case F32, I32 -> 4;
                case F64, I64 -> 8;
            };
        }
    }

    /**
     * Tensor type: tensor&lt;4x8xf32&gt;
     */
    public record TensorType(long[] shape, ScalarType elementType) implements Type {

        /**
         * Create a tensor type with the given shape and element type.
         */
        public static TensorType of(ScalarType elementType, long... shape) {
            return new TensorType(shape.clone(), elementType);
        }

        /**
         * Rank (number of dimensions).
         */
        public int rank() {
            return shape.length;
        }

        /**
         * Dimension size at given index.
         */
        public long dim(int i) {
            return shape[i];
        }

        /**
         * Total number of elements.
         */
        public long elementCount() {
            if (shape.length == 0) return 1; // Scalar
            long count = 1;
            for (long d : shape) {
                count *= d;
            }
            return count;
        }

        /**
         * Size in bytes.
         */
        public long byteSize() {
            return elementCount() * elementType.byteWidth();
        }

        @Override
        public String toMlir() {
            if (shape.length == 0) {
                return "tensor<" + elementType.toMlir() + ">";
            }
            StringBuilder sb = new StringBuilder("tensor<");
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) sb.append("x");
                sb.append(shape[i]);
            }
            sb.append("x").append(elementType.toMlir()).append(">");
            return sb.toString();
        }

        /**
         * Parse tensor type from MLIR string.
         * Format: tensor&lt;4x8xf32&gt; or tensor&lt;f32&gt; for scalar
         */
        public static TensorType fromMlir(String s) {
            if (!s.startsWith("tensor<") || !s.endsWith(">")) {
                throw new IllegalArgumentException("Invalid tensor type: " + s);
            }
            String inner = s.substring(7, s.length() - 1);

            // Find the element type (last component after 'x' or the only component)
            int lastX = inner.lastIndexOf('x');
            if (lastX == -1) {
                // Scalar tensor: tensor<f32>
                return new TensorType(new long[0], ScalarType.fromMlir(inner));
            }

            String elementTypeStr = inner.substring(lastX + 1);
            String shapeStr = inner.substring(0, lastX);

            ScalarType elementType = ScalarType.fromMlir(elementTypeStr);

            // Parse shape
            String[] dims = shapeStr.split("x");
            long[] shape = new long[dims.length];
            for (int i = 0; i < dims.length; i++) {
                shape[i] = Long.parseLong(dims[i]);
            }

            return new TensorType(shape, elementType);
        }
    }

    /**
     * Function type: (tensor&lt;...&gt;, tensor&lt;...&gt;) -&gt; (tensor&lt;...&gt;)
     */
    public record FunctionType(List<Type> inputTypes, List<Type> resultTypes) implements Type {

        @Override
        public String toMlir() {
            StringBuilder sb = new StringBuilder("(");
            for (int i = 0; i < inputTypes.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(inputTypes.get(i).toMlir());
            }
            sb.append(") -> (");
            for (int i = 0; i < resultTypes.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(resultTypes.get(i).toMlir());
            }
            sb.append(")");
            return sb.toString();
        }
    }
}
