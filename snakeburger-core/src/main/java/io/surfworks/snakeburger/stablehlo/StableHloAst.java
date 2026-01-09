package io.surfworks.snakeburger.stablehlo;

import java.util.List;
import java.util.Optional;

/**
 * AST classes for StableHLO MLIR representation.
 *
 * These classes capture the full structure of parsed StableHLO
 * for validation and translation to Babylon Op trees.
 */
public final class StableHloAst {

    private StableHloAst() {}

    // ==================== Types ====================

    /**
     * Base interface for all MLIR types.
     */
    public sealed interface Type permits TensorType, ScalarType, FunctionType {
        String toMlirString();
    }

    /**
     * Scalar element types: f32, f64, i32, i64, etc.
     */
    public record ScalarType(String name) implements Type {
        public static final ScalarType F16 = new ScalarType("f16");
        public static final ScalarType F32 = new ScalarType("f32");
        public static final ScalarType F64 = new ScalarType("f64");
        public static final ScalarType BF16 = new ScalarType("bf16");
        public static final ScalarType I1 = new ScalarType("i1");
        public static final ScalarType I8 = new ScalarType("i8");
        public static final ScalarType I16 = new ScalarType("i16");
        public static final ScalarType I32 = new ScalarType("i32");
        public static final ScalarType I64 = new ScalarType("i64");

        public static ScalarType of(String name) {
            return switch (name) {
                case "f16" -> F16;
                case "f32" -> F32;
                case "f64" -> F64;
                case "bf16" -> BF16;
                case "i1" -> I1;
                case "i8" -> I8;
                case "i16" -> I16;
                case "i32" -> I32;
                case "i64" -> I64;
                default -> new ScalarType(name);
            };
        }

        public boolean isFloatingPoint() {
            return name.startsWith("f") || name.equals("bf16");
        }

        public boolean isInteger() {
            return name.startsWith("i");
        }

        @Override
        public String toMlirString() {
            return name;
        }
    }

    /**
     * Tensor type: tensor<4x8xf32>
     */
    public record TensorType(List<Integer> shape, ScalarType elementType) implements Type {

        public int rank() {
            return shape.size();
        }

        public int dim(int i) {
            return shape.get(i);
        }

        public long elementCount() {
            long count = 1;
            for (int d : shape) {
                count *= d;
            }
            return count;
        }

        @Override
        public String toMlirString() {
            if (shape.isEmpty()) {
                return "tensor<" + elementType.toMlirString() + ">";
            }
            StringBuilder sb = new StringBuilder("tensor<");
            for (int i = 0; i < shape.size(); i++) {
                if (i > 0) sb.append("x");
                sb.append(shape.get(i));
            }
            sb.append("x").append(elementType.toMlirString()).append(">");
            return sb.toString();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof TensorType that)) return false;
            return shape.equals(that.shape) && elementType.equals(that.elementType);
        }
    }

    /**
     * Function type: (tensor<...>, tensor<...>) -> (tensor<...>)
     */
    public record FunctionType(List<Type> inputTypes, List<Type> resultTypes) implements Type {
        @Override
        public String toMlirString() {
            StringBuilder sb = new StringBuilder("(");
            for (int i = 0; i < inputTypes.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(inputTypes.get(i).toMlirString());
            }
            sb.append(") -> (");
            for (int i = 0; i < resultTypes.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(resultTypes.get(i).toMlirString());
            }
            sb.append(")");
            return sb.toString();
        }
    }

    // ==================== Values ====================

    /**
     * A value reference: %arg0, %1, %2_zero
     */
    public record Value(String name, Type type) {
        public String toMlirString() {
            return "%" + name;
        }

        @Override
        public String toString() {
            return "%" + name + " : " + type.toMlirString();
        }
    }

    // ==================== Attributes ====================

    /**
     * Base interface for MLIR attributes.
     */
    public sealed interface Attribute permits
            IntegerAttr, FloatAttr, StringAttr, ArrayAttr, DenseAttr, DotDimensionNumbers {
        String toMlirString();
    }

    public record IntegerAttr(long value) implements Attribute {
        @Override
        public String toMlirString() {
            return String.valueOf(value);
        }
    }

    public record FloatAttr(double value) implements Attribute {
        @Override
        public String toMlirString() {
            return String.valueOf(value);
        }
    }

    public record StringAttr(String value) implements Attribute {
        @Override
        public String toMlirString() {
            return "\"" + value + "\"";
        }
    }

    public record ArrayAttr(List<Attribute> values) implements Attribute {
        @Override
        public String toMlirString() {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < values.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(values.get(i).toMlirString());
            }
            sb.append("]");
            return sb.toString();
        }

        public List<Long> asIntegerList() {
            return values.stream()
                    .map(a -> ((IntegerAttr) a).value())
                    .toList();
        }
    }

    /**
     * Dense attribute for constants: dense<0.0>, dense<[1, 2, 3]>
     */
    public record DenseAttr(Object value, Type type) implements Attribute {
        @Override
        public String toMlirString() {
            return "dense<" + value + "> : " + type.toMlirString();
        }
    }

    /**
     * Dot dimension numbers for stablehlo.dot_general.
     */
    public record DotDimensionNumbers(
            List<Long> lhsBatchingDimensions,
            List<Long> rhsBatchingDimensions,
            List<Long> lhsContractingDimensions,
            List<Long> rhsContractingDimensions
    ) implements Attribute {
        @Override
        public String toMlirString() {
            return String.format(
                "#stablehlo.dot<lhs_batching_dimensions = %s, rhs_batching_dimensions = %s, " +
                "lhs_contracting_dimensions = %s, rhs_contracting_dimensions = %s>",
                formatList(lhsBatchingDimensions),
                formatList(rhsBatchingDimensions),
                formatList(lhsContractingDimensions),
                formatList(rhsContractingDimensions)
            );
        }

        private static String formatList(List<Long> list) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(list.get(i));
            }
            sb.append("]");
            return sb.toString();
        }
    }

    // ==================== Operations ====================

    /**
     * Base interface for all StableHLO operations.
     */
    public sealed interface Operation permits
            // Core ops
            DotGeneralOp, ConstantOp, ReturnOp, CustomCallOp,
            // Binary arithmetic
            AddOp, SubtractOp, MultiplyOp, DivideOp, MaximumOp, MinimumOp,
            PowerOp, RemainderOp, Atan2Op,
            // Unary math
            NegateOp, AbsOp, ExpOp, LogOp, TanhOp, SqrtOp, RsqrtOp,
            SinOp, CosOp, TanOp, CeilOp, FloorOp, SignOp, LogisticOp,
            Expm1Op, Log1pOp, CbrtOp, IsFiniteOp, PopcntOp, ClzOp,
            RoundNearestEvenOp, RoundNearestAfzOp,
            // Logical/bitwise
            NotOp, AndOp, OrOp, XorOp,
            ShiftLeftOp, ShiftRightArithmeticOp, ShiftRightLogicalOp,
            // Comparison and selection
            CompareOp, SelectOp, ClampOp,
            // Shape manipulation
            ReshapeOp, TransposeOp, BroadcastInDimOp, ConcatenateOp, SliceOp,
            ReverseOp, PadOp, IotaOp, GatherOp, ScatterOp,
            DynamicSliceOp, DynamicUpdateSliceOp,
            // Type conversion
            ConvertOp, BitcastConvertOp,
            // Reduction
            ReduceOp, ReduceWindowOp,
            // Convolution and neural network
            ConvolutionOp, BatchNormTrainingOp, BatchNormInferenceOp,
            // Control flow
            IfOp, WhileOp,
            // Other
            SortOp, RngOp, RngBitGeneratorOp {

        String opName();
        List<Value> results();
        List<Value> operands();
        TensorType tensorResultType();
    }

    /**
     * stablehlo.dot_general - General matrix multiplication.
     */
    public record DotGeneralOp(
            Value result,
            Value lhs,
            Value rhs,
            DotDimensionNumbers dimensionNumbers,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.dot_general"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /**
     * stablehlo.constant - Constant tensor value.
     */
    public record ConstantOp(
            Value result,
            DenseAttr value,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.constant"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(); }
    }

    /**
     * stablehlo.maximum - Element-wise maximum.
     */
    public record MaximumOp(
            Value result,
            Value lhs,
            Value rhs,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.maximum"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /**
     * stablehlo.add - Element-wise addition.
     */
    public record AddOp(
            Value result,
            Value lhs,
            Value rhs,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.add"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /**
     * stablehlo.multiply - Element-wise multiplication.
     */
    public record MultiplyOp(
            Value result,
            Value lhs,
            Value rhs,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.multiply"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /**
     * stablehlo.divide - Element-wise division.
     */
    public record DivideOp(
            Value result,
            Value lhs,
            Value rhs,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.divide"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /**
     * stablehlo.subtract - Element-wise subtraction.
     */
    public record SubtractOp(
            Value result,
            Value lhs,
            Value rhs,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.subtract"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /**
     * stablehlo.minimum - Element-wise minimum.
     */
    public record MinimumOp(
            Value result,
            Value lhs,
            Value rhs,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.minimum"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /** stablehlo.power - Element-wise power (x^y). */
    public record PowerOp(Value result, Value lhs, Value rhs, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.power"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /** stablehlo.remainder - Element-wise remainder (modulo). */
    public record RemainderOp(Value result, Value lhs, Value rhs, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.remainder"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /** stablehlo.atan2 - Element-wise arctangent of y/x. */
    public record Atan2Op(Value result, Value lhs, Value rhs, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.atan2"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /** stablehlo.and - Element-wise logical/bitwise AND. */
    public record AndOp(Value result, Value lhs, Value rhs, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.and"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /** stablehlo.or - Element-wise logical/bitwise OR. */
    public record OrOp(Value result, Value lhs, Value rhs, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.or"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /** stablehlo.xor - Element-wise logical/bitwise XOR. */
    public record XorOp(Value result, Value lhs, Value rhs, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.xor"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /** stablehlo.shift_left - Element-wise left shift. */
    public record ShiftLeftOp(Value result, Value lhs, Value rhs, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.shift_left"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /** stablehlo.shift_right_arithmetic - Element-wise arithmetic right shift. */
    public record ShiftRightArithmeticOp(Value result, Value lhs, Value rhs, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.shift_right_arithmetic"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /** stablehlo.shift_right_logical - Element-wise logical right shift. */
    public record ShiftRightLogicalOp(Value result, Value lhs, Value rhs, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.shift_right_logical"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /**
     * stablehlo.negate - Element-wise negation.
     */
    public record NegateOp(
            Value result,
            Value operand,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.negate"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.abs - Element-wise absolute value.
     */
    public record AbsOp(
            Value result,
            Value operand,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.abs"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.exponential - Element-wise exponential (e^x).
     */
    public record ExpOp(
            Value result,
            Value operand,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.exponential"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.log - Element-wise natural logarithm.
     */
    public record LogOp(
            Value result,
            Value operand,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.log"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.tanh - Element-wise hyperbolic tangent.
     */
    public record TanhOp(
            Value result,
            Value operand,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.tanh"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.sqrt - Element-wise square root.
     */
    public record SqrtOp(
            Value result,
            Value operand,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.sqrt"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.rsqrt - Element-wise reciprocal square root (1/sqrt(x)).
     */
    public record RsqrtOp(
            Value result,
            Value operand,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.rsqrt"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.sine - Element-wise sine. */
    public record SinOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.sine"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.cosine - Element-wise cosine. */
    public record CosOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.cosine"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.tan - Element-wise tangent. */
    public record TanOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.tan"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.ceil - Element-wise ceiling. */
    public record CeilOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.ceil"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.floor - Element-wise floor. */
    public record FloorOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.floor"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.sign - Element-wise sign. */
    public record SignOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.sign"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.logistic - Element-wise sigmoid (1/(1+e^-x)). */
    public record LogisticOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.logistic"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.exponential_minus_one - Element-wise e^x - 1. */
    public record Expm1Op(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.exponential_minus_one"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.log_plus_one - Element-wise log(1+x). */
    public record Log1pOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.log_plus_one"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.cbrt - Element-wise cube root. */
    public record CbrtOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.cbrt"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.is_finite - Element-wise finite check. */
    public record IsFiniteOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.is_finite"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.popcnt - Element-wise population count (number of set bits). */
    public record PopcntOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.popcnt"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.count_leading_zeros - Element-wise count of leading zeros. */
    public record ClzOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.count_leading_zeros"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.round_nearest_even - Round to nearest even. */
    public record RoundNearestEvenOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.round_nearest_even"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.round_nearest_afz - Round away from zero. */
    public record RoundNearestAfzOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.round_nearest_afz"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.not - Element-wise logical/bitwise NOT. */
    public record NotOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.not"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.reshape - Reshape tensor.
     */
    public record ReshapeOp(
            Value result,
            Value operand,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.reshape"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.transpose - Transpose tensor dimensions.
     */
    public record TransposeOp(
            Value result,
            Value operand,
            List<Long> permutation,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.transpose"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.broadcast_in_dim - Broadcast tensor.
     */
    public record BroadcastInDimOp(
            Value result,
            Value operand,
            List<Long> broadcastDimensions,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.broadcast_in_dim"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.reduce - Reduction operation.
     */
    public record ReduceOp(
            Value result,
            Value operand,
            Value initValue,
            List<Long> dimensions,
            String reducer, // "add", "max", etc.
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.reduce"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand, initValue); }
    }

    /**
     * Comparison direction for stablehlo.compare.
     */
    public enum ComparisonDirection {
        EQ, NE, GE, GT, LE, LT;

        public static ComparisonDirection fromString(String s) {
            return switch (s.toUpperCase()) {
                case "EQ" -> EQ;
                case "NE" -> NE;
                case "GE" -> GE;
                case "GT" -> GT;
                case "LE" -> LE;
                case "LT" -> LT;
                default -> throw new IllegalArgumentException("Unknown comparison direction: " + s);
            };
        }
    }

    /**
     * stablehlo.compare - Element-wise comparison.
     */
    public record CompareOp(
            Value result,
            Value lhs,
            Value rhs,
            ComparisonDirection direction,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.compare"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /**
     * stablehlo.select - Conditional selection (ternary).
     */
    public record SelectOp(
            Value result,
            Value pred,
            Value onTrue,
            Value onFalse,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.select"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(pred, onTrue, onFalse); }
    }

    /**
     * stablehlo.concatenate - Concatenate tensors along a dimension.
     */
    public record ConcatenateOp(
            Value result,
            List<Value> inputs,
            long dimension,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.concatenate"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return inputs; }
    }

    /**
     * stablehlo.slice - Extract a slice from a tensor.
     */
    public record SliceOp(
            Value result,
            Value operand,
            List<Long> startIndices,
            List<Long> limitIndices,
            List<Long> strides,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.slice"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /**
     * stablehlo.clamp - Clamp values to a range.
     */
    public record ClampOp(
            Value result,
            Value min,
            Value operand,
            Value max,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.clamp"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(min, operand, max); }
    }

    /**
     * stablehlo.convert - Type conversion.
     */
    public record ConvertOp(
            Value result,
            Value operand,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.convert"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.bitcast_convert - Bitcast to different element type. */
    public record BitcastConvertOp(Value result, Value operand, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.bitcast_convert"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    // ==================== Additional Shape Operations ====================

    /** stablehlo.reverse - Reverse tensor along dimensions. */
    public record ReverseOp(Value result, Value operand, List<Long> dimensions, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.reverse"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand); }
    }

    /** stablehlo.pad - Pad tensor with a value. */
    public record PadOp(
            Value result,
            Value operand,
            Value paddingValue,
            List<Long> edgePaddingLow,
            List<Long> edgePaddingHigh,
            List<Long> interiorPadding,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.pad"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand, paddingValue); }
    }

    /** stablehlo.iota - Generate indices along a dimension. */
    public record IotaOp(Value result, long iotaDimension, TensorType tensorResultType) implements Operation {
        @Override public String opName() { return "stablehlo.iota"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(); }
    }

    /** stablehlo.gather - Gather slices from operand. */
    public record GatherOp(
            Value result,
            Value operand,
            Value startIndices,
            List<Long> offsetDims,
            List<Long> collapsedSliceDims,
            List<Long> startIndexMap,
            long indexVectorDim,
            List<Long> sliceSizes,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.gather"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand, startIndices); }
    }

    /** stablehlo.scatter - Scatter updates into operand. */
    public record ScatterOp(
            Value result,
            Value operand,
            Value scatterIndices,
            Value updates,
            List<Long> updateWindowDims,
            List<Long> insertedWindowDims,
            List<Long> scatterDimsToOperandDims,
            long indexVectorDim,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.scatter"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand, scatterIndices, updates); }
    }

    /** stablehlo.dynamic_slice - Extract slice with dynamic indices. */
    public record DynamicSliceOp(
            Value result,
            Value operand,
            List<Value> startIndices,
            List<Long> sliceSizes,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.dynamic_slice"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() {
            var ops = new java.util.ArrayList<Value>();
            ops.add(operand);
            ops.addAll(startIndices);
            return ops;
        }
    }

    /** stablehlo.dynamic_update_slice - Update slice at dynamic indices. */
    public record DynamicUpdateSliceOp(
            Value result,
            Value operand,
            Value update,
            List<Value> startIndices,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.dynamic_update_slice"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() {
            var ops = new java.util.ArrayList<Value>();
            ops.add(operand);
            ops.add(update);
            ops.addAll(startIndices);
            return ops;
        }
    }

    // ==================== Reduction Operations ====================

    /** stablehlo.reduce_window - Windowed reduction (pooling). */
    public record ReduceWindowOp(
            Value result,
            Value operand,
            Value initValue,
            List<Long> windowDimensions,
            List<Long> windowStrides,
            List<Long> baseDilations,
            List<Long> windowDilations,
            List<Long> paddingLow,
            List<Long> paddingHigh,
            String reducer,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.reduce_window"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand, initValue); }
    }

    // ==================== Convolution and Neural Network ====================

    /** stablehlo.convolution - Convolution operation. */
    public record ConvolutionOp(
            Value result,
            Value lhs,
            Value rhs,
            List<Long> windowStrides,
            List<Long> paddingLow,
            List<Long> paddingHigh,
            List<Long> lhsDilation,
            List<Long> rhsDilation,
            long featureGroupCount,
            long batchGroupCount,
            // Dimension numbers
            long inputBatchDimension,
            long inputFeatureDimension,
            List<Long> inputSpatialDimensions,
            long kernelInputFeatureDimension,
            long kernelOutputFeatureDimension,
            List<Long> kernelSpatialDimensions,
            long outputBatchDimension,
            long outputFeatureDimension,
            List<Long> outputSpatialDimensions,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.convolution"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(lhs, rhs); }
    }

    /** stablehlo.batch_norm_training - Batch normalization (training). */
    public record BatchNormTrainingOp(
            Value output,
            Value batchMean,
            Value batchVar,
            Value operand,
            Value scale,
            Value offset,
            float epsilon,
            long featureIndex,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.batch_norm_training"; }
        @Override public List<Value> results() { return List.of(output, batchMean, batchVar); }
        @Override public List<Value> operands() { return List.of(operand, scale, offset); }
    }

    /** stablehlo.batch_norm_inference - Batch normalization (inference). */
    public record BatchNormInferenceOp(
            Value result,
            Value operand,
            Value scale,
            Value offset,
            Value mean,
            Value variance,
            float epsilon,
            long featureIndex,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.batch_norm_inference"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(operand, scale, offset, mean, variance); }
    }

    // ==================== Control Flow ====================

    /** stablehlo.if - Conditional execution. */
    public record IfOp(
            List<Value> results,
            Value pred,
            List<Operation> trueBranch,
            List<Operation> falseBranch,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.if"; }
        @Override public List<Value> operands() { return List.of(pred); }
    }

    /** stablehlo.while - While loop. */
    public record WhileOp(
            List<Value> results,
            List<Value> operands,
            List<Operation> condBody,
            List<Operation> body,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.while"; }
    }

    // ==================== Other Operations ====================

    /** stablehlo.sort - Sort tensor along dimension. */
    public record SortOp(
            Value result,
            List<Value> inputs,
            long dimension,
            boolean isStable,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.sort"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return inputs; }
    }

    /** RNG algorithm enum. */
    public enum RngAlgorithm {
        DEFAULT, THREE_FRY, PHILOX;
        public static RngAlgorithm fromString(String s) {
            return switch (s.toUpperCase()) {
                case "DEFAULT" -> DEFAULT;
                case "THREE_FRY" -> THREE_FRY;
                case "PHILOX" -> PHILOX;
                default -> DEFAULT;
            };
        }
    }

    /** stablehlo.rng - Random number generation. */
    public record RngOp(
            Value result,
            Value a,
            Value b,
            String distribution,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.rng"; }
        @Override public List<Value> results() { return List.of(result); }
        @Override public List<Value> operands() { return List.of(a, b); }
    }

    /** stablehlo.rng_bit_generator - Stateful random bit generation. */
    public record RngBitGeneratorOp(
            Value outputState,
            Value output,
            Value initialState,
            RngAlgorithm algorithm,
            TensorType tensorResultType
    ) implements Operation {
        @Override public String opName() { return "stablehlo.rng_bit_generator"; }
        @Override public List<Value> results() { return List.of(outputState, output); }
        @Override public List<Value> operands() { return List.of(initialState); }
    }

    /**
     * stablehlo.custom_call - Custom/unknown operation.
     */
    public record CustomCallOp(
            Value result,
            String callTarget,
            List<Value> inputs,
            TensorType tensorResultType
    ) implements Operation {
        @Override
        public String opName() { return "stablehlo.custom_call"; }

        @Override
        public List<Value> results() { return List.of(result); }

        @Override
        public List<Value> operands() { return inputs; }
    }

    /**
     * stablehlo.return - Function return.
     */
    public record ReturnOp(List<Value> operands) implements Operation {
        @Override
        public String opName() { return "stablehlo.return"; }

        @Override
        public List<Value> results() { return List.of(); }

        @Override
        public TensorType tensorResultType() { return null; }
    }

    // ==================== Function and Module ====================

    /**
     * Function argument.
     */
    public record Argument(String name, Type type) {
        public Value toValue() {
            return new Value(name, type);
        }
    }

    /**
     * A StableHLO function.
     */
    public record Function(
            String name,
            List<Argument> arguments,
            List<Type> resultTypes,
            List<Operation> body,
            boolean isPublic
    ) {
        public FunctionType type() {
            return new FunctionType(
                arguments.stream().map(Argument::type).toList(),
                resultTypes
            );
        }
    }

    /**
     * A StableHLO module.
     */
    public record Module(
            String name,
            List<Function> functions
    ) {
        public Optional<Function> getFunction(String name) {
            return functions.stream()
                    .filter(f -> f.name().equals(name))
                    .findFirst();
        }
    }
}
