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
            DotGeneralOp, ConstantOp, MaximumOp, AddOp, MultiplyOp, DivideOp,
            NegateOp, ReshapeOp, TransposeOp, ReturnOp, BroadcastInDimOp,
            ReduceOp, CustomCallOp {

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
