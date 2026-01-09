package io.surfworks.snakeburger.stablehlo;

import io.surfworks.snakeburger.stablehlo.StableHloAst.*;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;

import java.util.*;

/**
 * Emits Babylon-compatible representation from StableHLO AST.
 *
 * This generates a textual representation of the computation graph
 * that follows Babylon's Op tree format. The output can be used for:
 * - Debugging and verification
 * - Future integration with Babylon's code reflection API
 * - Generation of Java code that implements the computation
 *
 * The actual Babylon FuncOp construction requires the incubator API
 * which is still evolving. This emitter focuses on generating a
 * readable intermediate representation.
 */
public final class StableHloToBabylon {

    /**
     * Result of emitting a StableHLO module.
     */
    public record EmitResult(
            Module stableHloModule,
            String babylonText,
            List<String> operations
    ) {}

    private final StringBuilder sb = new StringBuilder();
    private final List<String> operations = new ArrayList<>();
    private int indent = 0;

    /**
     * Emits a StableHLO module to a Babylon-compatible text representation.
     */
    public EmitResult emit(Module module) {
        if (module.functions().isEmpty()) {
            throw new StableHloParseException("Module has no functions");
        }

        sb.setLength(0);
        operations.clear();

        emitLine("// Babylon Op Tree for StableHLO module: @%s", module.name());
        emitLine("// Generated from StableHLO MLIR");
        emitLine("");

        for (Function function : module.functions()) {
            emitFunction(function);
        }

        return new EmitResult(module, sb.toString(), new ArrayList<>(operations));
    }

    private void emitFunction(Function function) {
        // Emit function signature
        StringBuilder sig = new StringBuilder();
        sig.append("func @").append(function.name()).append("(");
        for (int i = 0; i < function.arguments().size(); i++) {
            if (i > 0) sig.append(", ");
            Argument arg = function.arguments().get(i);
            sig.append("%").append(arg.name()).append(": ").append(typeToString(arg.type()));
        }
        sig.append(") -> ");
        if (function.resultTypes().size() == 1) {
            sig.append(typeToString(function.resultTypes().get(0)));
        } else {
            sig.append("(");
            for (int i = 0; i < function.resultTypes().size(); i++) {
                if (i > 0) sig.append(", ");
                sig.append(typeToString(function.resultTypes().get(i)));
            }
            sig.append(")");
        }

        emitLine("%s {", sig);
        indent++;

        // Map from SSA names to their Java types
        Map<String, String> valueTypes = new HashMap<>();
        for (Argument arg : function.arguments()) {
            valueTypes.put(arg.name(), typeToJavaType(arg.type()));
        }

        // Emit operations
        for (Operation op : function.body()) {
            emitOperation(op, valueTypes);
        }

        indent--;
        emitLine("}");
        emitLine("");
    }

    private void emitOperation(Operation op, Map<String, String> valueTypes) {
        String opText = switch (op) {
            case ConstantOp c -> emitConstant(c, valueTypes);
            case DotGeneralOp d -> emitDotGeneral(d, valueTypes);
            case AddOp a -> emitBinaryOp(a, "add", a.lhs(), a.rhs(), valueTypes);
            case MultiplyOp m -> emitBinaryOp(m, "mul", m.lhs(), m.rhs(), valueTypes);
            case DivideOp d -> emitBinaryOp(d, "div", d.lhs(), d.rhs(), valueTypes);
            case MaximumOp m -> emitBinaryOp(m, "max", m.lhs(), m.rhs(), valueTypes);
            case NegateOp n -> emitUnaryOp(n, "neg", n.operand(), valueTypes);
            case ReshapeOp r -> emitReshape(r, valueTypes);
            case TransposeOp t -> emitTranspose(t, valueTypes);
            case BroadcastInDimOp b -> emitBroadcast(b, valueTypes);
            case ReduceOp r -> emitReduce(r, valueTypes);
            case CustomCallOp c -> emitCustomCall(c, valueTypes);
            case ReturnOp r -> emitReturn(r);
        };

        emitLine("%s", opText);
        operations.add(opText);
    }

    private String emitConstant(ConstantOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        Object value = op.value().value();
        String valueStr = value instanceof Double ? String.format("%.1f", value) : value.toString();

        return String.format("%%%-8s = constant %s<%s>  // %s",
                resultName, valueStr, shapeToString(op.tensorResultType()), javaType);
    }

    private String emitDotGeneral(DotGeneralOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        DotDimensionNumbers dims = op.dimensionNumbers();
        return String.format("%%%-8s = dot_general %%%s, %%%s [contract: %s×%s, batch: %s×%s]  // %s -> %s",
                resultName,
                op.lhs().name(),
                op.rhs().name(),
                dims.lhsContractingDimensions(),
                dims.rhsContractingDimensions(),
                dims.lhsBatchingDimensions(),
                dims.rhsBatchingDimensions(),
                typeToString(op.lhs().type()) + ", " + typeToString(op.rhs().type()),
                shapeToString(op.tensorResultType()));
    }

    private String emitBinaryOp(Operation op, String opName, Value lhs, Value rhs,
                                Map<String, String> valueTypes) {
        String resultName = op.results().get(0).name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = %s %%%s, %%%s  // %s",
                resultName, opName, lhs.name(), rhs.name(), shapeToString((TensorType) op.tensorResultType()));
    }

    private String emitUnaryOp(Operation op, String opName, Value operand,
                               Map<String, String> valueTypes) {
        String resultName = op.results().get(0).name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = %s %%%s  // %s",
                resultName, opName, operand.name(), shapeToString((TensorType) op.tensorResultType()));
    }

    private String emitReshape(ReshapeOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = reshape %%%s to %s",
                resultName, op.operand().name(), shapeToString(op.tensorResultType()));
    }

    private String emitTranspose(TransposeOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = transpose %%%s dims=%s  // -> %s",
                resultName, op.operand().name(), op.permutation(), shapeToString(op.tensorResultType()));
    }

    private String emitBroadcast(BroadcastInDimOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = broadcast %%%s dims=%s -> %s",
                resultName, op.operand().name(), op.broadcastDimensions(), shapeToString(op.tensorResultType()));
    }

    private String emitReduce(ReduceOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = reduce<%s> %%%s over dims=%s  // -> %s",
                resultName, op.reducer(), op.operand().name(), op.dimensions(), shapeToString(op.tensorResultType()));
    }

    private String emitCustomCall(CustomCallOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        StringBuilder args = new StringBuilder();
        for (int i = 0; i < op.inputs().size(); i++) {
            if (i > 0) args.append(", ");
            args.append("%").append(op.inputs().get(i).name());
        }

        return String.format("%%%-8s = call @%s(%s)  // -> %s",
                resultName, op.callTarget(), args, shapeToString(op.tensorResultType()));
    }

    private String emitReturn(ReturnOp op) {
        if (op.operands().isEmpty()) {
            return "return";
        }
        StringBuilder args = new StringBuilder("return ");
        for (int i = 0; i < op.operands().size(); i++) {
            if (i > 0) args.append(", ");
            args.append("%").append(op.operands().get(i).name());
        }
        return args.toString();
    }

    private String typeToString(Type type) {
        return type.toMlirString();
    }

    private String shapeToString(TensorType type) {
        if (type.shape().isEmpty()) {
            return type.elementType().name();
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < type.shape().size(); i++) {
            if (i > 0) sb.append("x");
            sb.append(type.shape().get(i));
        }
        sb.append("x").append(type.elementType().name());
        return sb.toString();
    }

    private String typeToJavaType(Type type) {
        if (type instanceof TensorType t) {
            String elemType = scalarToJavaType(t.elementType());
            return "Tensor<" + elemType + ">" + shapeAnnotation(t);
        } else if (type instanceof ScalarType s) {
            return scalarToJavaType(s);
        }
        return "Object";
    }

    private String scalarToJavaType(ScalarType type) {
        return switch (type.name()) {
            case "f16", "f32" -> "Float";
            case "f64" -> "Double";
            case "bf16" -> "BFloat16";
            case "i1" -> "Boolean";
            case "i8" -> "Byte";
            case "i16" -> "Short";
            case "i32" -> "Integer";
            case "i64" -> "Long";
            default -> "Object";
        };
    }

    private String shapeAnnotation(TensorType type) {
        if (type.shape().isEmpty()) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < type.shape().size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(type.shape().get(i));
        }
        sb.append("]");
        return sb.toString();
    }

    private void emitLine(String format, Object... args) {
        for (int i = 0; i < indent; i++) {
            sb.append("  ");
        }
        sb.append(String.format(format, args));
        sb.append("\n");
    }
}
