package io.surfworks.snakeburger.stablehlo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.surfworks.snakeburger.stablehlo.StableHloAst.AbsOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AddOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AfterAllOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AllGatherOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AllReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AllToAllOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AndOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Argument;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Atan2Op;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BatchNormGradOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BatchNormInferenceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BatchNormTrainingOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BitcastConvertOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.BroadcastInDimOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CaseOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CbrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CeilOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CholeskyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ClampOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ClzOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CollectiveBroadcastOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CollectivePermuteOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CompareOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ComplexOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CompositeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConcatenateOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConstantOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConvertOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ConvolutionOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CosOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CustomCallOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DivideOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DotDimensionNumbers;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DotGeneralOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DotOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicBroadcastInDimOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicConvOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicGatherOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicIotaOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicPadOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicReshapeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicSliceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.DynamicUpdateSliceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ExpOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Expm1Op;
import io.surfworks.snakeburger.stablehlo.StableHloAst.FftOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.FloorOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Function;
import io.surfworks.snakeburger.stablehlo.StableHloAst.GatherOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.GetDimensionSizeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.GetTupleElementOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.IfOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ImagOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.InfeedOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.IotaOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.IsFiniteOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Log1pOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.LogOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.LogisticOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MapOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MaximumOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MinimumOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import io.surfworks.snakeburger.stablehlo.StableHloAst.MultiplyOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.NegateOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.NotOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.OptimizationBarrierOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.OrOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.OutfeedOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.PadOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.PartitionIdOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.PopcntOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.PowerOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RealOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RecvOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReducePrecisionOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceScatterOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceWindowOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RemainderOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReplicaIdOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReshapeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReturnOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReverseOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RngBitGeneratorOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RngOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RoundNearestAfzOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RoundNearestEvenOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RsqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ScalarType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ScatterOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SelectAndScatterOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SelectOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SendOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ShiftLeftOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ShiftRightArithmeticOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ShiftRightLogicalOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SignOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SinOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SliceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SortOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SqrtOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SubtractOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TanhOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TanOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TensorType;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TransposeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TriangularSolveOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.TupleOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Type;
import io.surfworks.snakeburger.stablehlo.StableHloAst.UniformDequantizeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.UniformQuantizeOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Value;
import io.surfworks.snakeburger.stablehlo.StableHloAst.WhileOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.XorOp;

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
            // Binary elementwise ops
            case AddOp a -> emitBinaryOp(a, "add", a.lhs(), a.rhs(), valueTypes);
            case SubtractOp s -> emitBinaryOp(s, "sub", s.lhs(), s.rhs(), valueTypes);
            case MultiplyOp m -> emitBinaryOp(m, "mul", m.lhs(), m.rhs(), valueTypes);
            case DivideOp d -> emitBinaryOp(d, "div", d.lhs(), d.rhs(), valueTypes);
            case MaximumOp m -> emitBinaryOp(m, "max", m.lhs(), m.rhs(), valueTypes);
            case MinimumOp m -> emitBinaryOp(m, "min", m.lhs(), m.rhs(), valueTypes);
            case PowerOp p -> emitBinaryOp(p, "pow", p.lhs(), p.rhs(), valueTypes);
            case RemainderOp r -> emitBinaryOp(r, "rem", r.lhs(), r.rhs(), valueTypes);
            case Atan2Op a -> emitBinaryOp(a, "atan2", a.lhs(), a.rhs(), valueTypes);
            // Binary logical/bitwise ops
            case AndOp a -> emitBinaryOp(a, "and", a.lhs(), a.rhs(), valueTypes);
            case OrOp o -> emitBinaryOp(o, "or", o.lhs(), o.rhs(), valueTypes);
            case XorOp x -> emitBinaryOp(x, "xor", x.lhs(), x.rhs(), valueTypes);
            case ShiftLeftOp s -> emitBinaryOp(s, "shl", s.lhs(), s.rhs(), valueTypes);
            case ShiftRightArithmeticOp s -> emitBinaryOp(s, "shra", s.lhs(), s.rhs(), valueTypes);
            case ShiftRightLogicalOp s -> emitBinaryOp(s, "shrl", s.lhs(), s.rhs(), valueTypes);
            // Unary elementwise ops
            case NegateOp n -> emitUnaryOp(n, "neg", n.operand(), valueTypes);
            case AbsOp a -> emitUnaryOp(a, "abs", a.operand(), valueTypes);
            case ExpOp e -> emitUnaryOp(e, "exp", e.operand(), valueTypes);
            case LogOp l -> emitUnaryOp(l, "log", l.operand(), valueTypes);
            case TanhOp t -> emitUnaryOp(t, "tanh", t.operand(), valueTypes);
            case SqrtOp s -> emitUnaryOp(s, "sqrt", s.operand(), valueTypes);
            case RsqrtOp r -> emitUnaryOp(r, "rsqrt", r.operand(), valueTypes);
            case SinOp s -> emitUnaryOp(s, "sin", s.operand(), valueTypes);
            case CosOp c -> emitUnaryOp(c, "cos", c.operand(), valueTypes);
            case TanOp t -> emitUnaryOp(t, "tan", t.operand(), valueTypes);
            case CeilOp c -> emitUnaryOp(c, "ceil", c.operand(), valueTypes);
            case FloorOp f -> emitUnaryOp(f, "floor", f.operand(), valueTypes);
            case SignOp s -> emitUnaryOp(s, "sign", s.operand(), valueTypes);
            case LogisticOp l -> emitUnaryOp(l, "sigmoid", l.operand(), valueTypes);
            case Expm1Op e -> emitUnaryOp(e, "expm1", e.operand(), valueTypes);
            case Log1pOp l -> emitUnaryOp(l, "log1p", l.operand(), valueTypes);
            case CbrtOp c -> emitUnaryOp(c, "cbrt", c.operand(), valueTypes);
            case IsFiniteOp i -> emitIsFinite(i, valueTypes);
            case PopcntOp p -> emitUnaryOp(p, "popcnt", p.operand(), valueTypes);
            case ClzOp c -> emitUnaryOp(c, "clz", c.operand(), valueTypes);
            case RoundNearestEvenOp r -> emitUnaryOp(r, "round_even", r.operand(), valueTypes);
            case RoundNearestAfzOp r -> emitUnaryOp(r, "round_afz", r.operand(), valueTypes);
            case NotOp n -> emitUnaryOp(n, "not", n.operand(), valueTypes);
            // Shape ops
            case ReshapeOp r -> emitReshape(r, valueTypes);
            case TransposeOp t -> emitTranspose(t, valueTypes);
            case BroadcastInDimOp b -> emitBroadcast(b, valueTypes);
            case ConcatenateOp c -> emitConcatenate(c, valueTypes);
            case SliceOp s -> emitSlice(s, valueTypes);
            case ReverseOp r -> emitReverse(r, valueTypes);
            case PadOp p -> emitPad(p, valueTypes);
            case IotaOp i -> emitIota(i, valueTypes);
            case GatherOp g -> emitGather(g, valueTypes);
            case ScatterOp s -> emitScatter(s, valueTypes);
            case DynamicSliceOp d -> emitDynamicSlice(d, valueTypes);
            case DynamicUpdateSliceOp d -> emitDynamicUpdateSlice(d, valueTypes);
            // Conditional ops
            case CompareOp c -> emitCompare(c, valueTypes);
            case SelectOp s -> emitSelect(s, valueTypes);
            case ClampOp c -> emitClamp(c, valueTypes);
            // Type conversion
            case ConvertOp c -> emitConvert(c, valueTypes);
            case BitcastConvertOp b -> emitBitcastConvert(b, valueTypes);
            // Reduction
            case ReduceOp r -> emitReduce(r, valueTypes);
            case ReduceWindowOp r -> emitReduceWindow(r, valueTypes);
            // Convolution and neural network
            case ConvolutionOp c -> emitConvolution(c, valueTypes);
            case BatchNormTrainingOp b -> emitBatchNormTraining(b, valueTypes);
            case BatchNormInferenceOp b -> emitBatchNormInference(b, valueTypes);
            // Control flow
            case IfOp i -> emitIf(i, valueTypes);
            case WhileOp w -> emitWhile(w, valueTypes);
            // Linear algebra
            case DotOp d -> emitDot(d, valueTypes);
            case CholeskyOp c -> emitCholesky(c, valueTypes);
            case TriangularSolveOp t -> emitTriangularSolve(t, valueTypes);
            // Complex numbers
            case RealOp r -> emitReal(r, valueTypes);
            case ImagOp i -> emitImag(i, valueTypes);
            case ComplexOp c -> emitComplex(c, valueTypes);
            // Signal processing
            case FftOp f -> emitFft(f, valueTypes);
            // Other
            case SortOp s -> emitSort(s, valueTypes);
            case RngOp r -> emitRng(r, valueTypes);
            case RngBitGeneratorOp r -> emitRngBitGenerator(r, valueTypes);
            case CustomCallOp c -> emitCustomCall(c, valueTypes);
            case ReturnOp r -> emitReturn(r);
            // Dynamic shape operations
            case DynamicBroadcastInDimOp d -> emitDynamicBroadcastInDim(d, valueTypes);
            case DynamicGatherOp d -> emitDynamicGather(d, valueTypes);
            case DynamicIotaOp d -> emitDynamicIota(d, valueTypes);
            case DynamicPadOp d -> emitDynamicPad(d, valueTypes);
            case DynamicReshapeOp d -> emitDynamicReshape(d, valueTypes);
            case DynamicConvOp d -> emitDynamicConv(d, valueTypes);
            case GetDimensionSizeOp g -> emitGetDimensionSize(g, valueTypes);
            // Quantization operations
            case UniformQuantizeOp u -> emitUniformQuantize(u, valueTypes);
            case UniformDequantizeOp u -> emitUniformDequantize(u, valueTypes);
            // Additional reduction operations
            case ReducePrecisionOp r -> emitReducePrecision(r, valueTypes);
            case SelectAndScatterOp s -> emitSelectAndScatter(s, valueTypes);
            // Additional neural network operations
            case BatchNormGradOp b -> emitBatchNormGrad(b, valueTypes);
            // Additional control flow operations
            case CaseOp c -> emitCase(c, valueTypes);
            case MapOp m -> emitMap(m, valueTypes);
            // Distributed/collective operations
            case AfterAllOp a -> emitAfterAll(a, valueTypes);
            case AllGatherOp a -> emitAllGather(a, valueTypes);
            case AllReduceOp a -> emitAllReduce(a, valueTypes);
            case AllToAllOp a -> emitAllToAll(a, valueTypes);
            case CollectiveBroadcastOp c -> emitCollectiveBroadcast(c, valueTypes);
            case CollectivePermuteOp c -> emitCollectivePermute(c, valueTypes);
            case PartitionIdOp p -> emitPartitionId(p, valueTypes);
            case ReduceScatterOp r -> emitReduceScatter(r, valueTypes);
            case ReplicaIdOp r -> emitReplicaId(r, valueTypes);
            // Communication operations
            case InfeedOp i -> emitInfeed(i, valueTypes);
            case OutfeedOp o -> emitOutfeed(o, valueTypes);
            case RecvOp r -> emitRecv(r, valueTypes);
            case SendOp s -> emitSend(s, valueTypes);
            // Tuple operations
            case TupleOp t -> emitTuple(t, valueTypes);
            case GetTupleElementOp g -> emitGetTupleElement(g, valueTypes);
            // Other operations
            case OptimizationBarrierOp o -> emitOptimizationBarrier(o, valueTypes);
            case CompositeOp c -> emitComposite(c, valueTypes);
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

    private String emitConcatenate(ConcatenateOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        StringBuilder inputs = new StringBuilder();
        for (int i = 0; i < op.inputs().size(); i++) {
            if (i > 0) inputs.append(", ");
            inputs.append("%").append(op.inputs().get(i).name());
        }

        return String.format("%%%-8s = concatenate %s dim=%d  // -> %s",
                resultName, inputs, op.dimension(), shapeToString(op.tensorResultType()));
    }

    private String emitSlice(SliceOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = slice %%%s [%s:%s:%s]  // -> %s",
                resultName, op.operand().name(),
                op.startIndices(), op.limitIndices(), op.strides(),
                shapeToString(op.tensorResultType()));
    }

    private String emitCompare(CompareOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = compare<%s> %%%s, %%%s  // -> %s",
                resultName, op.direction(), op.lhs().name(), op.rhs().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitSelect(SelectOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = select %%%s ? %%%s : %%%s  // -> %s",
                resultName, op.pred().name(), op.onTrue().name(), op.onFalse().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitClamp(ClampOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = clamp %%%s, %%%s, %%%s  // -> %s",
                resultName, op.min().name(), op.operand().name(), op.max().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitConvert(ConvertOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = convert %%%s  // -> %s",
                resultName, op.operand().name(), shapeToString(op.tensorResultType()));
    }

    private String emitBitcastConvert(BitcastConvertOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = bitcast_convert %%%s  // -> %s",
                resultName, op.operand().name(), shapeToString(op.tensorResultType()));
    }

    private String emitIsFinite(IsFiniteOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = is_finite %%%s  // -> %s",
                resultName, op.operand().name(), shapeToString(op.tensorResultType()));
    }

    private String emitReverse(ReverseOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = reverse %%%s dims=%s  // -> %s",
                resultName, op.operand().name(), op.dimensions(), shapeToString(op.tensorResultType()));
    }

    private String emitPad(PadOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = pad %%%s with %%%s [low=%s, high=%s, interior=%s]  // -> %s",
                resultName, op.operand().name(), op.paddingValue().name(),
                op.edgePaddingLow(), op.edgePaddingHigh(), op.interiorPadding(),
                shapeToString(op.tensorResultType()));
    }

    private String emitIota(IotaOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = iota dim=%d  // -> %s",
                resultName, op.iotaDimension(), shapeToString(op.tensorResultType()));
    }

    private String emitGather(GatherOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = gather %%%s[%%%s]  // -> %s",
                resultName, op.operand().name(), op.startIndices().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitScatter(ScatterOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = scatter %%%s[%%%s] = %%%s  // -> %s",
                resultName, op.operand().name(), op.scatterIndices().name(),
                op.updates().name(), shapeToString(op.tensorResultType()));
    }

    private String emitDynamicSlice(DynamicSliceOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        StringBuilder indices = new StringBuilder();
        for (int i = 0; i < op.startIndices().size(); i++) {
            if (i > 0) indices.append(", ");
            indices.append("%").append(op.startIndices().get(i).name());
        }

        return String.format("%%%-8s = dynamic_slice %%%s[%s] sizes=%s  // -> %s",
                resultName, op.operand().name(), indices, op.sliceSizes(),
                shapeToString(op.tensorResultType()));
    }

    private String emitDynamicUpdateSlice(DynamicUpdateSliceOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        StringBuilder indices = new StringBuilder();
        for (int i = 0; i < op.startIndices().size(); i++) {
            if (i > 0) indices.append(", ");
            indices.append("%").append(op.startIndices().get(i).name());
        }

        return String.format("%%%-8s = dynamic_update_slice %%%s[%s] = %%%s  // -> %s",
                resultName, op.operand().name(), indices, op.update().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitReduceWindow(ReduceWindowOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = reduce_window<%s> %%%s window=%s strides=%s  // -> %s",
                resultName, op.reducer(), op.operand().name(),
                op.windowDimensions(), op.windowStrides(),
                shapeToString(op.tensorResultType()));
    }

    private String emitConvolution(ConvolutionOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = convolution %%%s, %%%s strides=%s  // -> %s",
                resultName, op.lhs().name(), op.rhs().name(),
                op.windowStrides(), shapeToString(op.tensorResultType()));
    }

    private String emitBatchNormTraining(BatchNormTrainingOp op, Map<String, String> valueTypes) {
        String resultName = op.output().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);
        valueTypes.put(op.batchMean().name(), javaType);
        valueTypes.put(op.batchVar().name(), javaType);

        return String.format("%%%-8s = batch_norm_training %%%s eps=%.6f feature=%d  // -> %s",
                resultName, op.operand().name(), op.epsilon(), op.featureIndex(),
                shapeToString(op.tensorResultType()));
    }

    private String emitBatchNormInference(BatchNormInferenceOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = batch_norm_inference %%%s eps=%.6f feature=%d  // -> %s",
                resultName, op.operand().name(), op.epsilon(), op.featureIndex(),
                shapeToString(op.tensorResultType()));
    }

    private String emitIf(IfOp op, Map<String, String> valueTypes) {
        String resultName = op.results().isEmpty() ? "void" : op.results().get(0).name();
        if (!op.results().isEmpty()) {
            String javaType = typeToJavaType(op.tensorResultType());
            valueTypes.put(resultName, javaType);
        }

        return String.format("%%%-8s = if %%%s [true: %d ops, false: %d ops]",
                resultName, op.pred().name(),
                op.trueBranch().size(), op.falseBranch().size());
    }

    private String emitWhile(WhileOp op, Map<String, String> valueTypes) {
        String resultName = op.results().isEmpty() ? "void" : op.results().get(0).name();
        if (!op.results().isEmpty()) {
            String javaType = typeToJavaType(op.tensorResultType());
            valueTypes.put(resultName, javaType);
        }

        return String.format("%%%-8s = while [cond: %d ops, body: %d ops]",
                resultName, op.condBody().size(), op.body().size());
    }

    private String emitSort(SortOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        StringBuilder inputs = new StringBuilder();
        for (int i = 0; i < op.inputs().size(); i++) {
            if (i > 0) inputs.append(", ");
            inputs.append("%").append(op.inputs().get(i).name());
        }

        return String.format("%%%-8s = sort %s dim=%d stable=%s  // -> %s",
                resultName, inputs, op.dimension(), op.isStable(),
                shapeToString(op.tensorResultType()));
    }

    private String emitRng(RngOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = rng<%s> %%%s, %%%s  // -> %s",
                resultName, op.distribution(), op.a().name(), op.b().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitRngBitGenerator(RngBitGeneratorOp op, Map<String, String> valueTypes) {
        String resultName = op.output().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);
        valueTypes.put(op.outputState().name(), javaType);

        return String.format("%%%-8s = rng_bit_generator<%s> %%%s  // -> %s",
                resultName, op.algorithm(), op.initialState().name(),
                shapeToString(op.tensorResultType()));
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

    // ==================== Linear Algebra ====================

    private String emitDot(DotOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = dot %%%s, %%%s  // -> %s",
                resultName, op.lhs().name(), op.rhs().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitCholesky(CholeskyOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = cholesky %%%s lower=%s  // -> %s",
                resultName, op.operand().name(), op.lower(),
                shapeToString(op.tensorResultType()));
    }

    private String emitTriangularSolve(TriangularSolveOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = triangular_solve %%%s, %%%s left=%s lower=%s transpose=%s  // -> %s",
                resultName, op.a().name(), op.b().name(),
                op.leftSide(), op.lower(), op.transposeA(),
                shapeToString(op.tensorResultType()));
    }

    // ==================== Complex Numbers ====================

    private String emitReal(RealOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = real %%%s  // -> %s",
                resultName, op.operand().name(), shapeToString(op.tensorResultType()));
    }

    private String emitImag(ImagOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = imag %%%s  // -> %s",
                resultName, op.operand().name(), shapeToString(op.tensorResultType()));
    }

    private String emitComplex(ComplexOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = complex %%%s, %%%s  // -> %s",
                resultName, op.real().name(), op.imag().name(),
                shapeToString(op.tensorResultType()));
    }

    // ==================== Signal Processing ====================

    private String emitFft(FftOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = fft<%s> %%%s length=%s  // -> %s",
                resultName, op.fftType(), op.operand().name(), op.fftLength(),
                shapeToString(op.tensorResultType()));
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

    // ==================== Dynamic Shape Operations ====================

    private String emitDynamicBroadcastInDim(DynamicBroadcastInDimOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = dynamic_broadcast_in_dim %%%s, %%%s dims=%s  // -> %s",
                resultName, op.operand().name(), op.outputDimensions().name(),
                op.broadcastDimensions(), shapeToString(op.tensorResultType()));
    }

    private String emitDynamicGather(DynamicGatherOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = dynamic_gather %%%s[%%%s, %%%s]  // -> %s",
                resultName, op.operand().name(), op.startIndices().name(),
                op.sliceSizes().name(), shapeToString(op.tensorResultType()));
    }

    private String emitDynamicIota(DynamicIotaOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = dynamic_iota %%%s dim=%d  // -> %s",
                resultName, op.outputShape().name(), op.iotaDimension(),
                shapeToString(op.tensorResultType()));
    }

    private String emitDynamicPad(DynamicPadOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = dynamic_pad %%%s with %%%s  // -> %s",
                resultName, op.operand().name(), op.paddingValue().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitDynamicReshape(DynamicReshapeOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = dynamic_reshape %%%s, %%%s  // -> %s",
                resultName, op.operand().name(), op.outputShape().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitDynamicConv(DynamicConvOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = dynamic_conv %%%s, %%%s, %%%s  // -> %s",
                resultName, op.lhs().name(), op.rhs().name(), op.padding().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitGetDimensionSize(GetDimensionSizeOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = get_dimension_size %%%s dim=%d  // -> %s",
                resultName, op.operand().name(), op.dimension(),
                shapeToString(op.tensorResultType()));
    }

    // ==================== Quantization Operations ====================

    private String emitUniformQuantize(UniformQuantizeOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = uniform_quantize %%%s  // -> %s",
                resultName, op.operand().name(), shapeToString(op.tensorResultType()));
    }

    private String emitUniformDequantize(UniformDequantizeOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = uniform_dequantize %%%s  // -> %s",
                resultName, op.operand().name(), shapeToString(op.tensorResultType()));
    }

    // ==================== Additional Reduction Operations ====================

    private String emitReducePrecision(ReducePrecisionOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = reduce_precision %%%s exp=%d mant=%d  // -> %s",
                resultName, op.operand().name(), op.exponentBits(), op.mantissaBits(),
                shapeToString(op.tensorResultType()));
    }

    private String emitSelectAndScatter(SelectAndScatterOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = select_and_scatter<%s,%s> %%%s, %%%s, %%%s  // -> %s",
                resultName, op.selectFn(), op.scatterFn(),
                op.operand().name(), op.source().name(), op.initValue().name(),
                shapeToString(op.tensorResultType()));
    }

    // ==================== Additional Neural Network Operations ====================

    private String emitBatchNormGrad(BatchNormGradOp op, Map<String, String> valueTypes) {
        String resultName = op.gradOperand().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = batch_norm_grad %%%s eps=%.6f feature=%d  // -> 3 outputs",
                resultName, op.operand().name(), op.epsilon(), op.featureIndex());
    }

    // ==================== Additional Control Flow Operations ====================

    private String emitCase(CaseOp op, Map<String, String> valueTypes) {
        String resultName = op.results().isEmpty() ? "void" : op.results().get(0).name();
        if (!op.results().isEmpty()) {
            String javaType = typeToJavaType(op.tensorResultType());
            valueTypes.put(resultName, javaType);
        }

        return String.format("%%%-8s = case %%%s [%d branches]",
                resultName, op.index().name(), op.branches().size());
    }

    private String emitMap(MapOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        StringBuilder inputs = new StringBuilder();
        for (int i = 0; i < op.inputs().size(); i++) {
            if (i > 0) inputs.append(", ");
            inputs.append("%").append(op.inputs().get(i).name());
        }

        return String.format("%%%-8s = map %s dims=%s  // -> %s",
                resultName, inputs, op.dimensions(), shapeToString(op.tensorResultType()));
    }

    // ==================== Distributed/Collective Operations ====================

    private String emitAfterAll(AfterAllOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        StringBuilder inputs = new StringBuilder();
        for (int i = 0; i < op.inputs().size(); i++) {
            if (i > 0) inputs.append(", ");
            inputs.append("%").append(op.inputs().get(i).name());
        }

        return String.format("%%%-8s = after_all %s  // barrier",
                resultName, inputs);
    }

    private String emitAllGather(AllGatherOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = all_gather %%%s dim=%d  // -> %s",
                resultName, op.operand().name(), op.allGatherDim(),
                shapeToString(op.tensorResultType()));
    }

    private String emitAllReduce(AllReduceOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = all_reduce<%s> %%%s  // -> %s",
                resultName, op.reducer(), op.operand().name(),
                shapeToString(op.tensorResultType()));
    }

    private String emitAllToAll(AllToAllOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = all_to_all %%%s split=%d concat=%d  // -> %s",
                resultName, op.operand().name(), op.splitDimension(), op.concatDimension(),
                shapeToString(op.tensorResultType()));
    }

    private String emitCollectiveBroadcast(CollectiveBroadcastOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = collective_broadcast %%%s  // -> %s",
                resultName, op.operand().name(), shapeToString(op.tensorResultType()));
    }

    private String emitCollectivePermute(CollectivePermuteOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = collective_permute %%%s  // -> %s",
                resultName, op.operand().name(), shapeToString(op.tensorResultType()));
    }

    private String emitPartitionId(PartitionIdOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = partition_id  // -> %s",
                resultName, shapeToString(op.tensorResultType()));
    }

    private String emitReduceScatter(ReduceScatterOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = reduce_scatter<%s> %%%s dim=%d  // -> %s",
                resultName, op.reducer(), op.operand().name(), op.scatterDimension(),
                shapeToString(op.tensorResultType()));
    }

    private String emitReplicaId(ReplicaIdOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = replica_id  // -> %s",
                resultName, shapeToString(op.tensorResultType()));
    }

    // ==================== Communication Operations ====================

    private String emitInfeed(InfeedOp op, Map<String, String> valueTypes) {
        String resultName = op.results().isEmpty() ? "void" : op.results().get(0).name();
        if (!op.results().isEmpty()) {
            String javaType = typeToJavaType(op.tensorResultType());
            valueTypes.put(resultName, javaType);
        }

        return String.format("%%%-8s = infeed %%%s config=\"%s\"  // host -> device",
                resultName, op.token().name(), op.infeedConfig());
    }

    private String emitOutfeed(OutfeedOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = outfeed %%%s config=\"%s\"  // device -> host",
                resultName, op.token().name(), op.outfeedConfig());
    }

    private String emitRecv(RecvOp op, Map<String, String> valueTypes) {
        String resultName = op.results().isEmpty() ? "void" : op.results().get(0).name();
        if (!op.results().isEmpty()) {
            String javaType = typeToJavaType(op.tensorResultType());
            valueTypes.put(resultName, javaType);
        }

        return String.format("%%%-8s = recv %%%s channel=%d  // -> %s",
                resultName, op.token().name(), op.channelId(),
                shapeToString(op.tensorResultType()));
    }

    private String emitSend(SendOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = send %%%s channel=%d  // token",
                resultName, op.token().name(), op.channelId());
    }

    // ==================== Tuple Operations ====================

    private String emitTuple(TupleOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        StringBuilder inputs = new StringBuilder();
        for (int i = 0; i < op.inputs().size(); i++) {
            if (i > 0) inputs.append(", ");
            inputs.append("%").append(op.inputs().get(i).name());
        }

        return String.format("%%%-8s = tuple %s  // -> tuple<%d>",
                resultName, inputs, op.inputs().size());
    }

    private String emitGetTupleElement(GetTupleElementOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        return String.format("%%%-8s = get_tuple_element %%%s index=%d  // -> %s",
                resultName, op.operand().name(), op.index(),
                shapeToString(op.tensorResultType()));
    }

    // ==================== Other Operations ====================

    private String emitOptimizationBarrier(OptimizationBarrierOp op, Map<String, String> valueTypes) {
        String resultName = op.results().isEmpty() ? "void" : op.results().get(0).name();
        if (!op.results().isEmpty()) {
            String javaType = typeToJavaType(op.tensorResultType());
            valueTypes.put(resultName, javaType);
        }

        StringBuilder inputs = new StringBuilder();
        for (int i = 0; i < op.operands().size(); i++) {
            if (i > 0) inputs.append(", ");
            inputs.append("%").append(op.operands().get(i).name());
        }

        return String.format("%%%-8s = optimization_barrier %s  // pass-through",
                resultName, inputs);
    }

    private String emitComposite(CompositeOp op, Map<String, String> valueTypes) {
        String resultName = op.result().name();
        String javaType = typeToJavaType(op.tensorResultType());
        valueTypes.put(resultName, javaType);

        StringBuilder inputs = new StringBuilder();
        for (int i = 0; i < op.inputs().size(); i++) {
            if (i > 0) inputs.append(", ");
            inputs.append("%").append(op.inputs().get(i).name());
        }

        return String.format("%%%-8s = composite \"%s\" %s version=%d  // -> %s",
                resultName, op.name(), inputs, op.version(),
                shapeToString(op.tensorResultType()));
    }
}
