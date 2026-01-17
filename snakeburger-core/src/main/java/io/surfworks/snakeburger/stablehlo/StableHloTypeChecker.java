package io.surfworks.snakeburger.stablehlo;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
 * Type checker for StableHLO AST.
 *
 * Performs semantic validation including:
 * - Value availability: operands must be defined before use
 * - Shape consistency: dimension sizes must match for operations
 * - Type consistency: element types must match for binary operations
 */
public final class StableHloTypeChecker {

    private final List<String> errors = new ArrayList<>();
    private final Set<String> definedValues = new HashSet<>();

    public StableHloTypeChecker() {}

    /**
     * Validates a module and returns a list of errors.
     * Returns empty list if validation passes.
     */
    public List<String> validate(Module module) {
        errors.clear();
        for (Function function : module.functions()) {
            validateFunction(function);
        }
        return new ArrayList<>(errors);
    }

    /**
     * Validates a module and throws if any errors are found.
     */
    public void check(Module module) {
        List<String> validationErrors = validate(module);
        if (!validationErrors.isEmpty()) {
            StringBuilder sb = new StringBuilder("StableHLO validation failed:\n");
            for (String error : validationErrors) {
                sb.append("  - ").append(error).append("\n");
            }
            throw new StableHloParseException(sb.toString());
        }
    }

    private void validateFunction(Function function) {
        definedValues.clear();

        // Register function arguments as defined values
        for (Argument arg : function.arguments()) {
            definedValues.add(arg.name());
        }

        // Validate each operation
        for (Operation op : function.body()) {
            validateOperation(op, function);
        }

        // Validate return op matches function result types
        if (!function.body().isEmpty()) {
            Operation lastOp = function.body().get(function.body().size() - 1);
            if (lastOp instanceof ReturnOp returnOp) {
                validateReturnTypes(returnOp, function);
            } else {
                error("Function '%s' does not end with a return operation", function.name());
            }
        }
    }

    private void validateOperation(Operation op, Function function) {
        // Check all operands are defined
        for (Value operand : op.operands()) {
            if (!definedValues.contains(operand.name())) {
                error("Undefined value '%%%s' used in %s", operand.name(), op.opName());
            }
        }

        // Register results as defined
        for (Value result : op.results()) {
            if (definedValues.contains(result.name())) {
                error("Value '%%%s' redefined in %s", result.name(), op.opName());
            }
            definedValues.add(result.name());
        }

        // Operation-specific validation
        switch (op) {
            case DotGeneralOp dotOp -> validateDotGeneral(dotOp);
            // Binary arithmetic ops
            case AddOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case SubtractOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case MultiplyOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case DivideOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case MaximumOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case MinimumOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case PowerOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case RemainderOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case Atan2Op o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            // Binary logical/bitwise ops
            case AndOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case OrOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case XorOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case ShiftLeftOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case ShiftRightArithmeticOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case ShiftRightLogicalOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            // Unary math ops
            case NegateOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case AbsOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case ExpOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case LogOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case TanhOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case SqrtOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case RsqrtOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case SinOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case CosOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case TanOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case CeilOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case FloorOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case SignOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case LogisticOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case Expm1Op o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case Log1pOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case CbrtOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case IsFiniteOp o -> validateIsFinite(o);
            case PopcntOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case ClzOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case RoundNearestEvenOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case RoundNearestAfzOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case NotOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            // Shape ops
            case ReshapeOp reshapeOp -> validateReshape(reshapeOp);
            case TransposeOp transposeOp -> validateTranspose(transposeOp);
            case BroadcastInDimOp broadcastOp -> validateBroadcastInDim(broadcastOp);
            case ConcatenateOp concatOp -> validateConcatenate(concatOp);
            case SliceOp sliceOp -> validateSlice(sliceOp);
            case ReverseOp o -> validateReverse(o);
            case PadOp o -> validatePad(o);
            case IotaOp o -> {} // Iota has no operands to validate
            case GatherOp o -> {} // Complex validation, skip for now
            case ScatterOp o -> {} // Complex validation, skip for now
            case DynamicSliceOp o -> {} // Complex validation, skip for now
            case DynamicUpdateSliceOp o -> {} // Complex validation, skip for now
            // Conditional ops
            case CompareOp compareOp -> validateCompare(compareOp);
            case SelectOp selectOp -> validateSelect(selectOp);
            case ClampOp clampOp -> validateClamp(clampOp);
            // Type conversion
            case ConvertOp convertOp -> validateConvert(convertOp);
            case BitcastConvertOp o -> validateConvert(new ConvertOp(o.result(), o.operand(), o.tensorResultType()));
            // Reduction
            case ReduceOp reduceOp -> validateReduce(reduceOp);
            case ReduceWindowOp o -> {} // Complex validation, skip for now
            // Convolution and neural network
            case ConvolutionOp o -> {} // Complex validation, skip for now
            case BatchNormTrainingOp o -> {} // Complex validation, skip for now
            case BatchNormInferenceOp o -> {} // Complex validation, skip for now
            // Control flow
            case IfOp o -> {} // Complex validation, skip for now
            case WhileOp o -> {} // Complex validation, skip for now
            // Linear algebra
            case DotOp o -> validateBinaryElementwise(o, o.lhs(), o.rhs(), o.tensorResultType());
            case CholeskyOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case TriangularSolveOp o -> {} // Complex validation, skip for now
            // Complex numbers
            case RealOp o -> {} // Complex validation, skip for now
            case ImagOp o -> {} // Complex validation, skip for now
            case ComplexOp o -> {} // Complex validation, skip for now
            // Signal processing
            case FftOp o -> {} // Complex validation, skip for now
            // Other
            case SortOp o -> {} // Complex validation, skip for now
            case RngOp o -> {} // Complex validation, skip for now
            case RngBitGeneratorOp o -> {} // Complex validation, skip for now
            case ConstantOp ignored -> {} // Constants are always valid
            case ReturnOp ignored -> {} // Validated separately
            case CustomCallOp ignored -> {} // Custom calls are opaque
            // Dynamic shape operations
            case DynamicBroadcastInDimOp o -> {} // Complex validation, skip for now
            case DynamicGatherOp o -> {} // Complex validation, skip for now
            case DynamicIotaOp o -> {} // Complex validation, skip for now
            case DynamicPadOp o -> {} // Complex validation, skip for now
            case DynamicReshapeOp o -> {} // Complex validation, skip for now
            case DynamicConvOp o -> {} // Complex validation, skip for now
            case GetDimensionSizeOp o -> {} // Complex validation, skip for now
            // Quantization operations
            case UniformQuantizeOp o -> validateConvert(new ConvertOp(o.result(), o.operand(), o.tensorResultType()));
            case UniformDequantizeOp o -> validateConvert(new ConvertOp(o.result(), o.operand(), o.tensorResultType()));
            // Additional reduction operations
            case ReducePrecisionOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case SelectAndScatterOp o -> {} // Complex validation, skip for now
            // Additional neural network operations
            case BatchNormGradOp o -> {} // Complex validation, skip for now
            // Additional control flow operations
            case CaseOp o -> {} // Complex validation, skip for now
            case MapOp o -> {} // Complex validation, skip for now
            // Distributed/collective operations
            case AfterAllOp o -> {} // Complex validation, skip for now
            case AllGatherOp o -> {} // Complex validation, skip for now
            case AllReduceOp o -> validateUnaryElementwise(o, o.operand(), o.tensorResultType());
            case AllToAllOp o -> {} // Complex validation, skip for now
            case CollectiveBroadcastOp o -> {} // Complex validation, skip for now
            case CollectivePermuteOp o -> {} // Complex validation, skip for now
            case PartitionIdOp o -> {} // No operands to validate
            case ReduceScatterOp o -> {} // Complex validation, skip for now
            case ReplicaIdOp o -> {} // No operands to validate
            // Communication operations
            case InfeedOp o -> {} // Complex validation, skip for now
            case OutfeedOp o -> {} // Complex validation, skip for now
            case RecvOp o -> {} // Complex validation, skip for now
            case SendOp o -> {} // Complex validation, skip for now
            // Tuple operations
            case TupleOp o -> {} // Complex validation, skip for now
            case GetTupleElementOp o -> {} // Complex validation, skip for now
            // Other operations
            case OptimizationBarrierOp o -> {} // Pass-through, no validation needed
            case CompositeOp o -> {} // Opaque, no validation needed
        }
    }

    private void validateDotGeneral(DotGeneralOp op) {
        Type lhsType = op.lhs().type();
        Type rhsType = op.rhs().type();

        if (!(lhsType instanceof TensorType lhsTensor)) {
            error("dot_general lhs must be tensor type, got %s", lhsType);
            return;
        }
        if (!(rhsType instanceof TensorType rhsTensor)) {
            error("dot_general rhs must be tensor type, got %s", rhsType);
            return;
        }

        // Check element types match
        if (!lhsTensor.elementType().equals(rhsTensor.elementType())) {
            error("dot_general operand element types must match: %s vs %s",
                    lhsTensor.elementType().toMlirString(),
                    rhsTensor.elementType().toMlirString());
        }

        DotDimensionNumbers dims = op.dimensionNumbers();

        // Validate contracting dimensions have matching sizes
        for (int i = 0; i < dims.lhsContractingDimensions().size(); i++) {
            int lhsDim = dims.lhsContractingDimensions().get(i).intValue();
            int rhsDim = dims.rhsContractingDimensions().get(i).intValue();

            if (lhsDim >= lhsTensor.rank()) {
                error("dot_general lhs contracting dimension %d out of range (rank=%d)",
                        lhsDim, lhsTensor.rank());
                continue;
            }
            if (rhsDim >= rhsTensor.rank()) {
                error("dot_general rhs contracting dimension %d out of range (rank=%d)",
                        rhsDim, rhsTensor.rank());
                continue;
            }

            int lhsSize = lhsTensor.dim(lhsDim);
            int rhsSize = rhsTensor.dim(rhsDim);
            if (lhsSize != rhsSize) {
                error("dot_general contracting dimensions must have same size: lhs dim %d = %d, rhs dim %d = %d",
                        lhsDim, lhsSize, rhsDim, rhsSize);
            }
        }

        // Validate batching dimensions have matching sizes
        for (int i = 0; i < dims.lhsBatchingDimensions().size(); i++) {
            int lhsDim = dims.lhsBatchingDimensions().get(i).intValue();
            int rhsDim = dims.rhsBatchingDimensions().get(i).intValue();

            if (lhsDim >= lhsTensor.rank() || rhsDim >= rhsTensor.rank()) {
                continue; // Already reported above
            }

            int lhsSize = lhsTensor.dim(lhsDim);
            int rhsSize = rhsTensor.dim(rhsDim);
            if (lhsSize != rhsSize) {
                error("dot_general batching dimensions must have same size: lhs dim %d = %d, rhs dim %d = %d",
                        lhsDim, lhsSize, rhsDim, rhsSize);
            }
        }

        // Validate result shape
        validateResultShape(op, computeDotGeneralResultShape(lhsTensor, rhsTensor, dims));
    }

    private TensorType computeDotGeneralResultShape(TensorType lhs, TensorType rhs, DotDimensionNumbers dims) {
        List<Integer> resultShape = new ArrayList<>();

        // Add batching dimensions
        for (Long batchDim : dims.lhsBatchingDimensions()) {
            resultShape.add(lhs.dim(batchDim.intValue()));
        }

        // Add non-contracting, non-batching dimensions from lhs
        Set<Integer> lhsExcluded = new HashSet<>();
        dims.lhsBatchingDimensions().forEach(d -> lhsExcluded.add(d.intValue()));
        dims.lhsContractingDimensions().forEach(d -> lhsExcluded.add(d.intValue()));

        for (int i = 0; i < lhs.rank(); i++) {
            if (!lhsExcluded.contains(i)) {
                resultShape.add(lhs.dim(i));
            }
        }

        // Add non-contracting, non-batching dimensions from rhs
        Set<Integer> rhsExcluded = new HashSet<>();
        dims.rhsBatchingDimensions().forEach(d -> rhsExcluded.add(d.intValue()));
        dims.rhsContractingDimensions().forEach(d -> rhsExcluded.add(d.intValue()));

        for (int i = 0; i < rhs.rank(); i++) {
            if (!rhsExcluded.contains(i)) {
                resultShape.add(rhs.dim(i));
            }
        }

        return new TensorType(resultShape, lhs.elementType());
    }

    private void validateBinaryElementwise(Operation op, Value lhs, Value rhs, TensorType resultType) {
        if (!(lhs.type() instanceof TensorType lhsTensor)) {
            error("%s lhs must be tensor type", op.opName());
            return;
        }
        if (!(rhs.type() instanceof TensorType rhsTensor)) {
            error("%s rhs must be tensor type", op.opName());
            return;
        }

        // Check shapes match exactly
        if (!lhsTensor.shape().equals(rhsTensor.shape())) {
            error("%s operand shapes must match: %s vs %s",
                    op.opName(),
                    lhsTensor.toMlirString(),
                    rhsTensor.toMlirString());
        }

        // Check element types match
        if (!lhsTensor.elementType().equals(rhsTensor.elementType())) {
            error("%s operand element types must match: %s vs %s",
                    op.opName(),
                    lhsTensor.elementType().toMlirString(),
                    rhsTensor.elementType().toMlirString());
        }

        // Result shape should match operands
        if (!resultType.shape().equals(lhsTensor.shape())) {
            error("%s result shape %s doesn't match operand shape %s",
                    op.opName(),
                    resultType.toMlirString(),
                    lhsTensor.toMlirString());
        }
    }

    private void validateUnaryElementwise(Operation op, Value operand, TensorType resultType) {
        if (!(operand.type() instanceof TensorType operandTensor)) {
            error("%s operand must be tensor type", op.opName());
            return;
        }

        // Result should match operand
        if (!resultType.shape().equals(operandTensor.shape())) {
            error("%s result shape %s doesn't match operand shape %s",
                    op.opName(),
                    resultType.toMlirString(),
                    operandTensor.toMlirString());
        }

        if (!resultType.elementType().equals(operandTensor.elementType())) {
            error("%s result element type %s doesn't match operand %s",
                    op.opName(),
                    resultType.elementType().toMlirString(),
                    operandTensor.elementType().toMlirString());
        }
    }

    private void validateReshape(ReshapeOp op) {
        if (!(op.operand().type() instanceof TensorType operandTensor)) {
            error("reshape operand must be tensor type");
            return;
        }

        TensorType resultType = op.tensorResultType();

        // Element counts must match
        long inputElements = operandTensor.elementCount();
        long outputElements = resultType.elementCount();

        if (inputElements != outputElements) {
            error("reshape element count mismatch: input has %d elements, output has %d",
                    inputElements, outputElements);
        }

        // Element types must match
        if (!operandTensor.elementType().equals(resultType.elementType())) {
            error("reshape element types must match: %s vs %s",
                    operandTensor.elementType().toMlirString(),
                    resultType.elementType().toMlirString());
        }
    }

    private void validateTranspose(TransposeOp op) {
        if (!(op.operand().type() instanceof TensorType operandTensor)) {
            error("transpose operand must be tensor type");
            return;
        }

        List<Long> perm = op.permutation();

        // Permutation must be valid
        if (perm.size() != operandTensor.rank()) {
            error("transpose permutation length %d doesn't match operand rank %d",
                    perm.size(), operandTensor.rank());
            return;
        }

        // Check permutation is valid (contains 0..rank-1 exactly once)
        Set<Long> seen = new HashSet<>();
        boolean hasInvalidPerm = false;
        for (Long p : perm) {
            if (p < 0 || p >= operandTensor.rank()) {
                error("transpose permutation value %d out of range [0, %d)", p, operandTensor.rank());
                hasInvalidPerm = true;
            }
            if (!seen.add(p)) {
                error("transpose permutation has duplicate value %d", p);
            }
        }

        // Skip result shape validation if permutation is invalid
        if (hasInvalidPerm) {
            return;
        }

        // Validate result shape
        List<Integer> expectedShape = new ArrayList<>();
        for (Long p : perm) {
            expectedShape.add(operandTensor.dim(p.intValue()));
        }

        if (!op.tensorResultType().shape().equals(expectedShape)) {
            error("transpose result shape %s doesn't match expected %s",
                    op.tensorResultType().toMlirString(),
                    new TensorType(expectedShape, operandTensor.elementType()).toMlirString());
        }
    }

    private void validateBroadcastInDim(BroadcastInDimOp op) {
        if (!(op.operand().type() instanceof TensorType operandTensor)) {
            error("broadcast_in_dim operand must be tensor type");
            return;
        }

        List<Long> dims = op.broadcastDimensions();
        TensorType resultType = op.tensorResultType();

        // Broadcast dimensions must match operand rank
        if (dims.size() != operandTensor.rank()) {
            error("broadcast_in_dim dimensions count %d doesn't match operand rank %d",
                    dims.size(), operandTensor.rank());
            return;
        }

        // Each broadcast dimension must be valid for result
        for (int i = 0; i < dims.size(); i++) {
            int dim = dims.get(i).intValue();
            if (dim < 0 || dim >= resultType.rank()) {
                error("broadcast_in_dim dimension %d out of range for result rank %d", dim, resultType.rank());
                continue;
            }

            // Operand dimension must be 1 or match result dimension
            int operandSize = operandTensor.dim(i);
            int resultSize = resultType.dim(dim);
            if (operandSize != 1 && operandSize != resultSize) {
                error("broadcast_in_dim operand dimension %d (size %d) incompatible with result dimension %d (size %d)",
                        i, operandSize, dim, resultSize);
            }
        }
    }

    private void validateReduce(ReduceOp op) {
        if (!(op.operand().type() instanceof TensorType operandTensor)) {
            error("reduce operand must be tensor type");
            return;
        }

        // Validate dimensions are in range
        for (Long dim : op.dimensions()) {
            if (dim < 0 || dim >= operandTensor.rank()) {
                error("reduce dimension %d out of range for operand rank %d", dim, operandTensor.rank());
            }
        }

        // Validate result shape
        List<Integer> expectedShape = new ArrayList<>();
        Set<Long> reducedDims = new HashSet<>(op.dimensions());
        for (int i = 0; i < operandTensor.rank(); i++) {
            if (!reducedDims.contains((long) i)) {
                expectedShape.add(operandTensor.dim(i));
            }
        }

        if (!op.tensorResultType().shape().equals(expectedShape)) {
            error("reduce result shape %s doesn't match expected %s",
                    op.tensorResultType().toMlirString(),
                    new TensorType(expectedShape, operandTensor.elementType()).toMlirString());
        }
    }

    private void validateCompare(CompareOp op) {
        if (!(op.lhs().type() instanceof TensorType lhsTensor)) {
            error("compare lhs must be tensor type");
            return;
        }
        if (!(op.rhs().type() instanceof TensorType rhsTensor)) {
            error("compare rhs must be tensor type");
            return;
        }

        // Shapes must match
        if (!lhsTensor.shape().equals(rhsTensor.shape())) {
            error("compare operand shapes must match: %s vs %s",
                    lhsTensor.toMlirString(), rhsTensor.toMlirString());
        }

        // Element types must match
        if (!lhsTensor.elementType().equals(rhsTensor.elementType())) {
            error("compare operand element types must match: %s vs %s",
                    lhsTensor.elementType().toMlirString(),
                    rhsTensor.elementType().toMlirString());
        }

        // Result must be i1 tensor with same shape
        TensorType resultType = op.tensorResultType();
        if (!resultType.shape().equals(lhsTensor.shape())) {
            error("compare result shape %s must match operand shape %s",
                    resultType.toMlirString(), lhsTensor.toMlirString());
        }
        if (!resultType.elementType().equals(ScalarType.I1)) {
            error("compare result must have element type i1, got %s",
                    resultType.elementType().toMlirString());
        }
    }

    private void validateSelect(SelectOp op) {
        if (!(op.pred().type() instanceof TensorType predTensor)) {
            error("select predicate must be tensor type");
            return;
        }
        if (!(op.onTrue().type() instanceof TensorType trueTensor)) {
            error("select on_true must be tensor type");
            return;
        }
        if (!(op.onFalse().type() instanceof TensorType falseTensor)) {
            error("select on_false must be tensor type");
            return;
        }

        // Predicate must be i1
        if (!predTensor.elementType().equals(ScalarType.I1)) {
            error("select predicate must have element type i1, got %s",
                    predTensor.elementType().toMlirString());
        }

        // All shapes must match
        if (!predTensor.shape().equals(trueTensor.shape())) {
            error("select predicate shape %s must match on_true shape %s",
                    predTensor.toMlirString(), trueTensor.toMlirString());
        }
        if (!trueTensor.shape().equals(falseTensor.shape())) {
            error("select on_true shape %s must match on_false shape %s",
                    trueTensor.toMlirString(), falseTensor.toMlirString());
        }

        // on_true and on_false element types must match
        if (!trueTensor.elementType().equals(falseTensor.elementType())) {
            error("select on_true and on_false element types must match: %s vs %s",
                    trueTensor.elementType().toMlirString(),
                    falseTensor.elementType().toMlirString());
        }

        // Result must match on_true/on_false
        TensorType resultType = op.tensorResultType();
        if (!resultType.equals(trueTensor)) {
            error("select result type %s must match on_true type %s",
                    resultType.toMlirString(), trueTensor.toMlirString());
        }
    }

    private void validateConcatenate(ConcatenateOp op) {
        if (op.inputs().isEmpty()) {
            error("concatenate requires at least one input");
            return;
        }

        TensorType firstType = null;
        for (Value input : op.inputs()) {
            if (!(input.type() instanceof TensorType inputTensor)) {
                error("concatenate input must be tensor type");
                return;
            }
            if (firstType == null) {
                firstType = inputTensor;
            }
        }

        int dim = (int) op.dimension();
        if (dim < 0 || dim >= firstType.rank()) {
            error("concatenate dimension %d out of range for rank %d", dim, firstType.rank());
            return;
        }

        // All inputs must have same rank and matching non-concat dimensions
        long totalConcatSize = 0;
        for (Value input : op.inputs()) {
            TensorType inputTensor = (TensorType) input.type();
            if (inputTensor.rank() != firstType.rank()) {
                error("concatenate inputs must have same rank: %d vs %d",
                        inputTensor.rank(), firstType.rank());
                continue;
            }
            for (int i = 0; i < inputTensor.rank(); i++) {
                if (i != dim && inputTensor.dim(i) != firstType.dim(i)) {
                    error("concatenate inputs must have matching dimensions except concat axis");
                    break;
                }
            }
            totalConcatSize += inputTensor.dim(dim);
        }

        // Validate result shape
        List<Integer> expectedShape = new ArrayList<>(firstType.shape());
        expectedShape.set(dim, (int) totalConcatSize);
        if (!op.tensorResultType().shape().equals(expectedShape)) {
            error("concatenate result shape %s doesn't match expected %s",
                    op.tensorResultType().toMlirString(),
                    new TensorType(expectedShape, firstType.elementType()).toMlirString());
        }
    }

    private void validateSlice(SliceOp op) {
        if (!(op.operand().type() instanceof TensorType operandTensor)) {
            error("slice operand must be tensor type");
            return;
        }

        int rank = operandTensor.rank();
        if (op.startIndices().size() != rank || op.limitIndices().size() != rank || op.strides().size() != rank) {
            error("slice start/limit/strides must have same size as operand rank %d", rank);
            return;
        }

        // Validate indices are in range and compute result shape
        List<Integer> expectedShape = new ArrayList<>();
        for (int i = 0; i < rank; i++) {
            long start = op.startIndices().get(i);
            long limit = op.limitIndices().get(i);
            long stride = op.strides().get(i);

            if (start < 0 || start > operandTensor.dim(i)) {
                error("slice start[%d]=%d out of range [0, %d]", i, start, operandTensor.dim(i));
            }
            if (limit < start || limit > operandTensor.dim(i)) {
                error("slice limit[%d]=%d invalid (start=%d, dim=%d)", i, limit, start, operandTensor.dim(i));
            }
            if (stride <= 0) {
                error("slice stride[%d]=%d must be positive", i, stride);
            }

            // Result size = ceil((limit - start) / stride)
            int size = (int) ((limit - start + stride - 1) / stride);
            expectedShape.add(size);
        }

        if (!op.tensorResultType().shape().equals(expectedShape)) {
            error("slice result shape %s doesn't match expected %s",
                    op.tensorResultType().toMlirString(),
                    new TensorType(expectedShape, operandTensor.elementType()).toMlirString());
        }
    }

    private void validateClamp(ClampOp op) {
        if (!(op.min().type() instanceof TensorType minTensor)) {
            error("clamp min must be tensor type");
            return;
        }
        if (!(op.operand().type() instanceof TensorType operandTensor)) {
            error("clamp operand must be tensor type");
            return;
        }
        if (!(op.max().type() instanceof TensorType maxTensor)) {
            error("clamp max must be tensor type");
            return;
        }

        // min and max can be scalar (rank 0) or match operand shape
        if (minTensor.rank() != 0 && !minTensor.shape().equals(operandTensor.shape())) {
            error("clamp min must be scalar or match operand shape");
        }
        if (maxTensor.rank() != 0 && !maxTensor.shape().equals(operandTensor.shape())) {
            error("clamp max must be scalar or match operand shape");
        }

        // Result must match operand
        if (!op.tensorResultType().equals(operandTensor)) {
            error("clamp result type %s must match operand type %s",
                    op.tensorResultType().toMlirString(), operandTensor.toMlirString());
        }
    }

    private void validateConvert(ConvertOp op) {
        if (!(op.operand().type() instanceof TensorType operandTensor)) {
            error("convert operand must be tensor type");
            return;
        }

        // Result must have same shape as operand
        if (!op.tensorResultType().shape().equals(operandTensor.shape())) {
            error("convert result shape %s must match operand shape %s",
                    op.tensorResultType().toMlirString(), operandTensor.toMlirString());
        }
        // Element type can differ (that's the point of convert)
    }

    private void validateIsFinite(IsFiniteOp op) {
        if (!(op.operand().type() instanceof TensorType operandTensor)) {
            error("is_finite operand must be tensor type");
            return;
        }

        // Operand must be floating point
        if (!operandTensor.elementType().isFloatingPoint()) {
            error("is_finite operand must have floating point element type, got %s",
                    operandTensor.elementType().toMlirString());
        }

        // Result must have same shape but i1 element type
        TensorType resultType = op.tensorResultType();
        if (!resultType.shape().equals(operandTensor.shape())) {
            error("is_finite result shape %s must match operand shape %s",
                    resultType.toMlirString(), operandTensor.toMlirString());
        }
        if (!resultType.elementType().equals(ScalarType.I1)) {
            error("is_finite result must have element type i1, got %s",
                    resultType.elementType().toMlirString());
        }
    }

    private void validateReverse(ReverseOp op) {
        if (!(op.operand().type() instanceof TensorType operandTensor)) {
            error("reverse operand must be tensor type");
            return;
        }

        // Validate dimensions are in range
        for (Long dim : op.dimensions()) {
            if (dim < 0 || dim >= operandTensor.rank()) {
                error("reverse dimension %d out of range for operand rank %d", dim, operandTensor.rank());
            }
        }

        // Result must match operand (reverse preserves shape)
        if (!op.tensorResultType().equals(operandTensor)) {
            error("reverse result type %s must match operand type %s",
                    op.tensorResultType().toMlirString(), operandTensor.toMlirString());
        }
    }

    private void validatePad(PadOp op) {
        if (!(op.operand().type() instanceof TensorType operandTensor)) {
            error("pad operand must be tensor type");
            return;
        }

        int rank = operandTensor.rank();

        // Padding arrays must match rank
        if (op.edgePaddingLow().size() != rank) {
            error("pad edge_padding_low size %d must match operand rank %d",
                    op.edgePaddingLow().size(), rank);
        }
        if (op.edgePaddingHigh().size() != rank) {
            error("pad edge_padding_high size %d must match operand rank %d",
                    op.edgePaddingHigh().size(), rank);
        }
        if (op.interiorPadding().size() != rank) {
            error("pad interior_padding size %d must match operand rank %d",
                    op.interiorPadding().size(), rank);
        }

        // Interior padding must be non-negative
        for (int i = 0; i < op.interiorPadding().size(); i++) {
            if (op.interiorPadding().get(i) < 0) {
                error("pad interior_padding[%d]=%d must be non-negative", i, op.interiorPadding().get(i));
            }
        }

        // Compute and validate result shape
        List<Integer> expectedShape = new ArrayList<>();
        for (int i = 0; i < rank; i++) {
            long low = op.edgePaddingLow().get(i);
            long high = op.edgePaddingHigh().get(i);
            long interior = op.interiorPadding().get(i);
            int operandDim = operandTensor.dim(i);
            // Result dim = low + operandDim + interior * (operandDim - 1) + high
            long resultDim = low + operandDim + interior * Math.max(0, operandDim - 1) + high;
            expectedShape.add((int) resultDim);
        }

        if (!op.tensorResultType().shape().equals(expectedShape)) {
            error("pad result shape %s doesn't match expected %s",
                    op.tensorResultType().toMlirString(),
                    new TensorType(expectedShape, operandTensor.elementType()).toMlirString());
        }
    }

    private void validateReturnTypes(ReturnOp returnOp, Function function) {
        List<Value> operands = returnOp.operands();
        List<Type> resultTypes = function.resultTypes();

        if (operands.size() != resultTypes.size()) {
            error("Function '%s' returns %d values but declares %d result types",
                    function.name(), operands.size(), resultTypes.size());
            return;
        }

        for (int i = 0; i < operands.size(); i++) {
            Type actual = operands.get(i).type();
            Type expected = resultTypes.get(i);

            if (!typesMatch(actual, expected)) {
                error("Function '%s' return value %d has type %s but declares %s",
                        function.name(), i, actual.toMlirString(), expected.toMlirString());
            }
        }
    }

    private boolean typesMatch(Type actual, Type expected) {
        if (actual instanceof TensorType at && expected instanceof TensorType et) {
            return at.equals(et);
        }
        if (actual instanceof ScalarType as && expected instanceof ScalarType es) {
            return as.equals(es);
        }
        return false;
    }

    private void validateResultShape(Operation op, TensorType expected) {
        if (!(op.tensorResultType() instanceof TensorType actual)) {
            error("%s result must be tensor type", op.opName());
            return;
        }

        if (!actual.equals(expected)) {
            error("%s result shape %s doesn't match computed shape %s",
                    op.opName(),
                    actual.toMlirString(),
                    expected.toMlirString());
        }
    }

    private void error(String format, Object... args) {
        errors.add(String.format(format, args));
    }
}
