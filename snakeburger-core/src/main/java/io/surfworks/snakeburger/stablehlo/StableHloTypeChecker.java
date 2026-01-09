package io.surfworks.snakeburger.stablehlo;

import io.surfworks.snakeburger.stablehlo.StableHloAst.*;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
            case AddOp addOp -> validateBinaryElementwise(addOp, addOp.lhs(), addOp.rhs(), addOp.tensorResultType());
            case MultiplyOp mulOp -> validateBinaryElementwise(mulOp, mulOp.lhs(), mulOp.rhs(), mulOp.tensorResultType());
            case DivideOp divOp -> validateBinaryElementwise(divOp, divOp.lhs(), divOp.rhs(), divOp.tensorResultType());
            case MaximumOp maxOp -> validateBinaryElementwise(maxOp, maxOp.lhs(), maxOp.rhs(), maxOp.tensorResultType());
            case NegateOp negOp -> validateUnaryElementwise(negOp, negOp.operand(), negOp.tensorResultType());
            case ReshapeOp reshapeOp -> validateReshape(reshapeOp);
            case TransposeOp transposeOp -> validateTranspose(transposeOp);
            case BroadcastInDimOp broadcastOp -> validateBroadcastInDim(broadcastOp);
            case ConstantOp ignored -> {} // Constants are always valid
            case ReturnOp ignored -> {} // Validated separately
            case ReduceOp reduceOp -> validateReduce(reduceOp);
            case CustomCallOp ignored -> {} // Custom calls are opaque
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
