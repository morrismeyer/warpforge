package io.surfworks.snakeburger.stablehlo;

import io.surfworks.snakeburger.stablehlo.StableHloAst.*;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ComparisonDirection;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Module;
import io.surfworks.snakeburger.stablehlo.StableHloTokenizer.Token;
import io.surfworks.snakeburger.stablehlo.StableHloTokenizer.TokenType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Parser for StableHLO MLIR text format.
 *
 * Implements a top-down recursive descent parser that produces
 * a complete AST representation of the StableHLO module.
 */
public final class StableHloParser {

    private final List<Token> tokens;
    private int pos;
    private final Map<String, Value> valueMap = new HashMap<>();

    public StableHloParser(List<Token> tokens) {
        this.tokens = tokens;
        this.pos = 0;
    }

    public static Module parse(String input) {
        StableHloTokenizer tokenizer = new StableHloTokenizer(input);
        List<Token> tokens = tokenizer.tokenize();
        StableHloParser parser = new StableHloParser(tokens);
        return parser.parseModule();
    }

    // ==================== Module Parsing ====================

    public Module parseModule() {
        expect(TokenType.IDENTIFIER, "module");
        String name = parseAtId();
        expect(TokenType.LBRACE);

        List<Function> functions = new ArrayList<>();
        while (!check(TokenType.RBRACE) && !check(TokenType.EOF)) {
            functions.add(parseFunction());
        }

        expect(TokenType.RBRACE);
        return new Module(name, functions);
    }

    // ==================== Function Parsing ====================

    private Function parseFunction() {
        // func.func public @name(%arg0: type, ...) -> (result_types) { body }
        expect(TokenType.IDENTIFIER, "func.func");

        boolean isPublic = false;
        if (checkIdentifier("public")) {
            advance();
            isPublic = true;
        }

        String name = parseAtId();
        expect(TokenType.LPAREN);

        // Parse arguments
        List<Argument> arguments = new ArrayList<>();
        valueMap.clear();

        while (!check(TokenType.RPAREN)) {
            if (!arguments.isEmpty()) {
                expect(TokenType.COMMA);
            }
            Argument arg = parseArgument();
            arguments.add(arg);
            valueMap.put(arg.name(), arg.toValue());
        }
        expect(TokenType.RPAREN);

        // Parse result types: -> (type, type, ...)
        expect(TokenType.ARROW);
        expect(TokenType.LPAREN);
        List<Type> resultTypes = new ArrayList<>();
        while (!check(TokenType.RPAREN)) {
            if (!resultTypes.isEmpty()) {
                expect(TokenType.COMMA);
            }
            resultTypes.add(parseType());
        }
        expect(TokenType.RPAREN);

        // Parse body
        expect(TokenType.LBRACE);
        List<Operation> body = new ArrayList<>();
        while (!check(TokenType.RBRACE)) {
            Operation op = parseOperation();
            body.add(op);
        }
        expect(TokenType.RBRACE);

        return new Function(name, arguments, resultTypes, body, isPublic);
    }

    private Argument parseArgument() {
        // %arg0: tensor<4x8xf32>
        String name = parsePercentId();
        expect(TokenType.COLON);
        Type type = parseType();
        return new Argument(name, type);
    }

    // ==================== Type Parsing ====================

    private Type parseType() {
        if (checkIdentifier("tensor")) {
            return parseTensorType();
        }
        // Scalar type
        Token t = expect(TokenType.IDENTIFIER);
        return ScalarType.of(t.value());
    }

    private TensorType parseTensorType() {
        expect(TokenType.IDENTIFIER, "tensor");
        expect(TokenType.LANGLE);

        List<Integer> shape = new ArrayList<>();
        ScalarType elementType = null;

        // Parse shape dimensions and element type
        // Format: tensor<4x8xf32> or tensor<f32> (scalar tensor)
        // Note: tokenizer may produce "x8xf32" as a single identifier
        while (!check(TokenType.RANGLE)) {
            if (check(TokenType.INTEGER)) {
                shape.add(Integer.parseInt(advance().value()));
                // Expect 'x' separator or end
                if (checkIdentifier("x")) {
                    advance();
                }
            } else if (check(TokenType.IDENTIFIER)) {
                String typeName = advance().value();
                // Handle combined tokens like "x8xf32" or "xf32"
                if (typeName.startsWith("x")) {
                    typeName = typeName.substring(1); // Remove leading 'x'
                }
                // Parse any embedded dimensions: "8xf32" or "8x16xf32"
                while (!typeName.isEmpty() && Character.isDigit(typeName.charAt(0))) {
                    int i = 0;
                    while (i < typeName.length() && Character.isDigit(typeName.charAt(i))) {
                        i++;
                    }
                    shape.add(Integer.parseInt(typeName.substring(0, i)));
                    typeName = typeName.substring(i);
                    if (typeName.startsWith("x")) {
                        typeName = typeName.substring(1);
                    }
                }
                if (!typeName.isEmpty()) {
                    elementType = ScalarType.of(typeName);
                }
            } else {
                throw error("Expected shape dimension or element type");
            }
        }

        expect(TokenType.RANGLE);

        if (elementType == null) {
            throw error("Missing element type in tensor type");
        }

        return new TensorType(shape, elementType);
    }

    // ==================== Operation Parsing ====================

    private Operation parseOperation() {
        // Check for return first
        if (checkIdentifier("stablehlo.return")) {
            return parseReturn();
        }

        // Regular op: %result = op_name operands : types
        String resultName = parsePercentId();
        expect(TokenType.EQUALS);

        Token opToken = expect(TokenType.IDENTIFIER);
        String opName = opToken.value();

        return switch (opName) {
            case "stablehlo.dot_general" -> parseDotGeneral(resultName);
            case "stablehlo.constant" -> parseConstant(resultName);
            // Binary elementwise ops
            case "stablehlo.add" -> parseAdd(resultName);
            case "stablehlo.subtract" -> parseSubtract(resultName);
            case "stablehlo.multiply" -> parseMultiply(resultName);
            case "stablehlo.divide" -> parseDivide(resultName);
            case "stablehlo.maximum" -> parseMaximum(resultName);
            case "stablehlo.minimum" -> parseMinimum(resultName);
            case "stablehlo.power" -> parsePower(resultName);
            case "stablehlo.remainder" -> parseRemainder(resultName);
            case "stablehlo.atan2" -> parseAtan2(resultName);
            // Binary logical/bitwise ops
            case "stablehlo.and" -> parseAnd(resultName);
            case "stablehlo.or" -> parseOr(resultName);
            case "stablehlo.xor" -> parseXor(resultName);
            case "stablehlo.shift_left" -> parseShiftLeft(resultName);
            case "stablehlo.shift_right_arithmetic" -> parseShiftRightArithmetic(resultName);
            case "stablehlo.shift_right_logical" -> parseShiftRightLogical(resultName);
            // Unary elementwise ops
            case "stablehlo.negate" -> parseNegate(resultName);
            case "stablehlo.abs" -> parseAbs(resultName);
            case "stablehlo.exponential" -> parseExp(resultName);
            case "stablehlo.log" -> parseLog(resultName);
            case "stablehlo.tanh" -> parseTanh(resultName);
            case "stablehlo.sqrt" -> parseSqrt(resultName);
            case "stablehlo.rsqrt" -> parseRsqrt(resultName);
            case "stablehlo.sine" -> parseSin(resultName);
            case "stablehlo.cosine" -> parseCos(resultName);
            case "stablehlo.tan" -> parseTan(resultName);
            case "stablehlo.ceil" -> parseCeil(resultName);
            case "stablehlo.floor" -> parseFloor(resultName);
            case "stablehlo.sign" -> parseSign(resultName);
            case "stablehlo.logistic" -> parseLogistic(resultName);
            case "stablehlo.exponential_minus_one" -> parseExpm1(resultName);
            case "stablehlo.log_plus_one" -> parseLog1p(resultName);
            case "stablehlo.cbrt" -> parseCbrt(resultName);
            case "stablehlo.is_finite" -> parseIsFinite(resultName);
            case "stablehlo.popcnt" -> parsePopcnt(resultName);
            case "stablehlo.count_leading_zeros" -> parseClz(resultName);
            case "stablehlo.round_nearest_even" -> parseRoundNearestEven(resultName);
            case "stablehlo.round_nearest_afz" -> parseRoundNearestAfz(resultName);
            case "stablehlo.not" -> parseNot(resultName);
            // Shape ops
            case "stablehlo.reshape" -> parseReshape(resultName);
            case "stablehlo.transpose" -> parseTranspose(resultName);
            case "stablehlo.broadcast_in_dim" -> parseBroadcastInDim(resultName);
            case "stablehlo.concatenate" -> parseConcatenate(resultName);
            case "stablehlo.slice" -> parseSlice(resultName);
            case "stablehlo.reverse" -> parseReverse(resultName);
            case "stablehlo.pad" -> parsePad(resultName);
            case "stablehlo.iota" -> parseIota(resultName);
            case "stablehlo.gather" -> parseGather(resultName);
            case "stablehlo.scatter" -> parseScatter(resultName);
            case "stablehlo.dynamic_slice" -> parseDynamicSlice(resultName);
            case "stablehlo.dynamic_update_slice" -> parseDynamicUpdateSlice(resultName);
            // Conditional ops
            case "stablehlo.compare" -> parseCompare(resultName);
            case "stablehlo.select" -> parseSelect(resultName);
            case "stablehlo.clamp" -> parseClamp(resultName);
            // Type conversion
            case "stablehlo.convert" -> parseConvert(resultName);
            case "stablehlo.bitcast_convert" -> parseBitcastConvert(resultName);
            // Reduction ops
            case "stablehlo.reduce" -> parseReduce(resultName);
            case "stablehlo.reduce_window" -> parseReduceWindow(resultName);
            // Neural network ops
            case "stablehlo.convolution" -> parseConvolution(resultName);
            case "stablehlo.batch_norm_training" -> parseBatchNormTraining(resultName);
            case "stablehlo.batch_norm_inference" -> parseBatchNormInference(resultName);
            // Sort and RNG
            case "stablehlo.sort" -> parseSort(resultName);
            case "stablehlo.rng" -> parseRng(resultName);
            case "stablehlo.rng_bit_generator" -> parseRngBitGenerator(resultName);
            // Other
            case "stablehlo.dot" -> parseDot(resultName);
            case "stablehlo.real" -> parseReal(resultName);
            case "stablehlo.imag" -> parseImag(resultName);
            case "stablehlo.complex" -> parseComplex(resultName);
            case "stablehlo.fft" -> parseFft(resultName);
            case "stablehlo.cholesky" -> parseCholesky(resultName);
            case "stablehlo.triangular_solve" -> parseTriangularSolve(resultName);
            case "stablehlo.custom_call" -> parseCustomCall(resultName);
            // Dynamic shape operations
            case "stablehlo.dynamic_broadcast_in_dim" -> parseDynamicBroadcastInDim(resultName);
            case "stablehlo.dynamic_gather" -> parseDynamicGather(resultName);
            case "stablehlo.dynamic_iota" -> parseDynamicIota(resultName);
            case "stablehlo.dynamic_pad" -> parseDynamicPad(resultName);
            case "stablehlo.dynamic_reshape" -> parseDynamicReshape(resultName);
            case "stablehlo.dynamic_conv" -> parseDynamicConv(resultName);
            case "stablehlo.get_dimension_size" -> parseGetDimensionSize(resultName);
            // Quantization operations
            case "stablehlo.uniform_quantize" -> parseUniformQuantize(resultName);
            case "stablehlo.uniform_dequantize" -> parseUniformDequantize(resultName);
            // Additional reduction operations
            case "stablehlo.reduce_precision" -> parseReducePrecision(resultName);
            case "stablehlo.select_and_scatter" -> parseSelectAndScatter(resultName);
            // Additional neural network operations
            case "stablehlo.batch_norm_grad" -> parseBatchNormGrad(resultName);
            // Control flow
            case "stablehlo.case" -> parseCase(resultName);
            case "stablehlo.map" -> parseMap(resultName);
            // Distributed/collective operations
            case "stablehlo.after_all" -> parseAfterAll(resultName);
            case "stablehlo.all_gather" -> parseAllGather(resultName);
            case "stablehlo.all_reduce" -> parseAllReduce(resultName);
            case "stablehlo.all_to_all" -> parseAllToAll(resultName);
            case "stablehlo.collective_broadcast" -> parseCollectiveBroadcast(resultName);
            case "stablehlo.collective_permute" -> parseCollectivePermute(resultName);
            case "stablehlo.partition_id" -> parsePartitionId(resultName);
            case "stablehlo.reduce_scatter" -> parseReduceScatter(resultName);
            case "stablehlo.replica_id" -> parseReplicaId(resultName);
            // Communication operations
            case "stablehlo.infeed" -> parseInfeed(resultName);
            case "stablehlo.outfeed" -> parseOutfeed(resultName);
            case "stablehlo.recv" -> parseRecv(resultName);
            case "stablehlo.send" -> parseSend(resultName);
            // Tuple operations
            case "stablehlo.tuple" -> parseTuple(resultName);
            case "stablehlo.get_tuple_element" -> parseGetTupleElement(resultName);
            // Other operations
            case "stablehlo.optimization_barrier" -> parseOptimizationBarrier(resultName);
            case "stablehlo.composite" -> parseComposite(resultName);
            default -> throw error("Unsupported operation: " + opName);
        };
    }

    private DotGeneralOp parseDotGeneral(String resultName) {
        // %1 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<...> : (type, type) -> type
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        DotDimensionNumbers dimNumbers = parseDotDimensionNumbers();

        expect(TokenType.COLON);
        // Skip input types (already have them from operands)
        expect(TokenType.LPAREN);
        parseType(); // lhs type
        expect(TokenType.COMMA);
        parseType(); // rhs type
        expect(TokenType.RPAREN);

        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DotGeneralOp(result, lhs, rhs, dimNumbers, resultType);
    }

    private DotDimensionNumbers parseDotDimensionNumbers() {
        // #stablehlo.dot<lhs_batching_dimensions = [...], ...>
        Token hash = expect(TokenType.HASH_ID);
        if (!hash.value().equals("#stablehlo.dot")) {
            throw error("Expected #stablehlo.dot, got " + hash.value());
        }
        expect(TokenType.LANGLE);

        List<Long> lhsBatching = new ArrayList<>();
        List<Long> rhsBatching = new ArrayList<>();
        List<Long> lhsContracting = new ArrayList<>();
        List<Long> rhsContracting = new ArrayList<>();

        while (!check(TokenType.RANGLE)) {
            if (checkIdentifier("lhs_batching_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                lhsBatching = parseIntegerList();
            } else if (checkIdentifier("rhs_batching_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                rhsBatching = parseIntegerList();
            } else if (checkIdentifier("lhs_contracting_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                lhsContracting = parseIntegerList();
            } else if (checkIdentifier("rhs_contracting_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                rhsContracting = parseIntegerList();
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in dot dimension numbers: " + peek());
            }
        }

        expect(TokenType.RANGLE);
        return new DotDimensionNumbers(lhsBatching, rhsBatching, lhsContracting, rhsContracting);
    }

    private List<Long> parseIntegerList() {
        expect(TokenType.LBRACKET);
        List<Long> values = new ArrayList<>();
        while (!check(TokenType.RBRACKET)) {
            if (!values.isEmpty()) {
                expect(TokenType.COMMA);
            }
            values.add(Long.parseLong(expect(TokenType.INTEGER).value()));
        }
        expect(TokenType.RBRACKET);
        return values;
    }

    private ConstantOp parseConstant(String resultName) {
        // %zero = stablehlo.constant dense<0.0> : tensor<4x16xf32>
        expect(TokenType.IDENTIFIER, "dense");
        expect(TokenType.LANGLE);

        Object value;
        if (check(TokenType.FLOAT)) {
            value = Double.parseDouble(advance().value());
        } else if (check(TokenType.INTEGER)) {
            value = Long.parseLong(advance().value());
        } else {
            throw error("Expected numeric constant value");
        }

        expect(TokenType.RANGLE);
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        DenseAttr denseAttr = new DenseAttr(value, resultType);
        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ConstantOp(result, denseAttr, resultType);
    }

    private MaximumOp parseMaximum(String resultName) {
        // %2 = stablehlo.maximum %1, %zero : tensor<4x16xf32>
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new MaximumOp(result, lhs, rhs, resultType);
    }

    private AddOp parseAdd(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new AddOp(result, lhs, rhs, resultType);
    }

    private MultiplyOp parseMultiply(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new MultiplyOp(result, lhs, rhs, resultType);
    }

    private DivideOp parseDivide(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DivideOp(result, lhs, rhs, resultType);
    }

    private NegateOp parseNegate(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new NegateOp(result, operand, resultType);
    }

    private ReshapeOp parseReshape(String resultName) {
        // %r = stablehlo.reshape %op : tensor<...> -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ReshapeOp(result, operand, resultType);
    }

    private TransposeOp parseTranspose(String resultName) {
        // %t = stablehlo.transpose %op, dims = [1, 0] : tensor<...> -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        expect(TokenType.IDENTIFIER, "dims");
        expect(TokenType.EQUALS);
        List<Long> permutation = parseIntegerList();
        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new TransposeOp(result, operand, permutation, resultType);
    }

    private BroadcastInDimOp parseBroadcastInDim(String resultName) {
        // %b = stablehlo.broadcast_in_dim %op, dims = [...] : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        expect(TokenType.IDENTIFIER, "dims");
        expect(TokenType.EQUALS);
        List<Long> dims = parseIntegerList();
        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType(); // input type
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new BroadcastInDimOp(result, operand, dims, resultType);
    }

    // ==================== New Binary Ops ====================

    private SubtractOp parseSubtract(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new SubtractOp(result, lhs, rhs, resultType);
    }

    private MinimumOp parseMinimum(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new MinimumOp(result, lhs, rhs, resultType);
    }

    // ==================== New Unary Math Ops ====================

    private AbsOp parseAbs(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new AbsOp(result, operand, resultType);
    }

    private ExpOp parseExp(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ExpOp(result, operand, resultType);
    }

    private LogOp parseLog(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new LogOp(result, operand, resultType);
    }

    private TanhOp parseTanh(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new TanhOp(result, operand, resultType);
    }

    private SqrtOp parseSqrt(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new SqrtOp(result, operand, resultType);
    }

    private RsqrtOp parseRsqrt(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new RsqrtOp(result, operand, resultType);
    }

    // ==================== New Shape Ops ====================

    private ConcatenateOp parseConcatenate(String resultName) {
        // %c = stablehlo.concatenate %a, %b, dim = 0 : (t, t) -> t
        List<Value> inputs = new ArrayList<>();
        inputs.add(lookupValue(parsePercentId()));

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("dim")) {
                break;
            }
            inputs.add(lookupValue(parsePercentId()));
        }

        expect(TokenType.IDENTIFIER, "dim");
        expect(TokenType.EQUALS);
        long dimension = Long.parseLong(expect(TokenType.INTEGER).value());

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        while (!check(TokenType.RPAREN)) {
            parseType();
            if (check(TokenType.COMMA)) advance();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ConcatenateOp(result, inputs, dimension, resultType);
    }

    private SliceOp parseSlice(String resultName) {
        // %s = stablehlo.slice %op, starts = [...], limits = [...], strides = [...] : t -> t
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        expect(TokenType.IDENTIFIER, "starts");
        expect(TokenType.EQUALS);
        List<Long> starts = parseIntegerList();

        expect(TokenType.COMMA);
        expect(TokenType.IDENTIFIER, "limits");
        expect(TokenType.EQUALS);
        List<Long> limits = parseIntegerList();

        expect(TokenType.COMMA);
        expect(TokenType.IDENTIFIER, "strides");
        expect(TokenType.EQUALS);
        List<Long> strides = parseIntegerList();

        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new SliceOp(result, operand, starts, limits, strides, resultType);
    }

    // ==================== New Conditional Ops ====================

    private CompareOp parseCompare(String resultName) {
        // %c = stablehlo.compare %a, %b, direction = GT : (t, t) -> tensor<...xi1>
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        expect(TokenType.IDENTIFIER, "direction");
        expect(TokenType.EQUALS);
        String dirStr = expect(TokenType.IDENTIFIER).value();
        ComparisonDirection direction = ComparisonDirection.fromString(dirStr);

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new CompareOp(result, lhs, rhs, direction, resultType);
    }

    private SelectOp parseSelect(String resultName) {
        // %s = stablehlo.select %pred, %on_true, %on_false : (pred_t, t, t) -> t
        Value pred = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value onTrue = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value onFalse = lookupValue(parsePercentId());

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new SelectOp(result, pred, onTrue, onFalse, resultType);
    }

    private ClampOp parseClamp(String resultName) {
        // %c = stablehlo.clamp %min, %operand, %max : (t, t, t) -> t
        Value min = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value max = lookupValue(parsePercentId());

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ClampOp(result, min, operand, max, resultType);
    }

    // ==================== Type Conversion ====================

    private ConvertOp parseConvert(String resultName) {
        // %c = stablehlo.convert %op : tensor<...xf32> -> tensor<...xf16>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ConvertOp(result, operand, resultType);
    }

    private BitcastConvertOp parseBitcastConvert(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new BitcastConvertOp(result, operand, resultType);
    }

    // ==================== Additional Binary Ops ====================

    private PowerOp parsePower(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new PowerOp(result, lhs, rhs, resultType);
    }

    private RemainderOp parseRemainder(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new RemainderOp(result, lhs, rhs, resultType);
    }

    private Atan2Op parseAtan2(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new Atan2Op(result, lhs, rhs, resultType);
    }

    // ==================== Binary Logical/Bitwise Ops ====================

    private AndOp parseAnd(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new AndOp(result, lhs, rhs, resultType);
    }

    private OrOp parseOr(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new OrOp(result, lhs, rhs, resultType);
    }

    private XorOp parseXor(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new XorOp(result, lhs, rhs, resultType);
    }

    private ShiftLeftOp parseShiftLeft(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ShiftLeftOp(result, lhs, rhs, resultType);
    }

    private ShiftRightArithmeticOp parseShiftRightArithmetic(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ShiftRightArithmeticOp(result, lhs, rhs, resultType);
    }

    private ShiftRightLogicalOp parseShiftRightLogical(String resultName) {
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ShiftRightLogicalOp(result, lhs, rhs, resultType);
    }

    // ==================== Additional Unary Math Ops ====================

    private SinOp parseSin(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new SinOp(result, operand, resultType);
    }

    private CosOp parseCos(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new CosOp(result, operand, resultType);
    }

    private TanOp parseTan(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new TanOp(result, operand, resultType);
    }

    private CeilOp parseCeil(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new CeilOp(result, operand, resultType);
    }

    private FloorOp parseFloor(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new FloorOp(result, operand, resultType);
    }

    private SignOp parseSign(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new SignOp(result, operand, resultType);
    }

    private LogisticOp parseLogistic(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new LogisticOp(result, operand, resultType);
    }

    private Expm1Op parseExpm1(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new Expm1Op(result, operand, resultType);
    }

    private Log1pOp parseLog1p(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new Log1pOp(result, operand, resultType);
    }

    private CbrtOp parseCbrt(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new CbrtOp(result, operand, resultType);
    }

    private IsFiniteOp parseIsFinite(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new IsFiniteOp(result, operand, resultType);
    }

    private PopcntOp parsePopcnt(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new PopcntOp(result, operand, resultType);
    }

    private ClzOp parseClz(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ClzOp(result, operand, resultType);
    }

    private RoundNearestEvenOp parseRoundNearestEven(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new RoundNearestEvenOp(result, operand, resultType);
    }

    private RoundNearestAfzOp parseRoundNearestAfz(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new RoundNearestAfzOp(result, operand, resultType);
    }

    private NotOp parseNot(String resultName) {
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new NotOp(result, operand, resultType);
    }

    // ==================== Additional Shape Ops ====================

    private ReverseOp parseReverse(String resultName) {
        // %r = stablehlo.reverse %op, dims = [0, 1] : tensor<...> -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        expect(TokenType.IDENTIFIER, "dims");
        expect(TokenType.EQUALS);
        List<Long> dimensions = parseIntegerList();
        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ReverseOp(result, operand, dimensions, resultType);
    }

    private PadOp parsePad(String resultName) {
        // %p = stablehlo.pad %op, %padding_value, low = [...], high = [...], interior = [...] : (t, t) -> t
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value paddingValue = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        expect(TokenType.IDENTIFIER, "low");
        expect(TokenType.EQUALS);
        List<Long> edgePaddingLow = parseIntegerList();

        expect(TokenType.COMMA);
        expect(TokenType.IDENTIFIER, "high");
        expect(TokenType.EQUALS);
        List<Long> edgePaddingHigh = parseIntegerList();

        expect(TokenType.COMMA);
        expect(TokenType.IDENTIFIER, "interior");
        expect(TokenType.EQUALS);
        List<Long> interiorPadding = parseIntegerList();

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new PadOp(result, operand, paddingValue, edgePaddingLow, edgePaddingHigh, interiorPadding, resultType);
    }

    private IotaOp parseIota(String resultName) {
        // %i = stablehlo.iota dim = 0 : tensor<4xf32>
        expect(TokenType.IDENTIFIER, "dim");
        expect(TokenType.EQUALS);
        long iotaDimension = Long.parseLong(expect(TokenType.INTEGER).value());
        expect(TokenType.COLON);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new IotaOp(result, iotaDimension, resultType);
    }

    private ReturnOp parseReturn() {
        advance(); // consume stablehlo.return

        List<Value> operands = new ArrayList<>();
        while (!check(TokenType.COLON)) {
            if (!operands.isEmpty()) {
                expect(TokenType.COMMA);
            }
            operands.add(lookupValue(parsePercentId()));
        }

        expect(TokenType.COLON);
        // Skip return types
        while (!check(TokenType.RBRACE) && !check(TokenType.EOF)) {
            if (check(TokenType.IDENTIFIER) && peek().value().equals("stablehlo.return")) {
                break;
            }
            advance();
        }

        return new ReturnOp(operands);
    }

    // ==================== Helpers ====================

    private String parsePercentId() {
        Token t = expect(TokenType.PERCENT_ID);
        return t.value().substring(1); // Remove leading %
    }

    private String parseAtId() {
        Token t = expect(TokenType.AT_ID);
        return t.value().substring(1); // Remove leading @
    }

    private String parseStringValue() {
        String str = expect(TokenType.STRING).value();
        // Remove surrounding quotes if present
        if (str.startsWith("\"") && str.endsWith("\"") && str.length() >= 2) {
            return str.substring(1, str.length() - 1);
        }
        return str;
    }

    private Value lookupValue(String name) {
        Value v = valueMap.get(name);
        if (v == null) {
            throw error("Undefined value: %" + name);
        }
        return v;
    }

    private Token peek() {
        return tokens.get(pos);
    }

    private Token advance() {
        return tokens.get(pos++);
    }

    private boolean check(TokenType type) {
        return peek().type() == type;
    }

    private boolean checkIdentifier(String value) {
        Token t = peek();
        return t.type() == TokenType.IDENTIFIER && t.value().equals(value);
    }

    private Token expect(TokenType type) {
        Token t = peek();
        if (t.type() != type) {
            throw error("Expected " + type + ", got " + t.type() + " (" + t.value() + ")");
        }
        return advance();
    }

    private Token expect(TokenType type, String value) {
        Token t = peek();
        if (t.type() != type || !t.value().equals(value)) {
            throw error("Expected " + type + "(" + value + "), got " + t.type() + "(" + t.value() + ")");
        }
        return advance();
    }

    private StableHloParseException error(String message) {
        Token t = peek();
        return new StableHloParseException(message, t.line(), t.column());
    }

    // ==================== Indexing Operations ====================

    private GatherOp parseGather(String resultName) {
        // %g = stablehlo.gather %operand, %start_indices, offset_dims=[...], collapsed_slice_dims=[...],
        //      start_index_map=[...], index_vector_dim=..., slice_sizes=[...] : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value startIndices = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        List<Long> offsetDims = new ArrayList<>();
        List<Long> collapsedSliceDims = new ArrayList<>();
        List<Long> startIndexMap = new ArrayList<>();
        long indexVectorDim = 0;
        List<Long> sliceSizes = new ArrayList<>();

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("offset_dims")) {
                advance();
                expect(TokenType.EQUALS);
                offsetDims = parseIntegerList();
            } else if (checkIdentifier("collapsed_slice_dims")) {
                advance();
                expect(TokenType.EQUALS);
                collapsedSliceDims = parseIntegerList();
            } else if (checkIdentifier("start_index_map")) {
                advance();
                expect(TokenType.EQUALS);
                startIndexMap = parseIntegerList();
            } else if (checkIdentifier("index_vector_dim")) {
                advance();
                expect(TokenType.EQUALS);
                indexVectorDim = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("slice_sizes")) {
                advance();
                expect(TokenType.EQUALS);
                sliceSizes = parseIntegerList();
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in gather: " + peek());
            }
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new GatherOp(result, operand, startIndices, offsetDims, collapsedSliceDims,
                startIndexMap, indexVectorDim, sliceSizes, resultType);
    }

    private ScatterOp parseScatter(String resultName) {
        // %s = stablehlo.scatter %operand, %scatter_indices, %updates, update_window_dims=[...],
        //      inserted_window_dims=[...], scatter_dims_to_operand_dims=[...], index_vector_dim=... : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value scatterIndices = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value updates = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        List<Long> updateWindowDims = new ArrayList<>();
        List<Long> insertedWindowDims = new ArrayList<>();
        List<Long> scatterDimsToOperandDims = new ArrayList<>();
        long indexVectorDim = 0;

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("update_window_dims")) {
                advance();
                expect(TokenType.EQUALS);
                updateWindowDims = parseIntegerList();
            } else if (checkIdentifier("inserted_window_dims")) {
                advance();
                expect(TokenType.EQUALS);
                insertedWindowDims = parseIntegerList();
            } else if (checkIdentifier("scatter_dims_to_operand_dims")) {
                advance();
                expect(TokenType.EQUALS);
                scatterDimsToOperandDims = parseIntegerList();
            } else if (checkIdentifier("index_vector_dim")) {
                advance();
                expect(TokenType.EQUALS);
                indexVectorDim = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in scatter: " + peek());
            }
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ScatterOp(result, operand, scatterIndices, updates, updateWindowDims,
                insertedWindowDims, scatterDimsToOperandDims, indexVectorDim, resultType);
    }

    private DynamicSliceOp parseDynamicSlice(String resultName) {
        // %d = stablehlo.dynamic_slice %operand, %start0, %start1, ..., sizes=[...] : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        List<Value> startIndices = new ArrayList<>();

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("sizes")) {
                break;
            }
            startIndices.add(lookupValue(parsePercentId()));
        }

        expect(TokenType.IDENTIFIER, "sizes");
        expect(TokenType.EQUALS);
        List<Long> sliceSizes = parseIntegerList();

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DynamicSliceOp(result, operand, startIndices, sliceSizes, resultType);
    }

    private DynamicUpdateSliceOp parseDynamicUpdateSlice(String resultName) {
        // %d = stablehlo.dynamic_update_slice %operand, %update, %start0, %start1, ... : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value update = lookupValue(parsePercentId());

        List<Value> startIndices = new ArrayList<>();
        while (check(TokenType.COMMA)) {
            advance();
            if (check(TokenType.COLON)) break;
            startIndices.add(lookupValue(parsePercentId()));
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DynamicUpdateSliceOp(result, operand, update, startIndices, resultType);
    }

    // ==================== Reduction Operations ====================

    private ReduceOp parseReduce(String resultName) {
        // %r = stablehlo.reduce %operand, %init, dims=[...], reducer=add : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value initValue = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        List<Long> dimensions = new ArrayList<>();
        String reducer = "add";

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("dims")) {
                advance();
                expect(TokenType.EQUALS);
                dimensions = parseIntegerList();
            } else if (checkIdentifier("reducer")) {
                advance();
                expect(TokenType.EQUALS);
                reducer = expect(TokenType.IDENTIFIER).value();
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in reduce: " + peek());
            }
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ReduceOp(result, operand, initValue, dimensions, reducer, resultType);
    }

    private ReduceWindowOp parseReduceWindow(String resultName) {
        // %r = stablehlo.reduce_window %operand, %init, window=[...], strides=[...], reducer=max : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value initValue = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        List<Long> windowDimensions = new ArrayList<>();
        List<Long> windowStrides = new ArrayList<>();
        List<Long> baseDilations = new ArrayList<>();
        List<Long> windowDilations = new ArrayList<>();
        List<Long> paddingLow = new ArrayList<>();
        List<Long> paddingHigh = new ArrayList<>();
        String reducer = "add";

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("window")) {
                advance();
                expect(TokenType.EQUALS);
                windowDimensions = parseIntegerList();
            } else if (checkIdentifier("strides")) {
                advance();
                expect(TokenType.EQUALS);
                windowStrides = parseIntegerList();
            } else if (checkIdentifier("base_dilations")) {
                advance();
                expect(TokenType.EQUALS);
                baseDilations = parseIntegerList();
            } else if (checkIdentifier("window_dilations")) {
                advance();
                expect(TokenType.EQUALS);
                windowDilations = parseIntegerList();
            } else if (checkIdentifier("padding_low")) {
                advance();
                expect(TokenType.EQUALS);
                paddingLow = parseIntegerList();
            } else if (checkIdentifier("padding_high")) {
                advance();
                expect(TokenType.EQUALS);
                paddingHigh = parseIntegerList();
            } else if (checkIdentifier("reducer")) {
                advance();
                expect(TokenType.EQUALS);
                reducer = expect(TokenType.IDENTIFIER).value();
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in reduce_window: " + peek());
            }
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ReduceWindowOp(result, operand, initValue, windowDimensions, windowStrides,
                baseDilations, windowDilations, paddingLow, paddingHigh, reducer, resultType);
    }

    // ==================== Neural Network Operations ====================

    private ConvolutionOp parseConvolution(String resultName) {
        // %c = stablehlo.convolution %lhs, %rhs, strides=[...], padding_low=[...], padding_high=[...],
        //      lhs_dilation=[...], rhs_dilation=[...], feature_group_count=1, batch_group_count=1,
        //      dimension_numbers=... : (...) -> tensor<...>
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        List<Long> windowStrides = new ArrayList<>();
        List<Long> paddingLow = new ArrayList<>();
        List<Long> paddingHigh = new ArrayList<>();
        List<Long> lhsDilation = new ArrayList<>();
        List<Long> rhsDilation = new ArrayList<>();
        long featureGroupCount = 1;
        long batchGroupCount = 1;
        // Dimension numbers
        long inputBatchDimension = 0;
        long inputFeatureDimension = 1;
        List<Long> inputSpatialDimensions = new ArrayList<>();
        long kernelInputFeatureDimension = 0;
        long kernelOutputFeatureDimension = 1;
        List<Long> kernelSpatialDimensions = new ArrayList<>();
        long outputBatchDimension = 0;
        long outputFeatureDimension = 1;
        List<Long> outputSpatialDimensions = new ArrayList<>();

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("strides")) {
                advance();
                expect(TokenType.EQUALS);
                windowStrides = parseIntegerList();
            } else if (checkIdentifier("padding_low")) {
                advance();
                expect(TokenType.EQUALS);
                paddingLow = parseIntegerList();
            } else if (checkIdentifier("padding_high")) {
                advance();
                expect(TokenType.EQUALS);
                paddingHigh = parseIntegerList();
            } else if (checkIdentifier("lhs_dilation")) {
                advance();
                expect(TokenType.EQUALS);
                lhsDilation = parseIntegerList();
            } else if (checkIdentifier("rhs_dilation")) {
                advance();
                expect(TokenType.EQUALS);
                rhsDilation = parseIntegerList();
            } else if (checkIdentifier("feature_group_count")) {
                advance();
                expect(TokenType.EQUALS);
                featureGroupCount = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("batch_group_count")) {
                advance();
                expect(TokenType.EQUALS);
                batchGroupCount = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("input_batch_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                inputBatchDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("input_feature_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                inputFeatureDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("input_spatial_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                inputSpatialDimensions = parseIntegerList();
            } else if (checkIdentifier("kernel_input_feature_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                kernelInputFeatureDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("kernel_output_feature_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                kernelOutputFeatureDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("kernel_spatial_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                kernelSpatialDimensions = parseIntegerList();
            } else if (checkIdentifier("output_batch_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                outputBatchDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("output_feature_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                outputFeatureDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("output_spatial_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                outputSpatialDimensions = parseIntegerList();
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in convolution: " + peek());
            }
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ConvolutionOp(result, lhs, rhs, windowStrides, paddingLow, paddingHigh,
                lhsDilation, rhsDilation, featureGroupCount, batchGroupCount,
                inputBatchDimension, inputFeatureDimension, inputSpatialDimensions,
                kernelInputFeatureDimension, kernelOutputFeatureDimension, kernelSpatialDimensions,
                outputBatchDimension, outputFeatureDimension, outputSpatialDimensions, resultType);
    }

    private BatchNormTrainingOp parseBatchNormTraining(String resultName) {
        // %out:3 = stablehlo.batch_norm_training %operand, %scale, %offset, epsilon=1e-5, feature_index=1 : (...) -> (...)
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value scale = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value offset = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        float epsilon = 1e-5f;
        long featureIndex = 1;

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("epsilon")) {
                advance();
                expect(TokenType.EQUALS);
                epsilon = Float.parseFloat(expect(TokenType.FLOAT).value());
            } else if (checkIdentifier("feature_index")) {
                advance();
                expect(TokenType.EQUALS);
                featureIndex = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in batch_norm_training: " + peek());
            }
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        // Result is a tuple (output, batch_mean, batch_var)
        expect(TokenType.LPAREN);
        TensorType resultType = parseTensorType();
        expect(TokenType.COMMA);
        TensorType meanType = parseTensorType();
        expect(TokenType.COMMA);
        TensorType varType = parseTensorType();
        expect(TokenType.RPAREN);

        Value output = new Value(resultName, resultType);
        Value batchMean = new Value(resultName + "_mean", meanType);
        Value batchVar = new Value(resultName + "_var", varType);
        valueMap.put(resultName, output);

        return new BatchNormTrainingOp(output, batchMean, batchVar, operand, scale, offset,
                epsilon, featureIndex, resultType);
    }

    private BatchNormInferenceOp parseBatchNormInference(String resultName) {
        // %out = stablehlo.batch_norm_inference %operand, %scale, %offset, %mean, %var, epsilon=1e-5, feature_index=1 : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value scale = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value offset = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value mean = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value variance = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        float epsilon = 1e-5f;
        long featureIndex = 1;

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("epsilon")) {
                advance();
                expect(TokenType.EQUALS);
                epsilon = Float.parseFloat(expect(TokenType.FLOAT).value());
            } else if (checkIdentifier("feature_index")) {
                advance();
                expect(TokenType.EQUALS);
                featureIndex = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in batch_norm_inference: " + peek());
            }
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new BatchNormInferenceOp(result, operand, scale, offset, mean, variance,
                epsilon, featureIndex, resultType);
    }

    // ==================== Sort and RNG ====================

    private SortOp parseSort(String resultName) {
        // %s = stablehlo.sort %input, dim=0, stable=true : (...) -> tensor<...>
        List<Value> inputs = new ArrayList<>();
        inputs.add(lookupValue(parsePercentId()));

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("dim") || checkIdentifier("stable")) {
                break;
            }
            inputs.add(lookupValue(parsePercentId()));
        }

        long dimension = 0;
        boolean isStable = false;

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("dim")) {
                advance();
                expect(TokenType.EQUALS);
                dimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("stable")) {
                advance();
                expect(TokenType.EQUALS);
                isStable = expect(TokenType.IDENTIFIER).value().equals("true");
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in sort: " + peek());
            }
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new SortOp(result, inputs, dimension, isStable, resultType);
    }

    private RngOp parseRng(String resultName) {
        // %r = stablehlo.rng %a, %b, distribution=uniform : (...) -> tensor<...>
        Value a = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value b = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        String distribution = "uniform";

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("distribution")) {
                advance();
                expect(TokenType.EQUALS);
                distribution = expect(TokenType.IDENTIFIER).value();
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in rng: " + peek());
            }
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new RngOp(result, a, b, distribution, resultType);
    }

    private RngBitGeneratorOp parseRngBitGenerator(String resultName) {
        // %r:2 = stablehlo.rng_bit_generator %state, algorithm=three_fry : (...) -> (...)
        Value initialState = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        RngAlgorithm algorithm = RngAlgorithm.DEFAULT;

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("algorithm")) {
                advance();
                expect(TokenType.EQUALS);
                algorithm = RngAlgorithm.fromString(expect(TokenType.IDENTIFIER).value());
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in rng_bit_generator: " + peek());
            }
        }

        expect(TokenType.COLON);
        // Skip input type
        parseType();
        expect(TokenType.ARROW);
        // Result is a tuple (output_state, output)
        expect(TokenType.LPAREN);
        TensorType stateType = parseTensorType();
        expect(TokenType.COMMA);
        TensorType outputType = parseTensorType();
        expect(TokenType.RPAREN);

        Value outputState = new Value(resultName + "_state", stateType);
        Value output = new Value(resultName, outputType);
        valueMap.put(resultName, output);

        return new RngBitGeneratorOp(outputState, output, initialState, algorithm, outputType);
    }

    // ==================== Additional Operations ====================

    private DotOp parseDot(String resultName) {
        // %d = stablehlo.dot %lhs, %rhs : (t, t) -> tensor<...>
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DotOp(result, lhs, rhs, resultType);
    }

    private RealOp parseReal(String resultName) {
        // %r = stablehlo.real %complex : tensor<...xcomplex<f32>> -> tensor<...xf32>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new RealOp(result, operand, resultType);
    }

    private ImagOp parseImag(String resultName) {
        // %i = stablehlo.imag %complex : tensor<...xcomplex<f32>> -> tensor<...xf32>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ImagOp(result, operand, resultType);
    }

    private ComplexOp parseComplex(String resultName) {
        // %c = stablehlo.complex %real, %imag : (t, t) -> tensor<...xcomplex<f32>>
        Value real = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value imag = lookupValue(parsePercentId());

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ComplexOp(result, real, imag, resultType);
    }

    private FftOp parseFft(String resultName) {
        // %f = stablehlo.fft %operand, type=FFT, length=[...] : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);

        String fftType = "FFT";
        List<Long> fftLength = new ArrayList<>();

        while (!check(TokenType.COLON)) {
            if (checkIdentifier("type")) {
                advance();
                expect(TokenType.EQUALS);
                fftType = expect(TokenType.IDENTIFIER).value();
            } else if (checkIdentifier("length")) {
                advance();
                expect(TokenType.EQUALS);
                fftLength = parseIntegerList();
            } else if (check(TokenType.COMMA)) {
                advance();
            } else {
                throw error("Unexpected token in fft: " + peek());
            }
        }

        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new FftOp(result, operand, fftType, fftLength, resultType);
    }

    private CholeskyOp parseCholesky(String resultName) {
        // %c = stablehlo.cholesky %a, lower=true : tensor<...> -> tensor<...>
        Value operand = lookupValue(parsePercentId());

        boolean lower = true;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("lower")) {
                advance();
                expect(TokenType.EQUALS);
                lower = expect(TokenType.IDENTIFIER).value().equals("true");
            }
        }

        expect(TokenType.COLON);
        parseType(); // input type
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new CholeskyOp(result, operand, lower, resultType);
    }

    private TriangularSolveOp parseTriangularSolve(String resultName) {
        // %x = stablehlo.triangular_solve %a, %b, left_side=true, lower=true, transpose_a=false : (...) -> tensor<...>
        Value a = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value b = lookupValue(parsePercentId());

        boolean leftSide = true;
        boolean lower = true;
        boolean transposeA = false;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("left_side")) {
                advance();
                expect(TokenType.EQUALS);
                leftSide = expect(TokenType.IDENTIFIER).value().equals("true");
            } else if (checkIdentifier("lower")) {
                advance();
                expect(TokenType.EQUALS);
                lower = expect(TokenType.IDENTIFIER).value().equals("true");
            } else if (checkIdentifier("transpose_a")) {
                advance();
                expect(TokenType.EQUALS);
                transposeA = expect(TokenType.IDENTIFIER).value().equals("true");
            }
        }

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new TriangularSolveOp(result, a, b, leftSide, lower, transposeA, resultType);
    }

    private CustomCallOp parseCustomCall(String resultName) {
        // %c = stablehlo.custom_call @target(%arg0, %arg1) : (...) -> tensor<...>
        String callTarget = parseAtId();
        expect(TokenType.LPAREN);

        List<Value> inputs = new ArrayList<>();
        while (!check(TokenType.RPAREN)) {
            if (!inputs.isEmpty()) {
                expect(TokenType.COMMA);
            }
            inputs.add(lookupValue(parsePercentId()));
        }
        expect(TokenType.RPAREN);

        expect(TokenType.COLON);
        // Skip input types
        expect(TokenType.LPAREN);
        while (!check(TokenType.RPAREN)) {
            parseType();
            if (check(TokenType.COMMA)) advance();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new CustomCallOp(result, callTarget, inputs, resultType);
    }

    // ==================== Dynamic Shape Operations ====================

    private DynamicBroadcastInDimOp parseDynamicBroadcastInDim(String resultName) {
        // %b = stablehlo.dynamic_broadcast_in_dim %operand, %output_dimensions, broadcast_dimensions=[...] : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value outputDimensions = lookupValue(parsePercentId());

        List<Long> broadcastDimensions = new ArrayList<>();
        List<Long> knownExpandingDimensions = new ArrayList<>();
        List<Long> knownNonexpandingDimensions = new ArrayList<>();

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("broadcast_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                broadcastDimensions = parseIntegerList();
            } else if (checkIdentifier("known_expanding_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                knownExpandingDimensions = parseIntegerList();
            } else if (checkIdentifier("known_nonexpanding_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                knownNonexpandingDimensions = parseIntegerList();
            }
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DynamicBroadcastInDimOp(result, operand, outputDimensions, broadcastDimensions,
                knownExpandingDimensions, knownNonexpandingDimensions, resultType);
    }

    private DynamicGatherOp parseDynamicGather(String resultName) {
        // %g = stablehlo.dynamic_gather %operand, %start_indices, %slice_sizes, offset_dims=[...], ... : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value startIndices = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value sliceSizes = lookupValue(parsePercentId());

        List<Long> offsetDims = new ArrayList<>();
        List<Long> collapsedSliceDims = new ArrayList<>();
        List<Long> startIndexMap = new ArrayList<>();
        long indexVectorDim = 0;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("offset_dims")) {
                advance();
                expect(TokenType.EQUALS);
                offsetDims = parseIntegerList();
            } else if (checkIdentifier("collapsed_slice_dims")) {
                advance();
                expect(TokenType.EQUALS);
                collapsedSliceDims = parseIntegerList();
            } else if (checkIdentifier("start_index_map")) {
                advance();
                expect(TokenType.EQUALS);
                startIndexMap = parseIntegerList();
            } else if (checkIdentifier("index_vector_dim")) {
                advance();
                expect(TokenType.EQUALS);
                indexVectorDim = Long.parseLong(expect(TokenType.INTEGER).value());
            }
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DynamicGatherOp(result, operand, startIndices, sliceSizes, offsetDims,
                collapsedSliceDims, startIndexMap, indexVectorDim, resultType);
    }

    private DynamicIotaOp parseDynamicIota(String resultName) {
        // %i = stablehlo.dynamic_iota %output_shape, dim = 0 : (...) -> tensor<...>
        Value outputShape = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        expect(TokenType.IDENTIFIER, "dim");
        expect(TokenType.EQUALS);
        long iotaDimension = Long.parseLong(expect(TokenType.INTEGER).value());

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DynamicIotaOp(result, outputShape, iotaDimension, resultType);
    }

    private DynamicPadOp parseDynamicPad(String resultName) {
        // %p = stablehlo.dynamic_pad %operand, %padding_value, %low, %high, %interior : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value paddingValue = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value edgePaddingLow = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value edgePaddingHigh = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value interiorPadding = lookupValue(parsePercentId());

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DynamicPadOp(result, operand, paddingValue, edgePaddingLow, edgePaddingHigh, interiorPadding, resultType);
    }

    private DynamicReshapeOp parseDynamicReshape(String resultName) {
        // %r = stablehlo.dynamic_reshape %operand, %output_shape : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value outputShape = lookupValue(parsePercentId());

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.COMMA);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DynamicReshapeOp(result, operand, outputShape, resultType);
    }

    private DynamicConvOp parseDynamicConv(String resultName) {
        // Similar to convolution but with dynamic padding tensor
        Value lhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value rhs = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value padding = lookupValue(parsePercentId());

        List<Long> windowStrides = new ArrayList<>();
        List<Long> lhsDilation = new ArrayList<>();
        List<Long> rhsDilation = new ArrayList<>();
        long featureGroupCount = 1;
        long batchGroupCount = 1;
        long inputBatchDimension = 0;
        long inputFeatureDimension = 1;
        List<Long> inputSpatialDimensions = new ArrayList<>();
        long kernelInputFeatureDimension = 0;
        long kernelOutputFeatureDimension = 1;
        List<Long> kernelSpatialDimensions = new ArrayList<>();
        long outputBatchDimension = 0;
        long outputFeatureDimension = 1;
        List<Long> outputSpatialDimensions = new ArrayList<>();

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("strides")) {
                advance();
                expect(TokenType.EQUALS);
                windowStrides = parseIntegerList();
            } else if (checkIdentifier("lhs_dilation")) {
                advance();
                expect(TokenType.EQUALS);
                lhsDilation = parseIntegerList();
            } else if (checkIdentifier("rhs_dilation")) {
                advance();
                expect(TokenType.EQUALS);
                rhsDilation = parseIntegerList();
            } else if (checkIdentifier("feature_group_count")) {
                advance();
                expect(TokenType.EQUALS);
                featureGroupCount = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("batch_group_count")) {
                advance();
                expect(TokenType.EQUALS);
                batchGroupCount = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("input_batch_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                inputBatchDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("input_feature_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                inputFeatureDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("input_spatial_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                inputSpatialDimensions = parseIntegerList();
            } else if (checkIdentifier("kernel_input_feature_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                kernelInputFeatureDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("kernel_output_feature_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                kernelOutputFeatureDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("kernel_spatial_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                kernelSpatialDimensions = parseIntegerList();
            } else if (checkIdentifier("output_batch_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                outputBatchDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("output_feature_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                outputFeatureDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("output_spatial_dimensions")) {
                advance();
                expect(TokenType.EQUALS);
                outputSpatialDimensions = parseIntegerList();
            }
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new DynamicConvOp(result, lhs, rhs, padding, windowStrides, lhsDilation, rhsDilation,
                featureGroupCount, batchGroupCount, inputBatchDimension, inputFeatureDimension,
                inputSpatialDimensions, kernelInputFeatureDimension, kernelOutputFeatureDimension,
                kernelSpatialDimensions, outputBatchDimension, outputFeatureDimension, outputSpatialDimensions, resultType);
    }

    private GetDimensionSizeOp parseGetDimensionSize(String resultName) {
        // %s = stablehlo.get_dimension_size %operand, dim = 0 : (...) -> tensor<i32>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        expect(TokenType.IDENTIFIER, "dim");
        expect(TokenType.EQUALS);
        long dimension = Long.parseLong(expect(TokenType.INTEGER).value());

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new GetDimensionSizeOp(result, operand, dimension, resultType);
    }

    // ==================== Quantization Operations ====================

    private UniformQuantizeOp parseUniformQuantize(String resultName) {
        // %q = stablehlo.uniform_quantize %operand : tensor<...xf32> -> tensor<...x!quant.uniform<...>>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new UniformQuantizeOp(result, operand, resultType);
    }

    private UniformDequantizeOp parseUniformDequantize(String resultName) {
        // %d = stablehlo.uniform_dequantize %operand : tensor<...x!quant.uniform<...>> -> tensor<...xf32>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new UniformDequantizeOp(result, operand, resultType);
    }

    // ==================== Additional Reduction Operations ====================

    private ReducePrecisionOp parseReducePrecision(String resultName) {
        // %r = stablehlo.reduce_precision %operand, exponent_bits=5, mantissa_bits=10 : tensor<...> -> tensor<...>
        Value operand = lookupValue(parsePercentId());

        int exponentBits = 5;
        int mantissaBits = 10;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("exponent_bits")) {
                advance();
                expect(TokenType.EQUALS);
                exponentBits = Integer.parseInt(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("mantissa_bits")) {
                advance();
                expect(TokenType.EQUALS);
                mantissaBits = Integer.parseInt(expect(TokenType.INTEGER).value());
            }
        }

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ReducePrecisionOp(result, operand, exponentBits, mantissaBits, resultType);
    }

    private SelectAndScatterOp parseSelectAndScatter(String resultName) {
        // %s = stablehlo.select_and_scatter %operand, %source, %init, window=[...], strides=[...], padding=[...], select=ge, scatter=add : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value source = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value initValue = lookupValue(parsePercentId());

        List<Long> windowDimensions = new ArrayList<>();
        List<Long> windowStrides = new ArrayList<>();
        List<Long> padding = new ArrayList<>();
        String selectFn = "ge";
        String scatterFn = "add";

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("window")) {
                advance();
                expect(TokenType.EQUALS);
                windowDimensions = parseIntegerList();
            } else if (checkIdentifier("strides")) {
                advance();
                expect(TokenType.EQUALS);
                windowStrides = parseIntegerList();
            } else if (checkIdentifier("padding")) {
                advance();
                expect(TokenType.EQUALS);
                padding = parseIntegerList();
            } else if (checkIdentifier("select")) {
                advance();
                expect(TokenType.EQUALS);
                selectFn = expect(TokenType.IDENTIFIER).value();
            } else if (checkIdentifier("scatter")) {
                advance();
                expect(TokenType.EQUALS);
                scatterFn = expect(TokenType.IDENTIFIER).value();
            }
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new SelectAndScatterOp(result, operand, source, initValue, windowDimensions,
                windowStrides, padding, selectFn, scatterFn, resultType);
    }

    // ==================== Additional Neural Network Operations ====================

    private BatchNormGradOp parseBatchNormGrad(String resultName) {
        // %grad:3 = stablehlo.batch_norm_grad %operand, %scale, %mean, %variance, %grad_output, epsilon=1e-5, feature_index=1 : (...) -> (...)
        Value operand = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value scale = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value mean = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value variance = lookupValue(parsePercentId());
        expect(TokenType.COMMA);
        Value gradOutput = lookupValue(parsePercentId());

        float epsilon = 1e-5f;
        long featureIndex = 1;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("epsilon")) {
                advance();
                expect(TokenType.EQUALS);
                epsilon = Float.parseFloat(expect(TokenType.FLOAT).value());
            } else if (checkIdentifier("feature_index")) {
                advance();
                expect(TokenType.EQUALS);
                featureIndex = Long.parseLong(expect(TokenType.INTEGER).value());
            }
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        expect(TokenType.LPAREN);
        TensorType gradOperandType = parseTensorType();
        expect(TokenType.COMMA);
        TensorType gradScaleType = parseTensorType();
        expect(TokenType.COMMA);
        TensorType gradOffsetType = parseTensorType();
        expect(TokenType.RPAREN);

        Value gradOperand = new Value(resultName, gradOperandType);
        Value gradScale = new Value(resultName + "_scale", gradScaleType);
        Value gradOffset = new Value(resultName + "_offset", gradOffsetType);
        valueMap.put(resultName, gradOperand);

        return new BatchNormGradOp(gradOperand, gradScale, gradOffset, operand, scale, mean, variance,
                gradOutput, epsilon, featureIndex, gradOperandType);
    }

    // ==================== Control Flow Operations ====================

    private CaseOp parseCase(String resultName) {
        // %r = stablehlo.case %index { ... } { ... } : (i32) -> tensor<...>
        Value index = lookupValue(parsePercentId());
        List<List<Operation>> branches = new ArrayList<>();

        // Parse branches (each is a region in { })
        while (check(TokenType.LBRACE)) {
            advance();
            List<Operation> branch = new ArrayList<>();
            while (!check(TokenType.RBRACE)) {
                branch.add(parseOperation());
            }
            expect(TokenType.RBRACE);
            branches.add(branch);
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new CaseOp(List.of(result), index, branches, resultType);
    }

    private MapOp parseMap(String resultName) {
        // %m = stablehlo.map %inputs, dims=[...] { computation } : (...) -> tensor<...>
        List<Value> inputs = new ArrayList<>();
        inputs.add(lookupValue(parsePercentId()));

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("dims")) {
                break;
            }
            inputs.add(lookupValue(parsePercentId()));
        }

        List<Long> dimensions = new ArrayList<>();
        if (checkIdentifier("dims")) {
            advance();
            expect(TokenType.EQUALS);
            dimensions = parseIntegerList();
        }

        List<Operation> computation = new ArrayList<>();
        if (check(TokenType.LBRACE)) {
            advance();
            while (!check(TokenType.RBRACE)) {
                computation.add(parseOperation());
            }
            expect(TokenType.RBRACE);
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new MapOp(result, inputs, dimensions, computation, resultType);
    }

    // ==================== Distributed/Collective Operations ====================

    private AfterAllOp parseAfterAll(String resultName) {
        // %token = stablehlo.after_all %token1, %token2 : (!stablehlo.token, ...) -> !stablehlo.token
        List<Value> inputs = new ArrayList<>();
        if (!check(TokenType.COLON)) {
            inputs.add(lookupValue(parsePercentId()));
            while (check(TokenType.COMMA)) {
                advance();
                if (check(TokenType.COLON)) break;
                inputs.add(lookupValue(parsePercentId()));
            }
        }

        expect(TokenType.COLON);
        // Skip types
        while (!check(TokenType.ARROW) && !check(TokenType.EOF)) {
            advance();
        }
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new AfterAllOp(result, inputs, resultType);
    }

    private AllGatherOp parseAllGather(String resultName) {
        // %g = stablehlo.all_gather %operand, all_gather_dim=0, replica_groups=[[0,1],[2,3]], channel_id=1 : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());

        long allGatherDim = 0;
        List<List<Long>> replicaGroups = new ArrayList<>();
        long channelId = 0;
        boolean useGlobalDeviceIds = false;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("all_gather_dim")) {
                advance();
                expect(TokenType.EQUALS);
                allGatherDim = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("replica_groups")) {
                advance();
                expect(TokenType.EQUALS);
                replicaGroups = parseNestedIntegerList();
            } else if (checkIdentifier("channel_id")) {
                advance();
                expect(TokenType.EQUALS);
                channelId = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("use_global_device_ids")) {
                advance();
                expect(TokenType.EQUALS);
                useGlobalDeviceIds = expect(TokenType.IDENTIFIER).value().equals("true");
            }
        }

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new AllGatherOp(result, operand, allGatherDim, replicaGroups, channelId, useGlobalDeviceIds, resultType);
    }

    private AllReduceOp parseAllReduce(String resultName) {
        // %r = stablehlo.all_reduce %operand, replica_groups=[[0,1]], channel_id=1, reducer=add : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());

        List<List<Long>> replicaGroups = new ArrayList<>();
        long channelId = 0;
        boolean useGlobalDeviceIds = false;
        String reducer = "add";

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("replica_groups")) {
                advance();
                expect(TokenType.EQUALS);
                replicaGroups = parseNestedIntegerList();
            } else if (checkIdentifier("channel_id")) {
                advance();
                expect(TokenType.EQUALS);
                channelId = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("use_global_device_ids")) {
                advance();
                expect(TokenType.EQUALS);
                useGlobalDeviceIds = expect(TokenType.IDENTIFIER).value().equals("true");
            } else if (checkIdentifier("reducer")) {
                advance();
                expect(TokenType.EQUALS);
                reducer = expect(TokenType.IDENTIFIER).value();
            }
        }

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new AllReduceOp(result, operand, replicaGroups, channelId, useGlobalDeviceIds, reducer, resultType);
    }

    private AllToAllOp parseAllToAll(String resultName) {
        // %a = stablehlo.all_to_all %operand, split_dimension=0, concat_dimension=1, split_count=2, replica_groups=[[...]] : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());

        long splitDimension = 0;
        long concatDimension = 0;
        long splitCount = 1;
        List<List<Long>> replicaGroups = new ArrayList<>();
        long channelId = 0;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("split_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                splitDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("concat_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                concatDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("split_count")) {
                advance();
                expect(TokenType.EQUALS);
                splitCount = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("replica_groups")) {
                advance();
                expect(TokenType.EQUALS);
                replicaGroups = parseNestedIntegerList();
            } else if (checkIdentifier("channel_id")) {
                advance();
                expect(TokenType.EQUALS);
                channelId = Long.parseLong(expect(TokenType.INTEGER).value());
            }
        }

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new AllToAllOp(result, operand, splitDimension, concatDimension, splitCount, replicaGroups, channelId, resultType);
    }

    private CollectiveBroadcastOp parseCollectiveBroadcast(String resultName) {
        // %b = stablehlo.collective_broadcast %operand, replica_groups=[[...]], channel_id=1 : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());

        List<List<Long>> replicaGroups = new ArrayList<>();
        long channelId = 0;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("replica_groups")) {
                advance();
                expect(TokenType.EQUALS);
                replicaGroups = parseNestedIntegerList();
            } else if (checkIdentifier("channel_id")) {
                advance();
                expect(TokenType.EQUALS);
                channelId = Long.parseLong(expect(TokenType.INTEGER).value());
            }
        }

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new CollectiveBroadcastOp(result, operand, replicaGroups, channelId, resultType);
    }

    private CollectivePermuteOp parseCollectivePermute(String resultName) {
        // %p = stablehlo.collective_permute %operand, source_target_pairs=[[0,1],[1,0]], channel_id=1 : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());

        List<List<Long>> sourceTargetPairs = new ArrayList<>();
        long channelId = 0;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("source_target_pairs")) {
                advance();
                expect(TokenType.EQUALS);
                sourceTargetPairs = parseNestedIntegerList();
            } else if (checkIdentifier("channel_id")) {
                advance();
                expect(TokenType.EQUALS);
                channelId = Long.parseLong(expect(TokenType.INTEGER).value());
            }
        }

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new CollectivePermuteOp(result, operand, sourceTargetPairs, channelId, resultType);
    }

    private PartitionIdOp parsePartitionId(String resultName) {
        // %p = stablehlo.partition_id : () -> tensor<ui32>
        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new PartitionIdOp(result, resultType);
    }

    private ReduceScatterOp parseReduceScatter(String resultName) {
        // %r = stablehlo.reduce_scatter %operand, scatter_dimension=0, replica_groups=[[...]], channel_id=1, reducer=add : (...) -> tensor<...>
        Value operand = lookupValue(parsePercentId());

        long scatterDimension = 0;
        List<List<Long>> replicaGroups = new ArrayList<>();
        long channelId = 0;
        boolean useGlobalDeviceIds = false;
        String reducer = "add";

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("scatter_dimension")) {
                advance();
                expect(TokenType.EQUALS);
                scatterDimension = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("replica_groups")) {
                advance();
                expect(TokenType.EQUALS);
                replicaGroups = parseNestedIntegerList();
            } else if (checkIdentifier("channel_id")) {
                advance();
                expect(TokenType.EQUALS);
                channelId = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("use_global_device_ids")) {
                advance();
                expect(TokenType.EQUALS);
                useGlobalDeviceIds = expect(TokenType.IDENTIFIER).value().equals("true");
            } else if (checkIdentifier("reducer")) {
                advance();
                expect(TokenType.EQUALS);
                reducer = expect(TokenType.IDENTIFIER).value();
            }
        }

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ReduceScatterOp(result, operand, scatterDimension, replicaGroups, channelId, useGlobalDeviceIds, reducer, resultType);
    }

    private ReplicaIdOp parseReplicaId(String resultName) {
        // %r = stablehlo.replica_id : () -> tensor<ui32>
        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new ReplicaIdOp(result, resultType);
    }

    // Helper for parsing nested integer lists like [[0,1],[2,3]]
    private List<List<Long>> parseNestedIntegerList() {
        expect(TokenType.LBRACKET);
        List<List<Long>> result = new ArrayList<>();
        while (!check(TokenType.RBRACKET)) {
            if (!result.isEmpty()) {
                expect(TokenType.COMMA);
            }
            result.add(parseIntegerList());
        }
        expect(TokenType.RBRACKET);
        return result;
    }

    // ==================== Communication Operations ====================

    private InfeedOp parseInfeed(String resultName) {
        // %data, %token = stablehlo.infeed %token, infeed_config="..." : (...) -> (tensor<...>, !stablehlo.token)
        Value token = lookupValue(parsePercentId());

        String infeedConfig = "";

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("infeed_config")) {
                advance();
                expect(TokenType.EQUALS);
                infeedConfig = parseStringValue();
            }
        }

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        expect(TokenType.LPAREN);
        TensorType dataType = parseTensorType();
        expect(TokenType.COMMA);
        parseType(); // token type
        expect(TokenType.RPAREN);

        Value data = new Value(resultName, dataType);
        Value outToken = new Value(resultName + "_token", null);
        valueMap.put(resultName, data);

        return new InfeedOp(List.of(data, outToken), token, infeedConfig, dataType);
    }

    private OutfeedOp parseOutfeed(String resultName) {
        // %token = stablehlo.outfeed %data, %token, outfeed_config="..." : (...) -> !stablehlo.token
        List<Value> inputs = new ArrayList<>();
        inputs.add(lookupValue(parsePercentId()));
        expect(TokenType.COMMA);
        Value token = lookupValue(parsePercentId());

        String outfeedConfig = "";

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("outfeed_config")) {
                advance();
                expect(TokenType.EQUALS);
                outfeedConfig = parseStringValue();
            }
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new OutfeedOp(result, inputs, token, outfeedConfig, resultType);
    }

    private RecvOp parseRecv(String resultName) {
        // %data, %token = stablehlo.recv %token, channel_id=1, channel_type=device_to_device : (...) -> (tensor<...>, !stablehlo.token)
        Value token = lookupValue(parsePercentId());

        long channelId = 0;
        String channelType = "device_to_device";

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("channel_id")) {
                advance();
                expect(TokenType.EQUALS);
                channelId = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("channel_type")) {
                advance();
                expect(TokenType.EQUALS);
                channelType = expect(TokenType.IDENTIFIER).value();
            }
        }

        expect(TokenType.COLON);
        parseType();
        expect(TokenType.ARROW);
        expect(TokenType.LPAREN);
        TensorType dataType = parseTensorType();
        expect(TokenType.COMMA);
        parseType(); // token type
        expect(TokenType.RPAREN);

        Value data = new Value(resultName, dataType);
        Value outToken = new Value(resultName + "_token", null);
        valueMap.put(resultName, data);

        return new RecvOp(List.of(data, outToken), token, channelId, channelType, dataType);
    }

    private SendOp parseSend(String resultName) {
        // %token = stablehlo.send %data, %token, channel_id=1, channel_type=device_to_device : (...) -> !stablehlo.token
        List<Value> inputs = new ArrayList<>();
        inputs.add(lookupValue(parsePercentId()));
        expect(TokenType.COMMA);
        Value token = lookupValue(parsePercentId());

        long channelId = 0;
        String channelType = "device_to_device";

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("channel_id")) {
                advance();
                expect(TokenType.EQUALS);
                channelId = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (checkIdentifier("channel_type")) {
                advance();
                expect(TokenType.EQUALS);
                channelType = expect(TokenType.IDENTIFIER).value();
            }
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new SendOp(result, inputs, token, channelId, channelType, resultType);
    }

    // ==================== Tuple Operations ====================

    private TupleOp parseTuple(String resultName) {
        // %t = stablehlo.tuple %a, %b, %c : (tensor<...>, ...) -> tuple<...>
        List<Value> inputs = new ArrayList<>();
        inputs.add(lookupValue(parsePercentId()));

        while (check(TokenType.COMMA)) {
            advance();
            if (check(TokenType.COLON)) break;
            inputs.add(lookupValue(parsePercentId()));
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new TupleOp(result, inputs, resultType);
    }

    private GetTupleElementOp parseGetTupleElement(String resultName) {
        // %e = stablehlo.get_tuple_element %tuple, index=0 : (tuple<...>) -> tensor<...>
        Value operand = lookupValue(parsePercentId());

        int index = 0;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("index")) {
                advance();
                expect(TokenType.EQUALS);
                index = Integer.parseInt(expect(TokenType.INTEGER).value());
            }
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new GetTupleElementOp(result, operand, index, resultType);
    }

    // ==================== Other Operations ====================

    private OptimizationBarrierOp parseOptimizationBarrier(String resultName) {
        // %r = stablehlo.optimization_barrier %a, %b : (tensor<...>, ...) -> (tensor<...>, ...)
        List<Value> operands = new ArrayList<>();
        operands.add(lookupValue(parsePercentId()));

        while (check(TokenType.COMMA)) {
            advance();
            if (check(TokenType.COLON)) break;
            operands.add(lookupValue(parsePercentId()));
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        expect(TokenType.LPAREN);
        TensorType resultType = parseTensorType();
        List<Value> results = new ArrayList<>();
        results.add(new Value(resultName, resultType));
        while (check(TokenType.COMMA)) {
            advance();
            results.add(new Value(resultName + "_" + results.size(), parseTensorType()));
        }
        expect(TokenType.RPAREN);

        valueMap.put(resultName, results.get(0));

        return new OptimizationBarrierOp(results, operands, resultType);
    }

    private CompositeOp parseComposite(String resultName) {
        // %c = stablehlo.composite %a, %b, name="my_op", composite_attributes=..., decomposition=@decomp, version=1 : (...) -> tensor<...>
        List<Value> inputs = new ArrayList<>();
        inputs.add(lookupValue(parsePercentId()));

        String name = "";
        String compositeAttributes = "";
        String decomposition = "";
        long version = 0;

        while (check(TokenType.COMMA)) {
            advance();
            if (checkIdentifier("name")) {
                advance();
                expect(TokenType.EQUALS);
                name = parseStringValue();
            } else if (checkIdentifier("composite_attributes")) {
                advance();
                expect(TokenType.EQUALS);
                compositeAttributes = parseStringValue();
            } else if (checkIdentifier("decomposition")) {
                advance();
                expect(TokenType.EQUALS);
                decomposition = parseAtId();
            } else if (checkIdentifier("version")) {
                advance();
                expect(TokenType.EQUALS);
                version = Long.parseLong(expect(TokenType.INTEGER).value());
            } else if (!check(TokenType.COLON)) {
                inputs.add(lookupValue(parsePercentId()));
            }
        }

        expect(TokenType.COLON);
        expect(TokenType.LPAREN);
        parseType();
        while (check(TokenType.COMMA)) {
            advance();
            parseType();
        }
        expect(TokenType.RPAREN);
        expect(TokenType.ARROW);
        TensorType resultType = parseTensorType();

        Value result = new Value(resultName, resultType);
        valueMap.put(resultName, result);

        return new CompositeOp(result, inputs, name, compositeAttributes, decomposition, version, resultType);
    }
}
