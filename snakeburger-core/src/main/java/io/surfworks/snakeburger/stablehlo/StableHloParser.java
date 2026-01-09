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
            // Conditional ops
            case "stablehlo.compare" -> parseCompare(resultName);
            case "stablehlo.select" -> parseSelect(resultName);
            case "stablehlo.clamp" -> parseClamp(resultName);
            // Type conversion
            case "stablehlo.convert" -> parseConvert(resultName);
            case "stablehlo.bitcast_convert" -> parseBitcastConvert(resultName);
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
}
