package io.surfworks.snakeburger.stablehlo;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tokenizer for StableHLO MLIR text format.
 *
 * Recognizes:
 * - Identifiers: module, func.func, stablehlo.dot_general, tensor, etc.
 * - Sigils: %, @, #
 * - Punctuation: ( ) { } < > [ ] : , = ->
 * - Numeric literals: integers and floats
 * - String literals: "..."
 * - Attributes: #stablehlo.dot<...>
 */
public final class StableHloTokenizer {

    public enum TokenType {
        // Keywords and identifiers
        IDENTIFIER,      // module, func.func, stablehlo.dot_general, tensor, public

        // Value references
        PERCENT_ID,      // %arg0, %1, %2_zero
        AT_ID,           // @main, @mlp_layer
        HASH_ID,         // #stablehlo.dot

        // Literals
        INTEGER,         // 0, 1, 4, 8, 16
        FLOAT,           // 0.0, 1.5
        STRING,          // "..."

        // Punctuation
        LPAREN,          // (
        RPAREN,          // )
        LBRACE,          // {
        RBRACE,          // }
        LANGLE,          // <
        RANGLE,          // >
        LBRACKET,        // [
        RBRACKET,        // ]
        COLON,           // :
        COMMA,           // ,
        EQUALS,          // =
        ARROW,           // ->

        // Special
        NEWLINE,         // for line tracking
        EOF
    }

    public record Token(TokenType type, String value, int line, int column) {
        @Override
        public String toString() {
            return String.format("%s(%s)@%d:%d", type, value, line, column);
        }
    }

    private final String input;
    private int pos;
    private int line;
    private int column;
    private int lineStart;

    public StableHloTokenizer(String input) {
        this.input = input;
        this.pos = 0;
        this.line = 1;
        this.column = 1;
        this.lineStart = 0;
    }

    public List<Token> tokenize() {
        List<Token> tokens = new ArrayList<>();

        while (pos < input.length()) {
            skipWhitespaceAndComments();
            if (pos >= input.length()) break;

            Token token = nextToken();
            if (token != null) {
                tokens.add(token);
            }
        }

        tokens.add(new Token(TokenType.EOF, "", line, column));
        return tokens;
    }

    private void skipWhitespaceAndComments() {
        while (pos < input.length()) {
            char c = input.charAt(pos);

            if (c == ' ' || c == '\t' || c == '\r') {
                pos++;
                column++;
            } else if (c == '\n') {
                pos++;
                line++;
                column = 1;
                lineStart = pos;
            } else if (c == '/' && pos + 1 < input.length() && input.charAt(pos + 1) == '/') {
                // Skip line comment
                while (pos < input.length() && input.charAt(pos) != '\n') {
                    pos++;
                }
            } else {
                break;
            }
        }
    }

    private Token nextToken() {
        if (pos >= input.length()) {
            return null;
        }

        int startLine = line;
        int startCol = column;
        char c = input.charAt(pos);

        // Arrow ->
        if (c == '-' && pos + 1 < input.length() && input.charAt(pos + 1) == '>') {
            pos += 2;
            column += 2;
            return new Token(TokenType.ARROW, "->", startLine, startCol);
        }

        // Single-char punctuation
        TokenType punct = switch (c) {
            case '(' -> TokenType.LPAREN;
            case ')' -> TokenType.RPAREN;
            case '{' -> TokenType.LBRACE;
            case '}' -> TokenType.RBRACE;
            case '<' -> TokenType.LANGLE;
            case '>' -> TokenType.RANGLE;
            case '[' -> TokenType.LBRACKET;
            case ']' -> TokenType.RBRACKET;
            case ':' -> TokenType.COLON;
            case ',' -> TokenType.COMMA;
            case '=' -> TokenType.EQUALS;
            default -> null;
        };

        if (punct != null) {
            pos++;
            column++;
            return new Token(punct, String.valueOf(c), startLine, startCol);
        }

        // Sigil tokens
        if (c == '%') {
            return scanPercentId(startLine, startCol);
        }
        if (c == '@') {
            return scanAtId(startLine, startCol);
        }
        if (c == '#') {
            return scanHashId(startLine, startCol);
        }

        // String literal
        if (c == '"') {
            return scanString(startLine, startCol);
        }

        // Number (integer or float)
        if (Character.isDigit(c) || (c == '-' && pos + 1 < input.length() && Character.isDigit(input.charAt(pos + 1)))) {
            return scanNumber(startLine, startCol);
        }

        // Identifier (including dotted names like func.func, stablehlo.dot_general)
        if (Character.isLetter(c) || c == '_') {
            return scanIdentifier(startLine, startCol);
        }

        throw new StableHloParseException(
            String.format("Unexpected character '%c' at line %d, column %d", c, startLine, startCol));
    }

    private Token scanPercentId(int startLine, int startCol) {
        StringBuilder sb = new StringBuilder();
        sb.append(input.charAt(pos++)); // %
        column++;

        while (pos < input.length()) {
            char c = input.charAt(pos);
            if (Character.isLetterOrDigit(c) || c == '_') {
                sb.append(c);
                pos++;
                column++;
            } else {
                break;
            }
        }

        return new Token(TokenType.PERCENT_ID, sb.toString(), startLine, startCol);
    }

    private Token scanAtId(int startLine, int startCol) {
        StringBuilder sb = new StringBuilder();
        sb.append(input.charAt(pos++)); // @
        column++;

        while (pos < input.length()) {
            char c = input.charAt(pos);
            if (Character.isLetterOrDigit(c) || c == '_') {
                sb.append(c);
                pos++;
                column++;
            } else {
                break;
            }
        }

        return new Token(TokenType.AT_ID, sb.toString(), startLine, startCol);
    }

    private Token scanHashId(int startLine, int startCol) {
        StringBuilder sb = new StringBuilder();
        sb.append(input.charAt(pos++)); // #
        column++;

        // Scan dotted identifier (e.g., stablehlo.dot)
        while (pos < input.length()) {
            char c = input.charAt(pos);
            if (Character.isLetterOrDigit(c) || c == '_' || c == '.') {
                sb.append(c);
                pos++;
                column++;
            } else {
                break;
            }
        }

        return new Token(TokenType.HASH_ID, sb.toString(), startLine, startCol);
    }

    private Token scanString(int startLine, int startCol) {
        StringBuilder sb = new StringBuilder();
        sb.append(input.charAt(pos++)); // opening "
        column++;

        while (pos < input.length()) {
            char c = input.charAt(pos);
            sb.append(c);
            pos++;
            column++;

            if (c == '"') {
                break;
            }
            if (c == '\\' && pos < input.length()) {
                // Escape sequence
                sb.append(input.charAt(pos++));
                column++;
            }
        }

        return new Token(TokenType.STRING, sb.toString(), startLine, startCol);
    }

    private Token scanNumber(int startLine, int startCol) {
        StringBuilder sb = new StringBuilder();
        boolean hasDecimal = false;

        // Optional leading minus
        if (input.charAt(pos) == '-') {
            sb.append('-');
            pos++;
            column++;
        }

        while (pos < input.length()) {
            char c = input.charAt(pos);
            if (Character.isDigit(c)) {
                sb.append(c);
                pos++;
                column++;
            } else if (c == '.' && !hasDecimal) {
                // Check if it's a decimal point (not end of identifier)
                if (pos + 1 < input.length() && Character.isDigit(input.charAt(pos + 1))) {
                    hasDecimal = true;
                    sb.append(c);
                    pos++;
                    column++;
                } else {
                    break;
                }
            } else if (c == 'e' || c == 'E') {
                // Scientific notation
                hasDecimal = true;
                sb.append(c);
                pos++;
                column++;
                if (pos < input.length() && (input.charAt(pos) == '+' || input.charAt(pos) == '-')) {
                    sb.append(input.charAt(pos++));
                    column++;
                }
            } else {
                break;
            }
        }

        TokenType type = hasDecimal ? TokenType.FLOAT : TokenType.INTEGER;
        return new Token(type, sb.toString(), startLine, startCol);
    }

    private Token scanIdentifier(int startLine, int startCol) {
        StringBuilder sb = new StringBuilder();

        while (pos < input.length()) {
            char c = input.charAt(pos);
            // Allow dotted identifiers like func.func, stablehlo.dot_general
            if (Character.isLetterOrDigit(c) || c == '_' || c == '.') {
                sb.append(c);
                pos++;
                column++;
            } else {
                break;
            }
        }

        return new Token(TokenType.IDENTIFIER, sb.toString(), startLine, startCol);
    }
}
