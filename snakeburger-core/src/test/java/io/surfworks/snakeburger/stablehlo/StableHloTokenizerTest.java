package io.surfworks.snakeburger.stablehlo;

import io.surfworks.snakeburger.stablehlo.StableHloTokenizer.Token;
import io.surfworks.snakeburger.stablehlo.StableHloTokenizer.TokenType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StableHloTokenizerTest {

    private List<Token> tokenize(String input) {
        return new StableHloTokenizer(input).tokenize();
    }

    private Token firstToken(String input) {
        List<Token> tokens = tokenize(input);
        assertFalse(tokens.isEmpty());
        return tokens.get(0);
    }

    @Nested
    class IdentifierTests {

        @Test
        void simpleIdentifier() {
            Token token = firstToken("module");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("module", token.value());
        }

        @Test
        void dottedIdentifier() {
            Token token = firstToken("func.func");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("func.func", token.value());
        }

        @Test
        void dottedIdentifierWithUnderscore() {
            Token token = firstToken("stablehlo.dot_general");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("stablehlo.dot_general", token.value());
        }

        @Test
        void identifierWithDigits() {
            Token token = firstToken("tensor32");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("tensor32", token.value());
        }

        @Test
        void underscoreIdentifier() {
            Token token = firstToken("_private");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("_private", token.value());
        }

        @Test
        void singleCharIdentifier() {
            Token token = firstToken("x");
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals("x", token.value());
        }

        @ParameterizedTest
        @ValueSource(strings = {"f32", "f64", "i32", "i64", "bf16", "f16", "i8", "i16", "i1"})
        void scalarTypeIdentifiers(String typeName) {
            Token token = firstToken(typeName);
            assertEquals(TokenType.IDENTIFIER, token.type());
            assertEquals(typeName, token.value());
        }
    }

    @Nested
    class SigilTests {

        @Test
        void percentId() {
            Token token = firstToken("%arg0");
            assertEquals(TokenType.PERCENT_ID, token.type());
            assertEquals("%arg0", token.value());
        }

        @Test
        void percentIdWithUnderscore() {
            Token token = firstToken("%result_1");
            assertEquals(TokenType.PERCENT_ID, token.type());
            assertEquals("%result_1", token.value());
        }

        @Test
        void percentIdNumeric() {
            Token token = firstToken("%0");
            assertEquals(TokenType.PERCENT_ID, token.type());
            assertEquals("%0", token.value());
        }

        @Test
        void percentIdLongNumeric() {
            Token token = firstToken("%123456");
            assertEquals(TokenType.PERCENT_ID, token.type());
            assertEquals("%123456", token.value());
        }

        @Test
        void atId() {
            Token token = firstToken("@main");
            assertEquals(TokenType.AT_ID, token.type());
            assertEquals("@main", token.value());
        }

        @Test
        void atIdWithUnderscore() {
            Token token = firstToken("@my_function");
            assertEquals(TokenType.AT_ID, token.type());
            assertEquals("@my_function", token.value());
        }

        @Test
        void hashId() {
            Token token = firstToken("#stablehlo.dot");
            assertEquals(TokenType.HASH_ID, token.type());
            assertEquals("#stablehlo.dot", token.value());
        }

        @Test
        void hashIdSimple() {
            Token token = firstToken("#attr");
            assertEquals(TokenType.HASH_ID, token.type());
            assertEquals("#attr", token.value());
        }
    }

    @Nested
    class NumericLiteralTests {

        @Test
        void integerZero() {
            Token token = firstToken("0");
            assertEquals(TokenType.INTEGER, token.type());
            assertEquals("0", token.value());
        }

        @Test
        void positiveInteger() {
            Token token = firstToken("42");
            assertEquals(TokenType.INTEGER, token.type());
            assertEquals("42", token.value());
        }

        @Test
        void largeInteger() {
            Token token = firstToken("1234567890");
            assertEquals(TokenType.INTEGER, token.type());
            assertEquals("1234567890", token.value());
        }

        @Test
        void negativeInteger() {
            Token token = firstToken("-42");
            assertEquals(TokenType.INTEGER, token.type());
            assertEquals("-42", token.value());
        }

        @Test
        void floatZero() {
            Token token = firstToken("0.0");
            assertEquals(TokenType.FLOAT, token.type());
            assertEquals("0.0", token.value());
        }

        @Test
        void positiveFloat() {
            Token token = firstToken("3.14159");
            assertEquals(TokenType.FLOAT, token.type());
            assertEquals("3.14159", token.value());
        }

        @Test
        void negativeFloat() {
            Token token = firstToken("-2.5");
            assertEquals(TokenType.FLOAT, token.type());
            assertEquals("-2.5", token.value());
        }

        @Test
        void scientificNotation() {
            Token token = firstToken("1e10");
            assertEquals(TokenType.FLOAT, token.type());
            assertEquals("1e10", token.value());
        }

        @Test
        void scientificNotationUpperE() {
            Token token = firstToken("1E10");
            assertEquals(TokenType.FLOAT, token.type());
            assertEquals("1E10", token.value());
        }

        @Test
        void scientificNotationNegativeExponent() {
            Token token = firstToken("1e-5");
            assertEquals(TokenType.FLOAT, token.type());
            assertEquals("1e-5", token.value());
        }

        @Test
        void scientificNotationPositiveExponent() {
            Token token = firstToken("2.5e+3");
            assertEquals(TokenType.FLOAT, token.type());
            assertEquals("2.5e+3", token.value());
        }
    }

    @Nested
    class PunctuationTests {

        @Test
        void leftParen() {
            Token token = firstToken("(");
            assertEquals(TokenType.LPAREN, token.type());
        }

        @Test
        void rightParen() {
            Token token = firstToken(")");
            assertEquals(TokenType.RPAREN, token.type());
        }

        @Test
        void leftBrace() {
            Token token = firstToken("{");
            assertEquals(TokenType.LBRACE, token.type());
        }

        @Test
        void rightBrace() {
            Token token = firstToken("}");
            assertEquals(TokenType.RBRACE, token.type());
        }

        @Test
        void leftAngle() {
            Token token = firstToken("<");
            assertEquals(TokenType.LANGLE, token.type());
        }

        @Test
        void rightAngle() {
            Token token = firstToken(">");
            assertEquals(TokenType.RANGLE, token.type());
        }

        @Test
        void leftBracket() {
            Token token = firstToken("[");
            assertEquals(TokenType.LBRACKET, token.type());
        }

        @Test
        void rightBracket() {
            Token token = firstToken("]");
            assertEquals(TokenType.RBRACKET, token.type());
        }

        @Test
        void colon() {
            Token token = firstToken(":");
            assertEquals(TokenType.COLON, token.type());
        }

        @Test
        void comma() {
            Token token = firstToken(",");
            assertEquals(TokenType.COMMA, token.type());
        }

        @Test
        void equals() {
            Token token = firstToken("=");
            assertEquals(TokenType.EQUALS, token.type());
        }

        @Test
        void arrow() {
            Token token = firstToken("->");
            assertEquals(TokenType.ARROW, token.type());
            assertEquals("->", token.value());
        }
    }

    @Nested
    class StringLiteralTests {

        @Test
        void emptyString() {
            Token token = firstToken("\"\"");
            assertEquals(TokenType.STRING, token.type());
            assertEquals("\"\"", token.value());
        }

        @Test
        void simpleString() {
            Token token = firstToken("\"hello\"");
            assertEquals(TokenType.STRING, token.type());
            assertEquals("\"hello\"", token.value());
        }

        @Test
        void stringWithSpaces() {
            Token token = firstToken("\"hello world\"");
            assertEquals(TokenType.STRING, token.type());
            assertEquals("\"hello world\"", token.value());
        }

        @Test
        void stringWithEscapedQuote() {
            Token token = firstToken("\"say \\\"hi\\\"\"");
            assertEquals(TokenType.STRING, token.type());
        }
    }

    @Nested
    class WhitespaceAndCommentTests {

        @Test
        void ignoresSpaces() {
            List<Token> tokens = tokenize("   module   ");
            assertEquals(2, tokens.size()); // IDENTIFIER + EOF
            assertEquals(TokenType.IDENTIFIER, tokens.get(0).type());
        }

        @Test
        void ignoresTabs() {
            List<Token> tokens = tokenize("\t\tmodule\t\t");
            assertEquals(2, tokens.size());
            assertEquals(TokenType.IDENTIFIER, tokens.get(0).type());
        }

        @Test
        void ignoresNewlines() {
            List<Token> tokens = tokenize("\n\nmodule\n\n");
            assertEquals(2, tokens.size());
            assertEquals(TokenType.IDENTIFIER, tokens.get(0).type());
        }

        @Test
        void ignoresLineComments() {
            List<Token> tokens = tokenize("// this is a comment\nmodule");
            assertEquals(2, tokens.size());
            assertEquals(TokenType.IDENTIFIER, tokens.get(0).type());
            assertEquals("module", tokens.get(0).value());
        }

        @Test
        void ignoresMultipleLineComments() {
            List<Token> tokens = tokenize("// comment 1\n// comment 2\nmodule");
            assertEquals(2, tokens.size());
            assertEquals("module", tokens.get(0).value());
        }

        @Test
        void commentAtEndOfLine() {
            List<Token> tokens = tokenize("module // inline comment");
            assertEquals(2, tokens.size());
            assertEquals("module", tokens.get(0).value());
        }
    }

    @Nested
    class LineAndColumnTrackingTests {

        @Test
        void firstTokenPosition() {
            Token token = firstToken("module");
            assertEquals(1, token.line());
            assertEquals(1, token.column());
        }

        @Test
        void tokenAfterNewline() {
            List<Token> tokens = tokenize("\nmodule");
            Token token = tokens.get(0);
            assertEquals(2, token.line());
            assertEquals(1, token.column());
        }

        @Test
        void tokenAfterSpaces() {
            List<Token> tokens = tokenize("   module");
            Token token = tokens.get(0);
            assertEquals(1, token.line());
            assertEquals(4, token.column());
        }

        @Test
        void multipleTokensOnSameLine() {
            List<Token> tokens = tokenize("a b c");
            assertEquals(1, tokens.get(0).column());
            assertEquals(3, tokens.get(1).column());
            assertEquals(5, tokens.get(2).column());
        }
    }

    @Nested
    class ComplexInputTests {

        @Test
        void tensorTypeDeclaration() {
            List<Token> tokens = tokenize("tensor<4x8xf32>");
            assertEquals(6, tokens.size()); // tensor < 4 x8xf32 > EOF
            assertEquals(TokenType.IDENTIFIER, tokens.get(0).type());
            assertEquals("tensor", tokens.get(0).value());
            assertEquals(TokenType.LANGLE, tokens.get(1).type());
            assertEquals(TokenType.INTEGER, tokens.get(2).type());
            assertEquals("4", tokens.get(2).value());
        }

        @Test
        void functionSignature() {
            List<Token> tokens = tokenize("func.func public @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>)");
            assertTrue(tokens.stream().anyMatch(t -> t.type() == TokenType.IDENTIFIER && t.value().equals("func.func")));
            assertTrue(tokens.stream().anyMatch(t -> t.type() == TokenType.IDENTIFIER && t.value().equals("public")));
            assertTrue(tokens.stream().anyMatch(t -> t.type() == TokenType.AT_ID && t.value().equals("@main")));
            assertTrue(tokens.stream().anyMatch(t -> t.type() == TokenType.PERCENT_ID && t.value().equals("%arg0")));
            assertTrue(tokens.stream().anyMatch(t -> t.type() == TokenType.ARROW));
        }

        @Test
        void operationWithAttributes() {
            String input = "%0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_contracting_dimensions = [1]>";
            List<Token> tokens = tokenize(input);

            assertTrue(tokens.stream().anyMatch(t -> t.type() == TokenType.PERCENT_ID && t.value().equals("%0")));
            assertTrue(tokens.stream().anyMatch(t -> t.type() == TokenType.EQUALS));
            assertTrue(tokens.stream().anyMatch(t -> t.type() == TokenType.IDENTIFIER && t.value().equals("stablehlo.dot_general")));
            assertTrue(tokens.stream().anyMatch(t -> t.type() == TokenType.HASH_ID && t.value().equals("#stablehlo.dot")));
        }

        @Test
        void denseConstant() {
            List<Token> tokens = tokenize("dense<0.0>");
            assertEquals(TokenType.IDENTIFIER, tokens.get(0).type());
            assertEquals("dense", tokens.get(0).value());
            assertEquals(TokenType.LANGLE, tokens.get(1).type());
            assertEquals(TokenType.FLOAT, tokens.get(2).type());
            assertEquals("0.0", tokens.get(2).value());
            assertEquals(TokenType.RANGLE, tokens.get(3).type());
        }

        @Test
        void emptyBrackets() {
            List<Token> tokens = tokenize("[]");
            assertEquals(3, tokens.size()); // [ ] EOF
            assertEquals(TokenType.LBRACKET, tokens.get(0).type());
            assertEquals(TokenType.RBRACKET, tokens.get(1).type());
        }

        @Test
        void bracketedList() {
            List<Token> tokens = tokenize("[0, 1, 2]");
            assertEquals(TokenType.LBRACKET, tokens.get(0).type());
            assertEquals(TokenType.INTEGER, tokens.get(1).type());
            assertEquals(TokenType.COMMA, tokens.get(2).type());
            assertEquals(TokenType.INTEGER, tokens.get(3).type());
        }
    }

    @Nested
    class EdgeCaseTests {

        @Test
        void emptyInput() {
            List<Token> tokens = tokenize("");
            assertEquals(1, tokens.size());
            assertEquals(TokenType.EOF, tokens.get(0).type());
        }

        @Test
        void onlyWhitespace() {
            List<Token> tokens = tokenize("   \t\n\r   ");
            assertEquals(1, tokens.size());
            assertEquals(TokenType.EOF, tokens.get(0).type());
        }

        @Test
        void onlyComments() {
            List<Token> tokens = tokenize("// comment\n// another");
            assertEquals(1, tokens.size());
            assertEquals(TokenType.EOF, tokens.get(0).type());
        }

        @Test
        void minusNotFollowedByDigitOrArrow() {
            // "-" followed by non-digit, non-">" should throw
            assertThrows(StableHloParseException.class, () -> tokenize("- x"));
        }

        @Test
        void unexpectedCharacter() {
            assertThrows(StableHloParseException.class, () -> tokenize("$invalid"));
        }

        @Test
        void consecutiveOperators() {
            List<Token> tokens = tokenize("= = =");
            assertEquals(4, tokens.size()); // 3 EQUALS + EOF
            assertEquals(TokenType.EQUALS, tokens.get(0).type());
            assertEquals(TokenType.EQUALS, tokens.get(1).type());
            assertEquals(TokenType.EQUALS, tokens.get(2).type());
        }

        @Test
        void arrowsConsecutive() {
            List<Token> tokens = tokenize("-> ->");
            assertEquals(3, tokens.size()); // 2 ARROW + EOF
            assertEquals(TokenType.ARROW, tokens.get(0).type());
            assertEquals(TokenType.ARROW, tokens.get(1).type());
        }
    }

    @Nested
    class RealWorldExampleTests {

        @Test
        void fullModuleTokenization() {
            String mlir = """
                module @main {
                  func.func public @forward(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> (tensor<4x16xf32>) {
                    %0 = stablehlo.dot_general %arg0, %arg1, #stablehlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]> : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
                    stablehlo.return %0 : tensor<4x16xf32>
                  }
                }
                """;

            List<Token> tokens = tokenize(mlir);

            // Verify we can tokenize without errors
            assertFalse(tokens.isEmpty());
            assertEquals(TokenType.EOF, tokens.get(tokens.size() - 1).type());

            // Verify key tokens exist
            assertTrue(tokens.stream().anyMatch(t -> t.value().equals("module")));
            assertTrue(tokens.stream().anyMatch(t -> t.value().equals("@main")));
            assertTrue(tokens.stream().anyMatch(t -> t.value().equals("func.func")));
            assertTrue(tokens.stream().anyMatch(t -> t.value().equals("stablehlo.dot_general")));
            assertTrue(tokens.stream().anyMatch(t -> t.value().equals("stablehlo.return")));
        }
    }
}
