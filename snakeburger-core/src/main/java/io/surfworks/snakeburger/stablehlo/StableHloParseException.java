package io.surfworks.snakeburger.stablehlo;

/**
 * Exception thrown during StableHLO parsing or validation.
 */
public class StableHloParseException extends RuntimeException {

    private final int line;
    private final int column;

    public StableHloParseException(String message) {
        super(message);
        this.line = -1;
        this.column = -1;
    }

    public StableHloParseException(String message, int line, int column) {
        super(String.format("%s at line %d, column %d", message, line, column));
        this.line = line;
        this.column = column;
    }

    public StableHloParseException(String message, Throwable cause) {
        super(message, cause);
        this.line = -1;
        this.column = -1;
    }

    public int getLine() {
        return line;
    }

    public int getColumn() {
        return column;
    }
}
