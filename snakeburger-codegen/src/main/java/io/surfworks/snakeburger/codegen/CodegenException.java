package io.surfworks.snakeburger.codegen;

/**
 * Exception thrown during code generation.
 */
public class CodegenException extends Exception {

    public CodegenException(String message) {
        super(message);
    }

    public CodegenException(String message, Throwable cause) {
        super(message, cause);
    }
}
