package io.surfworks.warpforge.core.tensor.typed.shape;

/**
 * Shape type for 2-dimensional tensors (matrices).
 *
 * <p>A matrix has two dimensions: rows and columns.
 * Use -1 for either dimension to indicate dynamic/unknown size.
 *
 * <p>Example:
 * <pre>{@code
 * TypedTensor<Matrix, F32, Cpu> weights = TypedTensor.zeros(
 *     new Matrix(768, 512), F32.INSTANCE, Cpu.INSTANCE);
 * }</pre>
 *
 * @param rows number of rows, or -1 if unknown
 * @param cols number of columns, or -1 if unknown
 */
public record Matrix(int rows, int cols) implements Shape {

    /**
     * Matrix with unknown dimensions (fully dynamic).
     */
    public static final Matrix DYNAMIC = new Matrix(-1, -1);

    /**
     * Creates a matrix shape with the given dimensions.
     *
     * @param rows number of rows (must be positive or -1 for dynamic)
     * @param cols number of columns (must be positive or -1 for dynamic)
     * @throws IllegalArgumentException if dimensions are invalid
     */
    public Matrix {
        if ((rows < -1 || rows == 0) || (cols < -1 || cols == 0)) {
            throw new IllegalArgumentException(
                "Matrix dimensions must be positive or -1 for dynamic, got: [" + rows + ", " + cols + "]");
        }
    }

    /**
     * Creates a matrix with dynamic rows but fixed columns.
     * Useful for batch processing where batch size varies.
     *
     * @param cols the fixed column count
     * @return a matrix shape with dynamic rows
     */
    public static Matrix withDynamicRows(int cols) {
        return new Matrix(-1, cols);
    }

    /**
     * Creates a matrix with fixed rows but dynamic columns.
     *
     * @param rows the fixed row count
     * @return a matrix shape with dynamic columns
     */
    public static Matrix withDynamicCols(int rows) {
        return new Matrix(rows, -1);
    }

    /**
     * Returns true if both dimensions are known.
     */
    @Override
    public boolean isFullyKnown() {
        return rows > 0 && cols > 0;
    }

    /**
     * Returns true if rows dimension is known.
     */
    public boolean isRowsKnown() {
        return rows > 0;
    }

    /**
     * Returns true if cols dimension is known.
     */
    public boolean isColsKnown() {
        return cols > 0;
    }

    @Override
    public int rank() {
        return 2;
    }

    @Override
    public int[] dimensions() {
        if (!isFullyKnown()) {
            throw new IllegalStateException("Cannot get dimensions for matrix with unknown dimensions");
        }
        return new int[]{rows, cols};
    }

    @Override
    public String toString() {
        String r = rows > 0 ? String.valueOf(rows) : "?";
        String c = cols > 0 ? String.valueOf(cols) : "?";
        return "Matrix[" + r + ", " + c + "]";
    }
}
