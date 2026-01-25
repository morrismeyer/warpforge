package io.surfworks.warpforge.core.tensor.typed.shape;

import io.surfworks.warpforge.core.tensor.typed.dim.Dim;

/**
 * Matrix shape with compile-time dimension type parameters.
 *
 * <p>DimMatrix encodes row and column dimensions at the type level, enabling
 * compile-time verification of matrix operation compatibility (e.g., matmul
 * inner dimension matching).
 *
 * <p>Example:
 * <pre>{@code
 * interface Batch extends Dim {}
 * interface Hidden extends Dim {}
 * interface Vocab extends Dim {}
 *
 * TypedTensor<DimMatrix<Batch, Hidden>, F32, Cpu> input = ...;
 * TypedTensor<DimMatrix<Hidden, Vocab>, F32, Cpu> classifier = ...;
 *
 * // matmul enforces Hidden dimension matches at compile time:
 * TypedTensor<DimMatrix<Batch, Vocab>, F32, Cpu> logits = DimOps.matmul(input, classifier);
 *
 * // This won't compile - dimension type mismatch:
 * // DimOps.matmul(input, TypedTensor<DimMatrix<Batch, Vocab>, ...>);
 * }</pre>
 *
 * @param <R> the dimension type for rows
 * @param <C> the dimension type for columns
 * @param rows the runtime row count
 * @param cols the runtime column count
 */
public record DimMatrix<R extends Dim, C extends Dim>(int rows, int cols) implements Shape {

    /**
     * Creates a DimMatrix with the specified dimensions.
     *
     * @param rows the number of rows (must be non-negative)
     * @param cols the number of columns (must be non-negative)
     * @throws IllegalArgumentException if any dimension is negative
     */
    public DimMatrix {
        if (rows < 0) {
            throw new IllegalArgumentException("Matrix rows must be non-negative: " + rows);
        }
        if (cols < 0) {
            throw new IllegalArgumentException("Matrix cols must be non-negative: " + cols);
        }
    }

    @Override
    public int rank() {
        return 2;
    }

    @Override
    public int[] dimensions() {
        return new int[]{rows, cols};
    }

    @Override
    public boolean isFullyKnown() {
        return true;
    }

    @Override
    public long elementCount() {
        return (long) rows * cols;
    }

    /**
     * Creates a transposed shape with swapped dimension types.
     *
     * <p>This preserves type safety: if you transpose a {@code DimMatrix<R, C>},
     * you get a {@code DimMatrix<C, R>}.
     *
     * @return a new DimMatrix with rows/cols and dimension types swapped
     */
    public DimMatrix<C, R> transposed() {
        return new DimMatrix<>(cols, rows);
    }

    @Override
    public String toString() {
        return "DimMatrix[" + rows + ", " + cols + "]";
    }
}
