package io.surfworks.warpforge.core.tensor.typed.dim;

/**
 * Wildcard dimension that can match any other dimension type.
 *
 * <p>Use {@code Any} when you don't need dimension type safety for a particular
 * axis, or when writing generic code that should accept any dimension:
 *
 * <pre>{@code
 * // Accept any batch size, but enforce Hidden dimension
 * void processHidden(TypedTensor<DimMatrix<Any, Hidden>, F32, Cpu> input) {
 *     // ...
 * }
 *
 * // Can be called with any row dimension:
 * processHidden(tensorWithBatch);     // DimMatrix<Batch, Hidden>
 * processHidden(tensorWith32Rows);    // DimMatrix<Numeric._32, Hidden>
 * }</pre>
 *
 * <p><strong>Warning:</strong> Using {@code Any} bypasses compile-time dimension
 * checking. Use it sparingly and only when necessary.
 *
 * <p>For full type safety, prefer using specific dimension markers from
 * {@link Semantic} or {@link Numeric}, or define your own.
 */
public interface Any extends Dim {}
