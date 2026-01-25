package io.surfworks.warpforge.core.tensor.typed.dim;

/**
 * Marker interface for dimension type parameters.
 *
 * <p>Dimensions encode logical tensor axes at the type level. Using consistent
 * dimension markers enables compile-time shape checking for operations like matmul.
 *
 * <p>Example:
 * <pre>{@code
 * // Define dimensions for your model
 * interface Batch extends Dim {}
 * interface SeqLen extends Dim {}
 * interface Hidden extends Dim {}
 *
 * // Create tensors with dimension-typed shapes
 * TypedTensor<DimMatrix<Batch, Hidden>, F32, Cpu> input = ...;
 * TypedTensor<DimMatrix<Hidden, SeqLen>, F32, Cpu> weights = ...;
 *
 * // matmul enforces Hidden dimension matches - checked at COMPILE TIME
 * TypedTensor<DimMatrix<Batch, SeqLen>, F32, Cpu> output = DimOps.matmul(input, weights);
 * }</pre>
 *
 * <p>WarpForge provides predefined dimension markers in two categories:
 * <ul>
 *   <li>{@link Semantic} - Named dimensions for common ML patterns (Batch, SeqLen, Hidden, etc.)
 *   <li>{@link Numeric} - Numeric dimensions for fixed sizes (_32, _768, _1024, etc.)
 * </ul>
 *
 * <p>Users can also define their own dimension markers by extending this interface:
 * <pre>{@code
 * interface MyCustomDim extends Dim {}
 * }</pre>
 *
 * @see Semantic for predefined semantic dimension markers
 * @see Numeric for predefined numeric dimension markers
 * @see Any for wildcard dimension matching
 */
public interface Dim {}
