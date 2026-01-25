package io.surfworks.warpforge.core.tensor.typed.dim;

import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.Cpu;
import io.surfworks.warpforge.core.tensor.typed.device.Nvidia;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.ops.DimOps;
import io.surfworks.warpforge.core.tensor.typed.shape.DimMatrix;
import io.surfworks.warpforge.core.tensor.typed.shape.DimVector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Compile-time shape checking verification.
 *
 * <p>This class documents the compile-time type safety provided by DimOps.
 * The commented-out code blocks demonstrate what SHOULD NOT compile if
 * uncommented - they represent shape/dtype/device mismatches that the
 * Java compiler will reject.
 *
 * <h2>How to Verify Compile-Time Safety</h2>
 * <ol>
 *   <li>Uncomment any of the "SHOULD NOT COMPILE" blocks below</li>
 *   <li>Attempt to compile the project</li>
 *   <li>Verify the compiler produces a type error</li>
 *   <li>Re-comment the block</li>
 * </ol>
 *
 * <h2>Type Safety Guarantees</h2>
 * <ul>
 *   <li><strong>Dimension matching</strong>: Inner dimensions in matmul must use the same type parameter</li>
 *   <li><strong>Dtype matching</strong>: Both operands must have the same dtype parameter</li>
 *   <li><strong>Device matching</strong>: Both operands must be on the same device</li>
 * </ul>
 */
@DisplayName("Compile-Time Shape Safety Documentation")
class CompileTimeShapeTest {

    // Define test dimension markers
    interface M extends Dim {}
    interface K extends Dim {}
    interface N extends Dim {}
    interface P extends Dim {}  // Different from K for mismatch tests

    @Test
    @DisplayName("Valid matmul - inner dimensions match")
    void validMatmulCompiles() {
        // This DOES compile and run correctly
        try (var a = TypedTensor.<DimMatrix<M, K>, F32, Cpu>zeros(
                new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);
             var b = TypedTensor.<DimMatrix<K, N>, F32, Cpu>zeros(
                     new DimMatrix<>(4, 5), F32.INSTANCE, Cpu.INSTANCE)) {

            // K matches K - this compiles
            TypedTensor<DimMatrix<M, N>, F32, Cpu> c = DimOps.matmul(a, b);
            assertNotNull(c);
            c.close();
        }
    }

    @Test
    @DisplayName("Valid matvec - dimensions match")
    void validMatvecCompiles() {
        try (var matrix = TypedTensor.<DimMatrix<M, N>, F32, Cpu>zeros(
                new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);
             var vector = TypedTensor.<DimVector<N>, F32, Cpu>zeros(
                     new DimVector<>(4), F32.INSTANCE, Cpu.INSTANCE)) {

            // N matches N - this compiles
            TypedTensor<DimVector<M>, F32, Cpu> result = DimOps.matvec(matrix, vector);
            assertNotNull(result);
            result.close();
        }
    }

    /*
     * ========================================================================
     * COMPILE-TIME ERROR EXAMPLES
     *
     * The following code blocks are commented out because they SHOULD NOT
     * compile. Each demonstrates a different type mismatch that the Java
     * compiler catches.
     *
     * To verify: Uncomment any block, attempt to build, observe compile error.
     * ========================================================================
     */

    // ------------------------------------------------------------------------
    // EXAMPLE 1: Inner dimension mismatch in matmul
    // ------------------------------------------------------------------------
    // Error: K vs P - inner dimensions don't match
    /*
    @Test
    void innerDimensionMismatch_SHOULD_NOT_COMPILE() {
        var a = TypedTensor.<DimMatrix<M, K>, F32, Cpu>zeros(
                new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);
        var b = TypedTensor.<DimMatrix<P, N>, F32, Cpu>zeros(  // P != K
                new DimMatrix<>(4, 5), F32.INSTANCE, Cpu.INSTANCE);

        // COMPILE ERROR: no matching method found for matmul
        // DimMatrix<M, K> has K columns, DimMatrix<P, N> has P rows
        // K and P are different types
        var c = DimOps.matmul(a, b);
    }
    */

    // ------------------------------------------------------------------------
    // EXAMPLE 2: Dtype mismatch
    // ------------------------------------------------------------------------
    // Error: F32 vs F64
    /*
    @Test
    void dtypeMismatch_SHOULD_NOT_COMPILE() {
        var a = TypedTensor.<DimMatrix<M, K>, F32, Cpu>zeros(
                new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);
        var b = TypedTensor.<DimMatrix<K, N>, F64, Cpu>zeros(  // F64 != F32
                new DimMatrix<>(4, 5), F64.INSTANCE, Cpu.INSTANCE);

        // COMPILE ERROR: F32 and F64 are incompatible type parameters
        var c = DimOps.matmul(a, b);
    }
    */

    // ------------------------------------------------------------------------
    // EXAMPLE 3: Device mismatch
    // ------------------------------------------------------------------------
    // Error: Cpu vs Nvidia
    /*
    @Test
    void deviceMismatch_SHOULD_NOT_COMPILE() {
        var a = TypedTensor.<DimMatrix<M, K>, F32, Cpu>zeros(
                new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);
        var b = TypedTensor.<DimMatrix<K, N>, F32, Nvidia>zeros(  // Nvidia != Cpu
                new DimMatrix<>(4, 5), F32.INSTANCE, Nvidia.DEFAULT);

        // COMPILE ERROR: Cpu and Nvidia are incompatible device types
        var c = DimOps.matmul(a, b);
    }
    */

    // ------------------------------------------------------------------------
    // EXAMPLE 4: Matvec dimension mismatch
    // ------------------------------------------------------------------------
    // Error: Matrix[M, N] @ Vector[P] - N != P
    /*
    @Test
    void matvecDimensionMismatch_SHOULD_NOT_COMPILE() {
        var matrix = TypedTensor.<DimMatrix<M, N>, F32, Cpu>zeros(
                new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);
        var vector = TypedTensor.<DimVector<P>, F32, Cpu>zeros(  // P != N
                new DimVector<>(4), F32.INSTANCE, Cpu.INSTANCE);

        // COMPILE ERROR: Vector dimension P doesn't match matrix column dimension N
        var result = DimOps.matvec(matrix, vector);
    }
    */

    // ------------------------------------------------------------------------
    // EXAMPLE 5: Self-multiply with wrong inner dimension
    // ------------------------------------------------------------------------
    // Error: [M, K] @ [M, K] - outer dims used as inner
    /*
    @Test
    void selfMultiplyWrongInner_SHOULD_NOT_COMPILE() {
        var a = TypedTensor.<DimMatrix<M, K>, F32, Cpu>zeros(
                new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);

        // COMPILE ERROR: Can't multiply [M, K] @ [M, K]
        // Would need [M, K] @ [K, ?] but a has rows=M, not rows=K
        var c = DimOps.matmul(a, a);
    }
    */

    // ------------------------------------------------------------------------
    // EXAMPLE 6: Batched matmul batch dimension mismatch
    // ------------------------------------------------------------------------
    // Error: Different batch dimensions B vs P
    /*
    @Test
    void batchedMatmulBatchMismatch_SHOULD_NOT_COMPILE() {
        interface B extends Dim {}

        var a = TypedTensor.<DimRank3<B, M, K>, F32, Cpu>zeros(
                new DimRank3<>(8, 3, 4), F32.INSTANCE, Cpu.INSTANCE);
        var b = TypedTensor.<DimRank3<P, K, N>, F32, Cpu>zeros(  // P != B
                new DimRank3<>(8, 4, 5), F32.INSTANCE, Cpu.INSTANCE);

        // COMPILE ERROR: Batch dimensions B and P don't match
        var c = DimOps.batchedMatmul(a, b);
    }
    */

    // ------------------------------------------------------------------------
    // EXAMPLE 7: Add shape mismatch
    // ------------------------------------------------------------------------
    // Error: Different dimension types
    /*
    @Test
    void addShapeMismatch_SHOULD_NOT_COMPILE() {
        var a = TypedTensor.<DimMatrix<M, K>, F32, Cpu>zeros(
                new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);
        var b = TypedTensor.<DimMatrix<M, N>, F32, Cpu>zeros(  // N != K
                new DimMatrix<>(3, 4), F32.INSTANCE, Cpu.INSTANCE);

        // COMPILE ERROR: DimMatrix<M, K> and DimMatrix<M, N> are different types
        var c = DimOps.add(a, b);
    }
    */

    // ------------------------------------------------------------------------
    // SUMMARY OF TYPE SAFETY GUARANTEES
    // ------------------------------------------------------------------------
    //
    // | Operation        | Type Parameter Requirement                          |
    // |-----------------|-----------------------------------------------------|
    // | matmul(A, B)    | A's col dim == B's row dim (same Dim type)          |
    // | batchedMatmul   | Batch dims must match + inner dims must match       |
    // | matvec(M, v)    | M's col dim == v's length dim                       |
    // | vecmat(v, M)    | v's length == M's row dim                           |
    // | add(A, B)       | A and B must have identical shape type              |
    // | All operations  | Dtype and Device parameters must match              |
    //
}
