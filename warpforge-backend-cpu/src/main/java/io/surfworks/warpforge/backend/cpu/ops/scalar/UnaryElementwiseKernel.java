package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.List;

/**
 * Base class for unary elementwise operations.
 * Applies a function to each element of the input tensor.
 * Uses Java Vector API for SIMD vectorization when available.
 */
public abstract class UnaryElementwiseKernel implements OpKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VECTOR_LENGTH = SPECIES.length();

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 1) {
            throw new IllegalArgumentException("Unary operation requires exactly 1 input, got " + inputs.size());
        }

        Tensor input = inputs.getFirst();
        TensorSpec spec = TensorSpec.fromAst(op.tensorResultType());

        float[] inputData = input.toFloatArray();
        float[] outputData = new float[inputData.length];

        // Check if this kernel supports vectorization
        if (supportsVectorization()) {
            // Process vectors
            int i = 0;
            int upperBound = SPECIES.loopBound(inputData.length);
            for (; i < upperBound; i += VECTOR_LENGTH) {
                FloatVector v = FloatVector.fromArray(SPECIES, inputData, i);
                FloatVector result = applyVector(v);
                result.intoArray(outputData, i);
            }
            // Process remaining elements
            for (; i < inputData.length; i++) {
                outputData[i] = apply(inputData[i]);
            }
        } else {
            // Scalar fallback
            for (int i = 0; i < inputData.length; i++) {
                outputData[i] = apply(inputData[i]);
            }
        }

        Tensor output = Tensor.fromFloatArray(outputData, spec.shape());
        return List.of(output);
    }

    /**
     * Apply the unary function to a single value.
     */
    protected abstract float apply(float x);

    /**
     * Apply the unary function to a vector of values.
     * Override in subclasses that support vectorization.
     */
    protected FloatVector applyVector(FloatVector v) {
        // Default: scalar fallback (process each lane separately)
        float[] arr = new float[VECTOR_LENGTH];
        v.intoArray(arr, 0);
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            arr[i] = apply(arr[i]);
        }
        return FloatVector.fromArray(SPECIES, arr, 0);
    }

    /**
     * Override to return true if this kernel has an efficient vector implementation.
     */
    protected boolean supportsVectorization() {
        return false; // Default: use scalar processing
    }
}
