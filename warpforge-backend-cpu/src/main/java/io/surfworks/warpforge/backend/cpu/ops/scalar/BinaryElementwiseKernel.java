package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.List;

/**
 * Base class for binary elementwise operations.
 * Applies a function to corresponding elements of two input tensors.
 * Uses Java Vector API for SIMD vectorization when available.
 */
public abstract class BinaryElementwiseKernel implements OpKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int VECTOR_LENGTH = SPECIES.length();

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("Binary operation requires exactly 2 inputs, got " + inputs.size());
        }

        Tensor lhs = inputs.get(0);
        Tensor rhs = inputs.get(1);
        TensorSpec spec = TensorSpec.fromAst(op.tensorResultType());

        float[] lhsData = lhs.toFloatArray();
        float[] rhsData = rhs.toFloatArray();

        // Element-wise operation (no broadcasting in base class)
        if (lhsData.length != rhsData.length) {
            throw new IllegalArgumentException(
                "Input tensors must have same number of elements for elementwise operation, got " +
                lhsData.length + " and " + rhsData.length);
        }

        float[] outputData = new float[lhsData.length];

        // Check if this kernel supports vectorization
        if (supportsVectorization()) {
            // Process vectors
            int i = 0;
            int upperBound = SPECIES.loopBound(lhsData.length);
            for (; i < upperBound; i += VECTOR_LENGTH) {
                FloatVector va = FloatVector.fromArray(SPECIES, lhsData, i);
                FloatVector vb = FloatVector.fromArray(SPECIES, rhsData, i);
                FloatVector result = applyVector(va, vb);
                result.intoArray(outputData, i);
            }
            // Process remaining elements
            for (; i < lhsData.length; i++) {
                outputData[i] = apply(lhsData[i], rhsData[i]);
            }
        } else {
            // Scalar fallback
            for (int i = 0; i < lhsData.length; i++) {
                outputData[i] = apply(lhsData[i], rhsData[i]);
            }
        }

        Tensor output = Tensor.fromFloatArray(outputData, spec.shape());
        return List.of(output);
    }

    /**
     * Apply the binary function to a pair of values.
     */
    protected abstract float apply(float a, float b);

    /**
     * Apply the binary function to vectors of values.
     * Override in subclasses that support vectorization.
     */
    protected FloatVector applyVector(FloatVector a, FloatVector b) {
        // Default: scalar fallback
        float[] arrA = new float[VECTOR_LENGTH];
        float[] arrB = new float[VECTOR_LENGTH];
        float[] result = new float[VECTOR_LENGTH];
        a.intoArray(arrA, 0);
        b.intoArray(arrB, 0);
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            result[i] = apply(arrA[i], arrB[i]);
        }
        return FloatVector.fromArray(SPECIES, result, 0);
    }

    /**
     * Override to return true if this kernel has an efficient vector implementation.
     */
    protected boolean supportsVectorization() {
        return false; // Default: use scalar processing
    }
}
