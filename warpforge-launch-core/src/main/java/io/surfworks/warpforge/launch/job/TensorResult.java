package io.surfworks.warpforge.launch.job;

import io.surfworks.warpforge.core.tensor.Tensor;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Result of a tensor computation - either inline data or a reference to a file.
 */
public sealed interface TensorResult permits TensorResult.Inline, TensorResult.FileRef {

    /**
     * Inline tensor data held in memory.
     */
    record Inline(Tensor tensor) implements TensorResult {
        public Inline {
            Objects.requireNonNull(tensor, "tensor cannot be null");
        }
    }

    /**
     * Reference to tensor data stored in a file (e.g., .npy format).
     */
    record FileRef(Path path, int[] shape, String dtype) implements TensorResult {
        public FileRef {
            Objects.requireNonNull(path, "path cannot be null");
            Objects.requireNonNull(shape, "shape cannot be null");
            Objects.requireNonNull(dtype, "dtype cannot be null");
            if (shape.length == 0) {
                throw new IllegalArgumentException("shape must have at least one dimension");
            }
        }

        /**
         * Returns the total number of elements.
         */
        public long elementCount() {
            long count = 1;
            for (int dim : shape) {
                count *= dim;
            }
            return count;
        }
    }

    /**
     * Creates an inline tensor result.
     */
    static TensorResult inline(Tensor tensor) {
        return new Inline(tensor);
    }

    /**
     * Creates a file reference tensor result.
     */
    static TensorResult fileRef(Path path, int[] shape, String dtype) {
        return new FileRef(path, shape, dtype);
    }
}
