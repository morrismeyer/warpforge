package io.surfworks.warpforge.core.tensor.typed.ops;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.List;

import io.surfworks.warpforge.core.tensor.ScalarType;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.typed.TypedTensor;
import io.surfworks.warpforge.core.tensor.typed.device.DeviceTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.DTypeTag;
import io.surfworks.warpforge.core.tensor.typed.dtype.F32;
import io.surfworks.warpforge.core.tensor.typed.dtype.F64;
import io.surfworks.warpforge.core.tensor.typed.shape.Dynamic;
import io.surfworks.warpforge.core.tensor.typed.shape.Matrix;
import io.surfworks.warpforge.core.tensor.typed.shape.Rank3;
import io.surfworks.warpforge.core.tensor.typed.shape.Shape;
import io.surfworks.warpforge.core.tensor.typed.shape.Vector;

/**
 * Type-safe slicing and indexing operations for tensors.
 *
 * <p>These operations extract or combine portions of tensors. They are essential
 * for transformer models which need:
 * <ul>
 *   <li>Token embeddings via gather (lookup table)</li>
 *   <li>Splitting heads for multi-head attention</li>
 *   <li>Concatenating attention outputs</li>
 *   <li>Slicing sequences for causal masking</li>
 * </ul>
 *
 * <p>Example:
 * <pre>{@code
 * // Embedding lookup: indices [batch, seq] into embedding table [vocab, hidden]
 * TypedTensor<Matrix, F32, Cpu> embeddings = ...;  // [vocab_size, hidden_dim]
 * TypedTensor<Matrix, I32, Cpu> indices = ...;     // [batch, seq_len]
 * TypedTensor<Rank3, F32, Cpu> output = SliceOps.gather(embeddings, indices, 0);
 * // Result: [batch, seq_len, hidden_dim]
 *
 * // Concatenate attention heads
 * List<TypedTensor<Rank3, F32, Cpu>> heads = ...;  // each [batch, seq, head_dim]
 * TypedTensor<Rank3, F32, Cpu> combined = SliceOps.cat(heads, 2);
 * // Result: [batch, seq, num_heads * head_dim]
 * }</pre>
 */
public final class SliceOps {

    private SliceOps() {
        // Utility class
    }

    // ==================== Gather Operations ====================

    /**
     * Gathers elements from a Matrix using integer indices.
     *
     * <p>This is the core operation for embedding lookups. Given an embedding
     * table [vocab_size, hidden_dim] and indices [batch, seq], produces
     * output [batch, seq, hidden_dim].
     *
     * @param input the embedding table [N, D]
     * @param indices the indices to gather [B, S] as int array
     * @param <D> dtype type of input
     * @param <V> device type
     * @return gathered tensor [B, S, D]
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Rank3, D, V> gather(TypedTensor<Matrix, D, V> input, int[][] indices) {
        int[] inputDims = input.dimensions();
        int vocabSize = inputDims[0];
        int hiddenDim = inputDims[1];

        int batchSize = indices.length;
        int seqLen = indices[0].length;

        Rank3 outputShape = new Rank3(batchSize, seqLen, hiddenDim);
        Tensor result = Tensor.zeros(input.underlying().dtype(), outputShape.dimensions());

        ScalarType dtype = input.underlying().dtype();
        if (dtype == ScalarType.F32) {
            gatherMatrixF32(input.underlying().data(), result.data(),
                    indices, vocabSize, hiddenDim, batchSize, seqLen);
        } else if (dtype == ScalarType.F64) {
            gatherMatrixF64(input.underlying().data(), result.data(),
                    indices, vocabSize, hiddenDim, batchSize, seqLen);
        } else {
            throw new UnsupportedOperationException("gather not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, outputShape, input.dtypeType(), input.deviceType());
    }

    /**
     * Gathers elements from a Matrix using a Vector of indices.
     *
     * <p>Given input [N, D] and indices [M], produces output [M, D].
     * This is simpler than the batched version - just a 1D gather along axis 0.
     *
     * @param input the source matrix [N, D]
     * @param indices the indices to gather
     * @param <D> dtype type
     * @param <V> device type
     * @return gathered matrix [M, D]
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> gatherRows(TypedTensor<Matrix, D, V> input, int[] indices) {
        int[] inputDims = input.dimensions();
        int numRows = inputDims[0];
        int numCols = inputDims[1];

        int numIndices = indices.length;
        Matrix outputShape = new Matrix(numIndices, numCols);
        Tensor result = Tensor.zeros(input.underlying().dtype(), outputShape.dimensions());

        ScalarType dtype = input.underlying().dtype();
        MemorySegment src = input.underlying().data();
        MemorySegment dst = result.data();

        if (dtype == ScalarType.F32) {
            for (int i = 0; i < numIndices; i++) {
                int rowIdx = indices[i];
                if (rowIdx < 0 || rowIdx >= numRows) {
                    throw new IndexOutOfBoundsException(
                            "Index " + rowIdx + " out of bounds for dimension 0 with size " + numRows);
                }
                long srcOffset = (long) rowIdx * numCols;
                long dstOffset = (long) i * numCols;
                for (int j = 0; j < numCols; j++) {
                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, srcOffset + j);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstOffset + j, val);
                }
            }
        } else if (dtype == ScalarType.F64) {
            for (int i = 0; i < numIndices; i++) {
                int rowIdx = indices[i];
                if (rowIdx < 0 || rowIdx >= numRows) {
                    throw new IndexOutOfBoundsException(
                            "Index " + rowIdx + " out of bounds for dimension 0 with size " + numRows);
                }
                long srcOffset = (long) rowIdx * numCols;
                long dstOffset = (long) i * numCols;
                for (int j = 0; j < numCols; j++) {
                    double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, srcOffset + j);
                    dst.setAtIndex(ValueLayout.JAVA_DOUBLE, dstOffset + j, val);
                }
            }
        } else {
            throw new UnsupportedOperationException("gatherRows not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, outputShape, input.dtypeType(), input.deviceType());
    }

    /**
     * Gathers elements from a Vector using indices.
     *
     * @param input the source vector [N]
     * @param indices the indices to gather
     * @param <D> dtype type
     * @param <V> device type
     * @return gathered vector [M]
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Vector, D, V> gatherVector(TypedTensor<Vector, D, V> input, int[] indices) {
        int[] inputDims = input.dimensions();
        int vecLen = inputDims[0];

        int numIndices = indices.length;
        Vector outputShape = new Vector(numIndices);
        Tensor result = Tensor.zeros(input.underlying().dtype(), outputShape.dimensions());

        ScalarType dtype = input.underlying().dtype();
        MemorySegment src = input.underlying().data();
        MemorySegment dst = result.data();

        if (dtype == ScalarType.F32) {
            for (int i = 0; i < numIndices; i++) {
                int idx = indices[i];
                if (idx < 0 || idx >= vecLen) {
                    throw new IndexOutOfBoundsException(
                            "Index " + idx + " out of bounds for vector with size " + vecLen);
                }
                float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, idx);
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, val);
            }
        } else if (dtype == ScalarType.F64) {
            for (int i = 0; i < numIndices; i++) {
                int idx = indices[i];
                if (idx < 0 || idx >= vecLen) {
                    throw new IndexOutOfBoundsException(
                            "Index " + idx + " out of bounds for vector with size " + vecLen);
                }
                double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, idx);
                dst.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
            }
        } else {
            throw new UnsupportedOperationException("gatherVector not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, outputShape, input.dtypeType(), input.deviceType());
    }

    // ==================== Slice Operations ====================

    /**
     * Extracts a contiguous slice from a Matrix along axis 0 (rows).
     *
     * @param input the input matrix [M, N]
     * @param start starting row index (inclusive)
     * @param end ending row index (exclusive)
     * @param <D> dtype type
     * @param <V> device type
     * @return sliced matrix [end-start, N]
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> sliceRows(TypedTensor<Matrix, D, V> input, int start, int end) {
        int[] dims = input.dimensions();
        int numRows = dims[0];
        int numCols = dims[1];

        if (start < 0 || end > numRows || start >= end) {
            throw new IllegalArgumentException(
                    "Invalid slice range [" + start + ", " + end + ") for dimension 0 with size " + numRows);
        }

        int sliceRows = end - start;
        Matrix outputShape = new Matrix(sliceRows, numCols);
        Tensor result = Tensor.zeros(input.underlying().dtype(), outputShape.dimensions());

        ScalarType dtype = input.underlying().dtype();
        MemorySegment src = input.underlying().data();
        MemorySegment dst = result.data();

        long srcOffset = (long) start * numCols;
        long copySize = (long) sliceRows * numCols;

        if (dtype == ScalarType.F32) {
            for (long i = 0; i < copySize; i++) {
                float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, srcOffset + i);
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, val);
            }
        } else if (dtype == ScalarType.F64) {
            for (long i = 0; i < copySize; i++) {
                double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, srcOffset + i);
                dst.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
            }
        } else {
            throw new UnsupportedOperationException("sliceRows not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, outputShape, input.dtypeType(), input.deviceType());
    }

    /**
     * Extracts a contiguous slice from a Vector.
     *
     * @param input the input vector [N]
     * @param start starting index (inclusive)
     * @param end ending index (exclusive)
     * @param <D> dtype type
     * @param <V> device type
     * @return sliced vector [end-start]
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Vector, D, V> sliceVector(TypedTensor<Vector, D, V> input, int start, int end) {
        int[] dims = input.dimensions();
        int vecLen = dims[0];

        if (start < 0 || end > vecLen || start >= end) {
            throw new IllegalArgumentException(
                    "Invalid slice range [" + start + ", " + end + ") for vector with size " + vecLen);
        }

        int sliceLen = end - start;
        Vector outputShape = new Vector(sliceLen);
        Tensor result = Tensor.zeros(input.underlying().dtype(), outputShape.dimensions());

        ScalarType dtype = input.underlying().dtype();
        MemorySegment src = input.underlying().data();
        MemorySegment dst = result.data();

        if (dtype == ScalarType.F32) {
            for (int i = 0; i < sliceLen; i++) {
                float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, start + i);
                dst.setAtIndex(ValueLayout.JAVA_FLOAT, i, val);
            }
        } else if (dtype == ScalarType.F64) {
            for (int i = 0; i < sliceLen; i++) {
                double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, start + i);
                dst.setAtIndex(ValueLayout.JAVA_DOUBLE, i, val);
            }
        } else {
            throw new UnsupportedOperationException("sliceVector not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, outputShape, input.dtypeType(), input.deviceType());
    }

    /**
     * Extracts a slice from a Rank3 tensor along axis 1 (sequence dimension).
     * Common for slicing sequences in attention.
     *
     * @param input the input tensor [B, S, D]
     * @param start starting sequence index (inclusive)
     * @param end ending sequence index (exclusive)
     * @param <D> dtype type
     * @param <V> device type
     * @return sliced tensor [B, end-start, D]
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Rank3, D, V> sliceSequence(TypedTensor<Rank3, D, V> input, int start, int end) {
        int[] dims = input.dimensions();
        int batchSize = dims[0];
        int seqLen = dims[1];
        int hiddenDim = dims[2];

        if (start < 0 || end > seqLen || start >= end) {
            throw new IllegalArgumentException(
                    "Invalid slice range [" + start + ", " + end + ") for sequence dimension with size " + seqLen);
        }

        int sliceLen = end - start;
        Rank3 outputShape = new Rank3(batchSize, sliceLen, hiddenDim);
        Tensor result = Tensor.zeros(input.underlying().dtype(), outputShape.dimensions());

        ScalarType dtype = input.underlying().dtype();
        MemorySegment src = input.underlying().data();
        MemorySegment dst = result.data();

        if (dtype == ScalarType.F32) {
            sliceSequenceF32(src, dst, batchSize, seqLen, hiddenDim, start, sliceLen);
        } else if (dtype == ScalarType.F64) {
            sliceSequenceF64(src, dst, batchSize, seqLen, hiddenDim, start, sliceLen);
        } else {
            throw new UnsupportedOperationException("sliceSequence not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, outputShape, input.dtypeType(), input.deviceType());
    }

    // ==================== Concatenate Operations ====================

    /**
     * Concatenates a list of Vectors into a single Vector.
     *
     * @param inputs the vectors to concatenate
     * @param <D> dtype type
     * @param <V> device type
     * @return concatenated vector
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Vector, D, V> catVectors(List<TypedTensor<Vector, D, V>> inputs) {
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("Cannot concatenate empty list");
        }

        // Compute total length
        int totalLen = 0;
        for (TypedTensor<Vector, D, V> input : inputs) {
            totalLen += input.dimensions()[0];
        }

        TypedTensor<Vector, D, V> first = inputs.get(0);
        Vector outputShape = new Vector(totalLen);
        Tensor result = Tensor.zeros(first.underlying().dtype(), outputShape.dimensions());

        ScalarType dtype = first.underlying().dtype();
        MemorySegment dst = result.data();

        long offset = 0;
        if (dtype == ScalarType.F32) {
            for (TypedTensor<Vector, D, V> input : inputs) {
                MemorySegment src = input.underlying().data();
                int len = input.dimensions()[0];
                for (int i = 0; i < len; i++) {
                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + i, val);
                }
                offset += len;
            }
        } else if (dtype == ScalarType.F64) {
            for (TypedTensor<Vector, D, V> input : inputs) {
                MemorySegment src = input.underlying().data();
                int len = input.dimensions()[0];
                for (int i = 0; i < len; i++) {
                    double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                    dst.setAtIndex(ValueLayout.JAVA_DOUBLE, offset + i, val);
                }
                offset += len;
            }
        } else {
            throw new UnsupportedOperationException("catVectors not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, outputShape, first.dtypeType(), first.deviceType());
    }

    /**
     * Concatenates matrices along axis 0 (stack rows).
     *
     * @param inputs the matrices to concatenate (must have same number of columns)
     * @param <D> dtype type
     * @param <V> device type
     * @return concatenated matrix
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Matrix, D, V> catRows(List<TypedTensor<Matrix, D, V>> inputs) {
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("Cannot concatenate empty list");
        }

        TypedTensor<Matrix, D, V> first = inputs.get(0);
        int numCols = first.dimensions()[1];

        // Validate and compute total rows
        int totalRows = 0;
        for (TypedTensor<Matrix, D, V> input : inputs) {
            if (input.dimensions()[1] != numCols) {
                throw new IllegalArgumentException(
                        "All matrices must have same number of columns. Expected " + numCols +
                        ", got " + input.dimensions()[1]);
            }
            totalRows += input.dimensions()[0];
        }

        Matrix outputShape = new Matrix(totalRows, numCols);
        Tensor result = Tensor.zeros(first.underlying().dtype(), outputShape.dimensions());

        ScalarType dtype = first.underlying().dtype();
        MemorySegment dst = result.data();

        long offset = 0;
        if (dtype == ScalarType.F32) {
            for (TypedTensor<Matrix, D, V> input : inputs) {
                MemorySegment src = input.underlying().data();
                long size = (long) input.dimensions()[0] * numCols;
                for (long i = 0; i < size; i++) {
                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, offset + i, val);
                }
                offset += size;
            }
        } else if (dtype == ScalarType.F64) {
            for (TypedTensor<Matrix, D, V> input : inputs) {
                MemorySegment src = input.underlying().data();
                long size = (long) input.dimensions()[0] * numCols;
                for (long i = 0; i < size; i++) {
                    double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, i);
                    dst.setAtIndex(ValueLayout.JAVA_DOUBLE, offset + i, val);
                }
                offset += size;
            }
        } else {
            throw new UnsupportedOperationException("catRows not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, outputShape, first.dtypeType(), first.deviceType());
    }

    /**
     * Concatenates Rank3 tensors along axis 2 (hidden dimension).
     * Common for combining attention heads.
     *
     * @param inputs the tensors to concatenate [B, S, D_i]
     * @param <D> dtype type
     * @param <V> device type
     * @return concatenated tensor [B, S, sum(D_i)]
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    TypedTensor<Rank3, D, V> catHidden(List<TypedTensor<Rank3, D, V>> inputs) {
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("Cannot concatenate empty list");
        }

        TypedTensor<Rank3, D, V> first = inputs.get(0);
        int batchSize = first.dimensions()[0];
        int seqLen = first.dimensions()[1];

        // Validate and compute total hidden dim
        int totalHidden = 0;
        for (TypedTensor<Rank3, D, V> input : inputs) {
            if (input.dimensions()[0] != batchSize || input.dimensions()[1] != seqLen) {
                throw new IllegalArgumentException(
                        "All tensors must have same batch and sequence dimensions. Expected [" +
                        batchSize + ", " + seqLen + ", *], got [" +
                        input.dimensions()[0] + ", " + input.dimensions()[1] + ", *]");
            }
            totalHidden += input.dimensions()[2];
        }

        Rank3 outputShape = new Rank3(batchSize, seqLen, totalHidden);
        Tensor result = Tensor.zeros(first.underlying().dtype(), outputShape.dimensions());

        ScalarType dtype = first.underlying().dtype();
        if (dtype == ScalarType.F32) {
            catHiddenF32(inputs, result.data(), batchSize, seqLen, totalHidden);
        } else if (dtype == ScalarType.F64) {
            catHiddenF64(inputs, result.data(), batchSize, seqLen, totalHidden);
        } else {
            throw new UnsupportedOperationException("catHidden not implemented for dtype: " + dtype);
        }

        return TypedTensor.from(result, outputShape, first.dtypeType(), first.deviceType());
    }

    // ==================== Split Operations ====================

    /**
     * Splits a Vector into multiple equal-sized chunks.
     *
     * @param input the input vector [N]
     * @param numChunks number of chunks (N must be divisible by numChunks)
     * @param <D> dtype type
     * @param <V> device type
     * @return list of vectors, each of size [N / numChunks]
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    List<TypedTensor<Vector, D, V>> splitVector(TypedTensor<Vector, D, V> input, int numChunks) {
        int[] dims = input.dimensions();
        int vecLen = dims[0];

        if (vecLen % numChunks != 0) {
            throw new IllegalArgumentException(
                    "Vector length " + vecLen + " is not divisible by " + numChunks);
        }

        int chunkSize = vecLen / numChunks;
        @SuppressWarnings("unchecked")
        TypedTensor<Vector, D, V>[] chunks = new TypedTensor[numChunks];

        for (int i = 0; i < numChunks; i++) {
            chunks[i] = sliceVector(input, i * chunkSize, (i + 1) * chunkSize);
        }

        return Arrays.asList(chunks);
    }

    /**
     * Splits a Rank3 tensor along axis 2 (hidden dimension) into equal-sized chunks.
     * Common for splitting into attention heads.
     *
     * @param input the input tensor [B, S, D]
     * @param numChunks number of chunks (D must be divisible by numChunks)
     * @param <D> dtype type
     * @param <V> device type
     * @return list of tensors, each of size [B, S, D / numChunks]
     */
    public static <D extends DTypeTag, V extends DeviceTag>
    List<TypedTensor<Rank3, D, V>> splitHidden(TypedTensor<Rank3, D, V> input, int numChunks) {
        int[] dims = input.dimensions();
        int batchSize = dims[0];
        int seqLen = dims[1];
        int hiddenDim = dims[2];

        if (hiddenDim % numChunks != 0) {
            throw new IllegalArgumentException(
                    "Hidden dimension " + hiddenDim + " is not divisible by " + numChunks);
        }

        int chunkSize = hiddenDim / numChunks;
        @SuppressWarnings("unchecked")
        TypedTensor<Rank3, D, V>[] chunks = new TypedTensor[numChunks];

        ScalarType dtype = input.underlying().dtype();
        MemorySegment src = input.underlying().data();

        for (int c = 0; c < numChunks; c++) {
            Rank3 chunkShape = new Rank3(batchSize, seqLen, chunkSize);
            Tensor chunkTensor = Tensor.zeros(dtype, chunkShape.dimensions());
            MemorySegment dst = chunkTensor.data();

            int hiddenOffset = c * chunkSize;

            if (dtype == ScalarType.F32) {
                splitHiddenChunkF32(src, dst, batchSize, seqLen, hiddenDim, hiddenOffset, chunkSize);
            } else if (dtype == ScalarType.F64) {
                splitHiddenChunkF64(src, dst, batchSize, seqLen, hiddenDim, hiddenOffset, chunkSize);
            } else {
                throw new UnsupportedOperationException("splitHidden not implemented for dtype: " + dtype);
            }

            chunks[c] = TypedTensor.from(chunkTensor, chunkShape, input.dtypeType(), input.deviceType());
        }

        return Arrays.asList(chunks);
    }

    // ==================== Internal Implementation ====================

    private static void gatherMatrixF32(MemorySegment src, MemorySegment dst,
                                        int[][] indices, int vocabSize, int hiddenDim,
                                        int batchSize, int seqLen) {
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int idx = indices[b][s];
                if (idx < 0 || idx >= vocabSize) {
                    throw new IndexOutOfBoundsException(
                            "Index " + idx + " out of bounds for vocabulary size " + vocabSize);
                }

                long srcOffset = (long) idx * hiddenDim;
                long dstOffset = ((long) b * seqLen + s) * hiddenDim;

                for (int d = 0; d < hiddenDim; d++) {
                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, srcOffset + d);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstOffset + d, val);
                }
            }
        }
    }

    private static void gatherMatrixF64(MemorySegment src, MemorySegment dst,
                                        int[][] indices, int vocabSize, int hiddenDim,
                                        int batchSize, int seqLen) {
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int idx = indices[b][s];
                if (idx < 0 || idx >= vocabSize) {
                    throw new IndexOutOfBoundsException(
                            "Index " + idx + " out of bounds for vocabulary size " + vocabSize);
                }

                long srcOffset = (long) idx * hiddenDim;
                long dstOffset = ((long) b * seqLen + s) * hiddenDim;

                for (int d = 0; d < hiddenDim; d++) {
                    double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, srcOffset + d);
                    dst.setAtIndex(ValueLayout.JAVA_DOUBLE, dstOffset + d, val);
                }
            }
        }
    }

    private static void sliceSequenceF32(MemorySegment src, MemorySegment dst,
                                         int batchSize, int seqLen, int hiddenDim,
                                         int start, int sliceLen) {
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < sliceLen; s++) {
                long srcOffset = ((long) b * seqLen + start + s) * hiddenDim;
                long dstOffset = ((long) b * sliceLen + s) * hiddenDim;

                for (int d = 0; d < hiddenDim; d++) {
                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, srcOffset + d);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstOffset + d, val);
                }
            }
        }
    }

    private static void sliceSequenceF64(MemorySegment src, MemorySegment dst,
                                         int batchSize, int seqLen, int hiddenDim,
                                         int start, int sliceLen) {
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < sliceLen; s++) {
                long srcOffset = ((long) b * seqLen + start + s) * hiddenDim;
                long dstOffset = ((long) b * sliceLen + s) * hiddenDim;

                for (int d = 0; d < hiddenDim; d++) {
                    double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, srcOffset + d);
                    dst.setAtIndex(ValueLayout.JAVA_DOUBLE, dstOffset + d, val);
                }
            }
        }
    }

    private static <D extends DTypeTag, V extends DeviceTag>
    void catHiddenF32(List<TypedTensor<Rank3, D, V>> inputs, MemorySegment dst,
                      int batchSize, int seqLen, int totalHidden) {
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                long dstBase = ((long) b * seqLen + s) * totalHidden;
                long hiddenOffset = 0;

                for (TypedTensor<Rank3, D, V> input : inputs) {
                    MemorySegment src = input.underlying().data();
                    int hiddenDim = input.dimensions()[2];
                    long srcBase = ((long) b * seqLen + s) * hiddenDim;

                    for (int d = 0; d < hiddenDim; d++) {
                        float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, srcBase + d);
                        dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstBase + hiddenOffset + d, val);
                    }
                    hiddenOffset += hiddenDim;
                }
            }
        }
    }

    private static <D extends DTypeTag, V extends DeviceTag>
    void catHiddenF64(List<TypedTensor<Rank3, D, V>> inputs, MemorySegment dst,
                      int batchSize, int seqLen, int totalHidden) {
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                long dstBase = ((long) b * seqLen + s) * totalHidden;
                long hiddenOffset = 0;

                for (TypedTensor<Rank3, D, V> input : inputs) {
                    MemorySegment src = input.underlying().data();
                    int hiddenDim = input.dimensions()[2];
                    long srcBase = ((long) b * seqLen + s) * hiddenDim;

                    for (int d = 0; d < hiddenDim; d++) {
                        double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, srcBase + d);
                        dst.setAtIndex(ValueLayout.JAVA_DOUBLE, dstBase + hiddenOffset + d, val);
                    }
                    hiddenOffset += hiddenDim;
                }
            }
        }
    }

    private static void splitHiddenChunkF32(MemorySegment src, MemorySegment dst,
                                            int batchSize, int seqLen, int hiddenDim,
                                            int hiddenOffset, int chunkSize) {
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                long srcBase = ((long) b * seqLen + s) * hiddenDim + hiddenOffset;
                long dstBase = ((long) b * seqLen + s) * chunkSize;

                for (int d = 0; d < chunkSize; d++) {
                    float val = src.getAtIndex(ValueLayout.JAVA_FLOAT, srcBase + d);
                    dst.setAtIndex(ValueLayout.JAVA_FLOAT, dstBase + d, val);
                }
            }
        }
    }

    private static void splitHiddenChunkF64(MemorySegment src, MemorySegment dst,
                                            int batchSize, int seqLen, int hiddenDim,
                                            int hiddenOffset, int chunkSize) {
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                long srcBase = ((long) b * seqLen + s) * hiddenDim + hiddenOffset;
                long dstBase = ((long) b * seqLen + s) * chunkSize;

                for (int d = 0; d < chunkSize; d++) {
                    double val = src.getAtIndex(ValueLayout.JAVA_DOUBLE, srcBase + d);
                    dst.setAtIndex(ValueLayout.JAVA_DOUBLE, dstBase + d, val);
                }
            }
        }
    }
}
