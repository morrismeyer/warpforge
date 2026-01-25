package io.surfworks.warpforge.data.stablehlo;

import io.surfworks.warpforge.data.stablehlo.StableHloTypes.TensorType;

import java.util.List;
import java.util.Map;

/**
 * StableHLO operation definitions for warpforge-data.
 *
 * <p>Provides a minimal set of operations needed for benchmark model representation.
 * Operations can be serialized to MLIR text format.
 */
public final class StableHloOps {

    private StableHloOps() {}

    /**
     * Base interface for all StableHLO operations.
     */
    public sealed interface Operation permits
            AddOp, SubtractOp, MultiplyOp, DivideOp,
            NegateOp, AbsOp, ExpOp, LogOp, SqrtOp, TanhOp, SinOp, CosOp,
            DotOp, DotGeneralOp, ConvolutionOp,
            BroadcastInDimOp, TransposeOp, ReshapeOp, SliceOp, ConcatenateOp,
            ReduceOp, MaxOp, MinOp,
            ConstantOp, ReturnOp {

        /**
         * Operation name in StableHLO dialect.
         */
        String opName();

        /**
         * Result type(s) of this operation.
         */
        List<TensorType> resultTypes();

        /**
         * Emit MLIR text for this operation.
         *
         * @param resultNames Names for result values (e.g., "1", "2")
         * @param operandNames Names of operand values
         * @return MLIR text representation
         */
        String toMlir(List<String> resultNames, List<String> operandNames);
    }

    // ==================== Elementwise Binary Operations ====================

    public record AddOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.add";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.add %" + operandNames.get(0)
                    + ", %" + operandNames.get(1) + " : " + resultType.toMlir();
        }
    }

    public record SubtractOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.subtract";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.subtract %" + operandNames.get(0)
                    + ", %" + operandNames.get(1) + " : " + resultType.toMlir();
        }
    }

    public record MultiplyOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.multiply";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.multiply %" + operandNames.get(0)
                    + ", %" + operandNames.get(1) + " : " + resultType.toMlir();
        }
    }

    public record DivideOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.divide";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.divide %" + operandNames.get(0)
                    + ", %" + operandNames.get(1) + " : " + resultType.toMlir();
        }
    }

    public record MaxOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.maximum";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.maximum %" + operandNames.get(0)
                    + ", %" + operandNames.get(1) + " : " + resultType.toMlir();
        }
    }

    public record MinOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.minimum";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.minimum %" + operandNames.get(0)
                    + ", %" + operandNames.get(1) + " : " + resultType.toMlir();
        }
    }

    // ==================== Elementwise Unary Operations ====================

    public record NegateOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.negate";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.negate %" + operandNames.get(0)
                    + " : " + resultType.toMlir();
        }
    }

    public record AbsOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.abs";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.abs %" + operandNames.get(0)
                    + " : " + resultType.toMlir();
        }
    }

    public record ExpOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.exponential";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.exponential %" + operandNames.get(0)
                    + " : " + resultType.toMlir();
        }
    }

    public record LogOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.log";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.log %" + operandNames.get(0)
                    + " : " + resultType.toMlir();
        }
    }

    public record SqrtOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.sqrt";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.sqrt %" + operandNames.get(0)
                    + " : " + resultType.toMlir();
        }
    }

    public record TanhOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.tanh";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.tanh %" + operandNames.get(0)
                    + " : " + resultType.toMlir();
        }
    }

    public record SinOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.sine";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.sine %" + operandNames.get(0)
                    + " : " + resultType.toMlir();
        }
    }

    public record CosOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.cosine";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.cosine %" + operandNames.get(0)
                    + " : " + resultType.toMlir();
        }
    }

    // ==================== Matrix Operations ====================

    /**
     * Simple dot product (matrix multiplication).
     */
    public record DotOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.dot";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.dot %" + operandNames.get(0)
                    + ", %" + operandNames.get(1) + " : " + resultType.toMlir();
        }
    }

    /**
     * General dot product with explicit dimension mapping (batched matmul).
     */
    public record DotGeneralOp(
            TensorType resultType,
            List<Long> lhsBatchDims,
            List<Long> rhsBatchDims,
            List<Long> lhsContractDims,
            List<Long> rhsContractDims
    ) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.dot_general";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            StringBuilder sb = new StringBuilder();
            sb.append("%").append(resultNames.get(0)).append(" = stablehlo.dot_general %")
                    .append(operandNames.get(0)).append(", %").append(operandNames.get(1))
                    .append(", dot_dimension_numbers = <batching_dims = [")
                    .append(formatDims(lhsBatchDims)).append("] x [")
                    .append(formatDims(rhsBatchDims)).append("], contracting_dims = [")
                    .append(formatDims(lhsContractDims)).append("] x [")
                    .append(formatDims(rhsContractDims)).append("]>")
                    .append(" : ").append(resultType.toMlir());
            return sb.toString();
        }

        private String formatDims(List<Long> dims) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < dims.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(dims.get(i));
            }
            return sb.toString();
        }
    }

    /**
     * Convolution operation.
     */
    public record ConvolutionOp(
            TensorType resultType,
            long[] windowStrides,
            long[] padding,
            long[] lhsDilation,
            long[] rhsDilation,
            long featureGroupCount,
            long batchGroupCount
    ) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.convolution";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            StringBuilder sb = new StringBuilder();
            sb.append("%").append(resultNames.get(0)).append(" = stablehlo.convolution %")
                    .append(operandNames.get(0)).append(", %").append(operandNames.get(1))
                    .append(" window_strides = ").append(formatArray(windowStrides))
                    .append(" padding = ").append(formatPadding(padding))
                    .append(" lhs_dilation = ").append(formatArray(lhsDilation))
                    .append(" rhs_dilation = ").append(formatArray(rhsDilation))
                    .append(" feature_group_count = ").append(featureGroupCount)
                    .append(" batch_group_count = ").append(batchGroupCount)
                    .append(" : ").append(resultType.toMlir());
            return sb.toString();
        }

        private String formatArray(long[] arr) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(arr[i]);
            }
            sb.append("]");
            return sb.toString();
        }

        private String formatPadding(long[] pad) {
            // Padding is [low0, high0, low1, high1, ...]
            StringBuilder sb = new StringBuilder("dense<[[");
            for (int i = 0; i < pad.length / 2; i++) {
                if (i > 0) sb.append("], [");
                sb.append(pad[i * 2]).append(", ").append(pad[i * 2 + 1]);
            }
            sb.append("]]>");
            return sb.toString();
        }
    }

    // ==================== Shape Operations ====================

    public record BroadcastInDimOp(TensorType resultType, long[] broadcastDimensions) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.broadcast_in_dim";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            StringBuilder sb = new StringBuilder();
            sb.append("%").append(resultNames.get(0)).append(" = stablehlo.broadcast_in_dim %")
                    .append(operandNames.get(0))
                    .append(", dims = [");
            for (int i = 0; i < broadcastDimensions.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(broadcastDimensions[i]);
            }
            sb.append("] : ").append(resultType.toMlir());
            return sb.toString();
        }
    }

    public record TransposeOp(TensorType resultType, long[] permutation) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.transpose";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            StringBuilder sb = new StringBuilder();
            sb.append("%").append(resultNames.get(0)).append(" = stablehlo.transpose %")
                    .append(operandNames.get(0))
                    .append(", permutation = [");
            for (int i = 0; i < permutation.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(permutation[i]);
            }
            sb.append("] : ").append(resultType.toMlir());
            return sb.toString();
        }
    }

    public record ReshapeOp(TensorType resultType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.reshape";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.reshape %" + operandNames.get(0)
                    + " : " + resultType.toMlir();
        }
    }

    public record SliceOp(TensorType resultType, long[] startIndices, long[] limitIndices, long[] strides) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.slice";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            StringBuilder sb = new StringBuilder();
            sb.append("%").append(resultNames.get(0)).append(" = stablehlo.slice %")
                    .append(operandNames.get(0))
                    .append(" [").append(formatIndices(startIndices))
                    .append("] : [").append(formatIndices(limitIndices))
                    .append("] : [").append(formatIndices(strides))
                    .append("] : ").append(resultType.toMlir());
            return sb.toString();
        }

        private String formatIndices(long[] indices) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < indices.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(indices[i]);
            }
            return sb.toString();
        }
    }

    public record ConcatenateOp(TensorType resultType, long dimension) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.concatenate";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            StringBuilder sb = new StringBuilder();
            sb.append("%").append(resultNames.get(0)).append(" = stablehlo.concatenate ");
            for (int i = 0; i < operandNames.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append("%").append(operandNames.get(i));
            }
            sb.append(", dimension = ").append(dimension)
                    .append(" : ").append(resultType.toMlir());
            return sb.toString();
        }
    }

    // ==================== Reduction Operations ====================

    public record ReduceOp(TensorType resultType, long[] dimensions, String reduceType) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.reduce";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            StringBuilder sb = new StringBuilder();
            sb.append("%").append(resultNames.get(0)).append(" = stablehlo.reduce %")
                    .append(operandNames.get(0))
                    .append(" across dimensions = [");
            for (int i = 0; i < dimensions.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(dimensions[i]);
            }
            sb.append("] : ").append(resultType.toMlir())
                    .append(" {\n  ").append(reduceType).append("\n}");
            return sb.toString();
        }
    }

    // ==================== Constants and Return ====================

    public record ConstantOp(TensorType resultType, Object value) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.constant";
        }

        @Override
        public List<TensorType> resultTypes() {
            return List.of(resultType);
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            return "%" + resultNames.get(0) + " = stablehlo.constant dense<" + formatValue(value)
                    + "> : " + resultType.toMlir();
        }

        private String formatValue(Object val) {
            if (val instanceof float[] arr) {
                if (arr.length == 1) return String.valueOf(arr[0]);
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < arr.length; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(arr[i]);
                }
                sb.append("]");
                return sb.toString();
            } else if (val instanceof Float f) {
                return f.toString();
            } else if (val instanceof Double d) {
                return d.toString();
            } else if (val instanceof Number n) {
                return n.toString();
            }
            return val.toString();
        }
    }

    public record ReturnOp(List<TensorType> resultTypes) implements Operation {
        @Override
        public String opName() {
            return "stablehlo.return";
        }

        @Override
        public List<TensorType> resultTypes() {
            return resultTypes;
        }

        @Override
        public String toMlir(List<String> resultNames, List<String> operandNames) {
            StringBuilder sb = new StringBuilder("stablehlo.return ");
            for (int i = 0; i < operandNames.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append("%").append(operandNames.get(i));
            }
            sb.append(" : ");
            for (int i = 0; i < resultTypes.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(resultTypes.get(i).toMlir());
            }
            return sb.toString();
        }
    }
}
