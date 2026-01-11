package io.surfworks.warpforge.backend.cpu.ops.scalar;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;
import io.surfworks.warpforge.core.tensor.TensorSpec;

import java.util.List;

/**
 * stablehlo.constant - Creates a constant tensor.
 */
public class ConstantKernel implements OpKernel {

    @Override
    public List<Tensor> execute(StableHloAst.Operation op, List<Tensor> inputs) {
        if (!(op instanceof StableHloAst.ConstantOp constantOp)) {
            throw new IllegalArgumentException("Expected ConstantOp");
        }

        TensorSpec spec = TensorSpec.fromAst(constantOp.tensorResultType());
        StableHloAst.DenseAttr attr = constantOp.value();
        Object value = attr.value();

        float[] data = extractFloatData(value, spec.elementCount());
        Tensor output = Tensor.fromFloatArray(data, spec.shape());
        return List.of(output);
    }

    private float[] extractFloatData(Object value, long elementCount) {
        if (value instanceof Number num) {
            // Scalar constant - broadcast to all elements
            float[] data = new float[(int) elementCount];
            float val = num.floatValue();
            for (int i = 0; i < data.length; i++) {
                data[i] = val;
            }
            return data;
        } else if (value instanceof List<?> list) {
            // List of values
            float[] data = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object elem = list.get(i);
                if (elem instanceof Number num) {
                    data[i] = num.floatValue();
                } else if (elem instanceof List<?> innerList) {
                    // Nested list - flatten recursively
                    float[] flat = flattenList(innerList);
                    return flat;
                } else {
                    throw new IllegalArgumentException("Unsupported element type: " + elem.getClass());
                }
            }
            return data;
        } else if (value instanceof float[] arr) {
            return arr;
        } else if (value instanceof double[] arr) {
            float[] data = new float[arr.length];
            for (int i = 0; i < arr.length; i++) {
                data[i] = (float) arr[i];
            }
            return data;
        } else {
            throw new IllegalArgumentException("Unsupported constant value type: " + value.getClass());
        }
    }

    private float[] flattenList(List<?> list) {
        java.util.ArrayList<Float> flat = new java.util.ArrayList<>();
        flattenListRecursive(list, flat);
        float[] result = new float[flat.size()];
        for (int i = 0; i < flat.size(); i++) {
            result[i] = flat.get(i);
        }
        return result;
    }

    private void flattenListRecursive(List<?> list, java.util.ArrayList<Float> result) {
        for (Object elem : list) {
            if (elem instanceof Number num) {
                result.add(num.floatValue());
            } else if (elem instanceof List<?> nested) {
                flattenListRecursive(nested, result);
            } else {
                throw new IllegalArgumentException("Unsupported nested element: " + elem.getClass());
            }
        }
    }
}
