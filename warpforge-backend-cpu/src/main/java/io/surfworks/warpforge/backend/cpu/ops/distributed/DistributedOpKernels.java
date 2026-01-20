package io.surfworks.warpforge.backend.cpu.ops.distributed;

import java.util.ArrayList;
import java.util.List;

import io.surfworks.snakeburger.stablehlo.StableHloAst;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AfterAllOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AllGatherOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AllReduceOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.AllToAllOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CollectiveBroadcastOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.CollectivePermuteOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.InfeedOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.Operation;
import io.surfworks.snakeburger.stablehlo.StableHloAst.OutfeedOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.PartitionIdOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.RecvOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReduceScatterOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.ReplicaIdOp;
import io.surfworks.snakeburger.stablehlo.StableHloAst.SendOp;
import io.surfworks.warpforge.backend.cpu.ops.OpKernel;
import io.surfworks.warpforge.core.tensor.Tensor;

/**
 * CPU stub kernels for distributed/communication operations.
 *
 * <p>These operations are designed for multi-device/multi-process execution.
 * On a single CPU, they either act as identity operations or throw
 * appropriate errors indicating the need for a distributed runtime.
 */
public final class DistributedOpKernels {

    private DistributedOpKernels() {}

    /**
     * CPU kernel for stablehlo.after_all - synchronization barrier.
     * On single CPU, this is a no-op that produces a token.
     */
    public static class AfterAllKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            // Returns an empty tensor representing the token
            return List.of(Tensor.zeros(new int[]{1}));
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.AfterAllOp;
        }
    }

    /**
     * CPU kernel for stablehlo.all_gather - gather from all replicas.
     * On single CPU, acts as identity (only one replica).
     */
    public static class AllGatherKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            if (inputs.isEmpty()) {
                throw new IllegalArgumentException("all_gather requires at least 1 input");
            }
            // Single replica: just return the input
            List<Tensor> outputs = new ArrayList<>();
            for (Tensor input : inputs) {
                outputs.add(input.copy());
            }
            return outputs;
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.AllGatherOp;
        }
    }

    /**
     * CPU kernel for stablehlo.all_reduce - reduce across all replicas.
     * On single CPU, acts as identity (only one replica).
     */
    public static class AllReduceKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            if (inputs.isEmpty()) {
                throw new IllegalArgumentException("all_reduce requires at least 1 input");
            }
            // Single replica: reduction with one element is identity
            List<Tensor> outputs = new ArrayList<>();
            for (Tensor input : inputs) {
                outputs.add(input.copy());
            }
            return outputs;
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.AllReduceOp;
        }
    }

    /**
     * CPU kernel for stablehlo.all_to_all - all-to-all communication.
     * On single CPU, acts as identity (only one replica).
     */
    public static class AllToAllKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            if (inputs.isEmpty()) {
                throw new IllegalArgumentException("all_to_all requires at least 1 input");
            }
            // Single replica: all-to-all with one replica is identity
            List<Tensor> outputs = new ArrayList<>();
            for (Tensor input : inputs) {
                outputs.add(input.copy());
            }
            return outputs;
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.AllToAllOp;
        }
    }

    /**
     * CPU kernel for stablehlo.collective_broadcast - broadcast from one replica.
     * On single CPU, acts as identity (only one replica).
     */
    public static class CollectiveBroadcastKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            if (inputs.isEmpty()) {
                throw new IllegalArgumentException("collective_broadcast requires at least 1 input");
            }
            return List.of(inputs.get(0).copy());
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.CollectiveBroadcastOp;
        }
    }

    /**
     * CPU kernel for stablehlo.collective_permute - permute across replicas.
     * On single CPU, acts as identity (only one replica).
     */
    public static class CollectivePermuteKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            if (inputs.isEmpty()) {
                throw new IllegalArgumentException("collective_permute requires at least 1 input");
            }
            return List.of(inputs.get(0).copy());
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.CollectivePermuteOp;
        }
    }

    /**
     * CPU kernel for stablehlo.infeed - receive data from host.
     * On single CPU, returns zeros (no actual infeed).
     */
    public static class InfeedKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            // Return a token and placeholder data
            return List.of(
                Tensor.zeros(new int[]{1}),  // token
                Tensor.zeros(new int[]{1})   // placeholder data
            );
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.InfeedOp;
        }
    }

    /**
     * CPU kernel for stablehlo.outfeed - send data to host.
     * On single CPU, discards data and returns token.
     */
    public static class OutfeedKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            // Return just a token
            return List.of(Tensor.zeros(new int[]{1}));
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.OutfeedOp;
        }
    }

    /**
     * CPU kernel for stablehlo.partition_id - get current partition ID.
     * On single CPU, always returns 0 (only one partition).
     */
    public static class PartitionIdKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            Tensor output = Tensor.zeros(new int[]{});  // Scalar
            output.copyFrom(new float[]{0});
            return List.of(output);
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.PartitionIdOp;
        }
    }

    /**
     * CPU kernel for stablehlo.recv - receive from another device.
     * On single CPU, returns zeros (no actual communication).
     */
    public static class RecvKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            // Return token and placeholder data
            return List.of(
                Tensor.zeros(new int[]{1}),  // token
                Tensor.zeros(new int[]{1})   // placeholder data
            );
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.RecvOp;
        }
    }

    /**
     * CPU kernel for stablehlo.reduce_scatter - reduce then scatter.
     * On single CPU, acts as identity (only one replica).
     */
    public static class ReduceScatterKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            if (inputs.isEmpty()) {
                throw new IllegalArgumentException("reduce_scatter requires at least 1 input");
            }
            // Single replica: reduce-scatter with one replica is identity
            List<Tensor> outputs = new ArrayList<>();
            for (Tensor input : inputs) {
                outputs.add(input.copy());
            }
            return outputs;
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.ReduceScatterOp;
        }
    }

    /**
     * CPU kernel for stablehlo.replica_id - get current replica ID.
     * On single CPU, always returns 0 (only one replica).
     */
    public static class ReplicaIdKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            Tensor output = Tensor.zeros(new int[]{});  // Scalar
            output.copyFrom(new float[]{0});
            return List.of(output);
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.ReplicaIdOp;
        }
    }

    /**
     * CPU kernel for stablehlo.send - send to another device.
     * On single CPU, discards data and returns token.
     */
    public static class SendKernel implements OpKernel {
        @Override
        public List<Tensor> execute(Operation op, List<Tensor> inputs) {
            // Return just a token
            return List.of(Tensor.zeros(new int[]{1}));
        }

        @Override
        public boolean supports(Operation op) {
            return op instanceof StableHloAst.SendOp;
        }
    }
}
