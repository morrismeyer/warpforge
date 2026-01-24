# WarpForge Benchmark: BERT Fine-tuning on SQuAD

This document describes the end-to-end AI/ML benchmark that validates the WarpForge stack,
integrating SnakeGrinder, SnakeBurger, and WarpForge with distributed collectives across
the Mark1 lab's heterogeneous GPU cluster.

## Benchmark Selection Rationale

We selected BERT fine-tuning on SQuAD as the primary benchmark because it satisfies all
requirements for a comprehensive WarpForge validation:

```
+---------------------+----------------------------------------------------------+
| Criterion           | BERT on SQuAD                                            |
+---------------------+----------------------------------------------------------+
| Dataset size        | 35 MB download, ~125 MB total (fits in memory)           |
| Target metric       | F1 > 88% (BERT-base), F1 > 90% (BERT-large)              |
| Training time       | 24 min BERT-base, 68 min BERT-large (single V100)        |
| Industry status     | MLPerf standard until v5.1, widely cited benchmark       |
| Collectives used    | AllReduce (gradients), Barrier (sync), AllGather (opt)   |
| Traceable           | Yes - torch.fx.symbolic_trace compatible                 |
| Reproducible        | Fixed dataset, deterministic with seed                   |
+---------------------+----------------------------------------------------------+
```

### Comparison with Alternatives

```
+-------------------+-------------+----------+------------------------+---------------------+
| Benchmark         | Dataset     | Runtime  | Pros                   | Cons                |
+-------------------+-------------+----------+------------------------+---------------------+
| BERT/SQuAD        | 35 MB       | 30-60min | Small, well-defined    | Older benchmark     |
| ResNet-50/ImageNet| 150 GB      | Hours    | Classic MLPerf         | Huge dataset        |
| Llama 3.1 8B      | 100+ GB     | Hours    | Current MLPerf v5.1    | Too large for night |
| MNIST             | 11 MB       | Minutes  | Trivial to run         | Not representative  |
| CIFAR-10/ResNet   | 170 MB      | 30 min   | Moderate size          | Less industry use   |
+-------------------+-------------+----------+------------------------+---------------------+
```

## Expected Results

Based on industry benchmarks from DeepSpeed and HuggingFace:

```
+------------------+---------------+---------------+
| Metric           | BERT-base     | BERT-large    |
+------------------+---------------+---------------+
| F1 Score         | 88.5%         | 90.5-93%      |
| Exact Match      | 81.2%         | 84.4%         |
| Training Time    | 24 min        | 68 min        |
| (single V100)    |               |               |
+------------------+---------------+---------------+
```

For distributed training on Mark1 (2 nodes, NVIDIA + AMD):

```
+------------------+---------------+---------------+
| Metric           | Target        | Tolerance     |
+------------------+---------------+---------------+
| F1 Score         | > 88.0%       | +/- 0.5%      |
| Loss Convergence | < 0.5         | Monotonic     |
| Collective Perf  | > 30 Gbps     | AllReduce avg |
| Training Time    | < 45 min      | BERT-base     |
+------------------+---------------+---------------+
```

## Architecture

### Phase 1: PyTorch + WarpForge Collectives (Nightly CI Target)

This phase validates that warpforge-io collectives work correctly in real distributed
training across heterogeneous GPUs.

```
+----------------------------------+        +----------------------------------+
|  NVIDIA Node (mark1nvidia)       |        |  AMD Node (mark1amd)             |
+----------------------------------+        +----------------------------------+
|                                  |        |                                  |
|  PyTorch BERT Model              |        |  PyTorch BERT Model              |
|         |                        |        |         |                        |
|         v                        |        |         v                        |
|  Forward Pass (Embeddings,       |        |  Forward Pass (Embeddings,       |
|    Attention, FFN layers)        |        |    Attention, FFN layers)        |
|         |                        |        |         |                        |
|         v                        |        |         v                        |
|  Backward Pass (Gradients)       |        |  Backward Pass (Gradients)       |
|         |                        |        |         |                        |
|         v                        |        |         v                        |
|  +----------------------------+  |        |  +----------------------------+  |
|  | warpforge-io               |  |<------>|  | warpforge-io               |  |
|  | AllReduce(gradients)       |  |  UCC   |  | AllReduce(gradients)       |  |
|  | over Mellanox 100GbE RDMA  |  |  RDMA  |  | over Mellanox 100GbE RDMA  |  |
|  +----------------------------+  |        |  +----------------------------+  |
|         |                        |        |         |                        |
|         v                        |        |         v                        |
|  Optimizer Step (Adam)           |        |  Optimizer Step (Adam)           |
|                                  |        |                                  |
+----------------------------------+        +----------------------------------+
```

**What this validates:**
- UCC collectives work correctly for gradient synchronization
- Heterogeneous GPU training (NVIDIA + AMD) produces correct results
- RDMA performance is utilized effectively
- Loss converges to expected values

### Phase 2: Full WarpForge Stack (Future Target)

This phase validates the complete compilation and execution pipeline.

```
+------------------------------------------------------------------+
|  PyTorch BERT Model (nn.Module)                                  |
+------------------------------------------------------------------+
                              |
                              v torch.fx.symbolic_trace
+------------------------------------------------------------------+
|  SnakeGrinder                                                    |
|  +------------------------------------------------------------+  |
|  | FX Graph -> StableHLO Converter                            |  |
|  | - Attention ops -> stablehlo.dot_general                   |  |
|  | - Layer norms -> stablehlo.reduce + stablehlo.broadcast    |  |
|  | - Activations -> stablehlo.custom_call("gelu")             |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                              |
                              v  .mlir file (StableHLO text format)
+------------------------------------------------------------------+
|  SnakeBurger                                                     |
|  +------------------------------------------------------------+  |
|  | StableHLO Parser                                           |  |
|  |        |                                                   |  |
|  |        v                                                   |  |
|  | Type Checker                                               |  |
|  |        |                                                   |  |
|  |        v                                                   |  |
|  | Babylon Code Reflection IR                                 |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                              |
                              v
+----------------------------------+        +----------------------------------+
|  WarpForge Backend (NVIDIA)      |        |  WarpForge Backend (AMD)         |
+----------------------------------+        +----------------------------------+
|  - cuBLAS for matmuls            |        |  - hipBLAS for matmuls           |
|  - cuDNN for attention           |        |  - MIOpen for attention          |
|  - Custom CUDA kernels           |        |  - Custom HIP kernels            |
+----------------------------------+        +----------------------------------+
          |                                           |
          +-------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  warpforge-io Collectives (UCC over Mellanox RDMA)               |
|  - AllReduce for gradient synchronization                        |
|  - Barrier for epoch boundaries                                  |
|  - AllGather for model checkpointing                             |
+------------------------------------------------------------------+
```

## Nightly Build Integration

### Tiered Testing Strategy

```
+----------+----------+------------------+----------------------------------------+
| Tier     | Duration | Configuration    | What it validates                      |
+----------+----------+------------------+----------------------------------------+
| Quick    | 5 min    | 1 epoch          | Collectives work, loss decreases       |
|          |          | batch_size=8     | No crashes or hangs                    |
+----------+----------+------------------+----------------------------------------+
| Nightly  | 30 min   | 2 epochs         | Convergence trajectory is correct      |
|          |          | batch_size=16    | Collective performance > 30 Gbps       |
+----------+----------+------------------+----------------------------------------+
| Weekend  | 2 hrs    | Full training    | F1 > 88% achieved                      |
|          |          | batch_size=32    | Full regression validation             |
+----------+----------+------------------+----------------------------------------+
```

### CI Workflow Integration

The benchmark integrates with the nightly build as Phase 8:

```
Phase 1-3: Clean, Build, Unit Tests (existing)
Phase 4:   SnakeGrinder Distribution Tests (existing)
Phase 5:   Native Image Build (existing)
Phase 6:   GPU Box Tests (existing)
Phase 7:   Head-to-Head Collective Benchmark (existing)
Phase 8:   BERT/SQuAD Training Benchmark (NEW)
           - Download SQuAD dataset (cached)
           - Run distributed BERT training
           - Validate loss convergence
           - Check collective performance
           - Compare against baseline metrics
```

### Regression Detection

The benchmark tracks these metrics and fails if regression exceeds threshold:

```
+----------------------+----------------+-------------+
| Metric               | Baseline       | Threshold   |
+----------------------+----------------+-------------+
| Final Loss           | < 0.5          | +20%        |
| F1 Score (weekend)   | > 88.0%        | -1.0%       |
| AllReduce Throughput | > 30 Gbps      | -15%        |
| Training Throughput  | > 50 samples/s | -20%        |
| GPU Memory Usage     | < 12 GB        | +25%        |
+----------------------+----------------+-------------+
```

## Implementation Plan

### Directory Structure

```
warpforge-benchmark/
+-- build.gradle                    # Gradle build configuration
+-- src/
|   +-- main/
|   |   +-- java/
|   |   |   +-- io/surfworks/warpforge/benchmark/
|   |   |       +-- BertSquadBenchmark.java      # Main benchmark runner
|   |   |       +-- DistributedTrainer.java      # PyTorch training coordination
|   |   |       +-- MetricsCollector.java        # Loss, F1, throughput tracking
|   |   |       +-- BaselineValidator.java       # Regression detection
|   |   +-- python/
|   |       +-- bert_squad_training.py           # PyTorch training script
|   |       +-- model_tracing.py                 # SnakeGrinder integration
|   |       +-- evaluate_squad.py                # F1 score computation
|   +-- test/
|       +-- java/
|           +-- BertSquadBenchmarkTest.java      # Smoke tests
+-- datasets/
|   +-- squad/                                   # Cached SQuAD dataset
+-- baselines/
|   +-- bert-base-metrics.json                   # Expected metrics
+-- results/
    +-- nightly-YYYYMMDD/                        # Results by date
```

### Gradle Tasks

```
+-------------------------------+------------------------------------------------+
| Task                          | Description                                    |
+-------------------------------+------------------------------------------------+
| :warpforge-benchmark:quick    | 5-min smoke test (1 epoch)                     |
| :warpforge-benchmark:nightly  | 30-min nightly run (2 epochs)                  |
| :warpforge-benchmark:weekend  | 2-hr full training (convergence)               |
| :warpforge-benchmark:baseline | Update baseline metrics from current run       |
| :warpforge-benchmark:report   | Generate HTML report from results              |
+-------------------------------+------------------------------------------------+
```

### Key Dependencies

```
+-------------------+---------+----------------------------------------------+
| Dependency        | Version | Purpose                                      |
+-------------------+---------+----------------------------------------------+
| PyTorch           | 2.7.0   | Model training (via GraalPy)                 |
| Transformers      | 4.40+   | BERT model and tokenizer                     |
| Datasets          | 2.18+   | SQuAD dataset loading                        |
| warpforge-io      | 0.0.1   | UCC collectives for distributed training     |
| snakegrinder-dist | 0.0.1   | Model tracing to StableHLO (Phase 2)         |
+-------------------+---------+----------------------------------------------+
```

## Mixed GPU Cluster Considerations

The Mark1 lab runs a heterogeneous cluster (NVIDIA + AMD), which presents unique challenges:

```
+-------------------------+--------------------------------------------------+
| Challenge               | WarpForge Solution                               |
+-------------------------+--------------------------------------------------+
| Different GPU vendors   | UCC/UCX abstracts transport layer                |
| Different memory sizes  | Dynamic batch sizing per device                  |
| Different compute rates | Gradient accumulation to balance load            |
| Different frameworks    | warpforge-io provides unified collective API     |
+-------------------------+--------------------------------------------------+
```

This benchmark validates that WarpForge enables **vendor-agnostic distributed training**,
a key differentiator from NCCL-only solutions.

## Dataset and Licensing

```
+-------------+------------------------------------------------------------------+
| Item        | Details                                                          |
+-------------+------------------------------------------------------------------+
| Dataset     | SQuAD v1.1 (Stanford Question Answering Dataset)                 |
| Size        | 35 MB download, 125 MB on disk                                   |
| License     | CC BY-SA 4.0                                                     |
| Source      | https://huggingface.co/datasets/rajpurkar/squad                  |
| Caching     | Downloaded once, cached in datasets/ directory                   |
+-------------+------------------------------------------------------------------+
```

## References

- [MLPerf Training Benchmark](https://mlcommons.org/benchmarks/training/)
- [HuggingFace SQuAD Dataset](https://huggingface.co/datasets/rajpurkar/squad)
- [DeepSpeed BERT Fine-tuning](https://www.deepspeed.ai/tutorials/bert-finetuning/)
- [Mixed AMD/NVIDIA Distributed Training](https://home.mlops.community/public/blogs/distributed-training-in-mlops-break-gpu-vendor-lock-in-distributed-mlops-across-mixed-amd-and-nvidia-clusters)
- [PyTorch Distributed Data Parallel](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)

## Future Work

1. **Phase 2 Implementation** - Full SnakeGrinder -> SnakeBurger -> WarpForge pipeline
2. **GPU Direct RDMA** - Bypass host memory for gradient synchronization
3. **Larger Models** - BERT-large, GPT-2, eventually Llama
4. **Multi-node Scaling** - Extend beyond 2 nodes when hardware available
5. **Automated Baseline Updates** - CI automatically updates baselines on improvement
