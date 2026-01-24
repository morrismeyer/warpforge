# WarpForge Benchmark Suite

This document describes the end-to-end AI/ML benchmark suite that validates the WarpForge stack
by comparing WarpForge execution against PyTorch golden reference results.

## Benchmark Philosophy: Golden Reference Validation

Every benchmark follows a **two-pass validation** approach:

```
+===========================================================================+
|                        TWO-PASS VALIDATION                                |
+===========================================================================+
|                                                                           |
|  PASS 1: PyTorch Golden Reference                                         |
|  ─────────────────────────────────                                        |
|  - Pure PyTorch with standard distributed training (NCCL/RCCL/Gloo)       |
|  - NO WarpForge components                                                |
|  - Produces golden reference:                                             |
|    * Final model weights/checkpoint                                       |
|    * Loss curve (per iteration)                                           |
|    * Final metrics (F1, mAP, perplexity)                                  |
|    * Intermediate activations at checkpoints                              |
|                                                                           |
|  PASS 2: WarpForge Validation                                             |
|  ────────────────────────────                                             |
|  - Full WarpForge stack:                                                  |
|    * SnakeGrinder: PyTorch model -> StableHLO                             |
|    * SnakeBurger: StableHLO -> Babylon IR                                 |
|    * WarpForge Backends: Execute on NVIDIA/AMD GPUs                       |
|    * warpforge-io: UCC collectives for distributed sync                   |
|  - Compares against golden reference:                                     |
|    * Weights match within tolerance                                       |
|    * Loss curve matches                                                   |
|    * Final metrics match                                                  |
|    * Intermediate activations match (numerical correctness)               |
|                                                                           |
+===========================================================================+
```

This approach ensures WarpForge produces **numerically identical results** to PyTorch,
validating both correctness and the full compilation/execution pipeline.

## Mark1 Lab Hardware

```
+-------------+-------------------------+--------+---------------------------+
| Node        | GPU                     | VRAM   | Notes                     |
+-------------+-------------------------+--------+---------------------------+
| mark1nvidia | NVIDIA GeForce RTX 4080 | 16 GB  | Ada Lovelace, CUDA 12.x   |
| mark1amd    | AMD Radeon RX 9070 XT   | 16 GB  | RDNA 4, ROCm 6.x          |
+-------------+-------------------------+--------+---------------------------+
| Interconnect| Mellanox ConnectX-5     | 100GbE | RDMA, ~56 Gbps achieved   |
+-------------+-------------------------+--------+---------------------------+
```

## Benchmark Schedule

```
+============+=======================+================+=======================+
| Schedule   | Benchmark             | Duration       | License               |
+============+=======================+================+=======================+
| Nightly    | BERT-large / SQuAD    | 2-3 hours      | CC BY-SA 4.0          |
| (weekdays) | (NLP, Transformers)   | (both passes)  | (commercial OK)       |
+------------+-----------------------+----------------+-----------------------+
| Saturday   | Faster R-CNN / COCO   | 10-12 hours    | CC BY 4.0             |
| overnight  | (Vision, Detection)   | (both passes)  | (commercial OK)       |
+------------+-----------------------+----------------+-----------------------+
| Sunday     | Llama 3.1 8B / QLoRA  | 8-12 hours     | Meta Community        |
| overnight  | (LLM, Fine-tuning)    | (both passes)  | (commercial OK)       |
+============+=======================+================+=======================+
```

All datasets use commercially-friendly licenses suitable for WarpForge as a product.

## Benchmark 1: BERT-large on SQuAD (Nightly)

### Overview

```
+-------------------+----------------------------------------------------------+
| Item              | Details                                                  |
+-------------------+----------------------------------------------------------+
| Model             | BERT-large-uncased (340M parameters)                     |
| Dataset           | SQuAD v1.1 (35 MB download, 100K+ QA pairs)              |
| Task              | Question Answering fine-tuning                           |
| Target Metric     | F1 > 90%, Exact Match > 84%                              |
| Training Time     | ~90 min per pass (2 passes = ~3 hours total)             |
| VRAM Required     | ~14 GB (fits in 16 GB with batch_size=8)                 |
| License           | CC BY-SA 4.0 (commercial OK with attribution)            |
| Download          | https://huggingface.co/datasets/rajpurkar/squad          |
+-------------------+----------------------------------------------------------+
```

### Why BERT-large/SQuAD

- **Transformer architecture** - Dominant in modern AI, exercises attention ops
- **Small dataset** - 35 MB, downloads in seconds
- **Well-defined metrics** - F1 score has clear expected values
- **Fast iteration** - Full training in ~90 minutes per pass
- **Industry standard** - Was MLPerf benchmark until v5.1

### Pass 1: PyTorch Golden Reference

```
+----------------------------------+        +----------------------------------+
|  mark1nvidia (RTX 4080)          |        |  mark1amd (RX 9070 XT)           |
+----------------------------------+        +----------------------------------+
|                                  |        |                                  |
|  PyTorch BERT-large              |        |  PyTorch BERT-large              |
|  HuggingFace Transformers        |        |  HuggingFace Transformers        |
|         |                        |        |         |                        |
|         v                        |        |         v                        |
|  Forward/Backward Pass           |        |  Forward/Backward Pass           |
|         |                        |        |         |                        |
|         v                        |        |         v                        |
|  +----------------------------+  |        |  +----------------------------+  |
|  | PyTorch DDP                |  |<------>|  | PyTorch DDP                |  |
|  | NCCL/Gloo AllReduce        |  |  TCP   |  | RCCL/Gloo AllReduce        |  |
|  | (standard PyTorch)         |  |        |  | (standard PyTorch)         |  |
|  +----------------------------+  |        |  +----------------------------+  |
|         |                        |        |         |                        |
|         v                        |        |         v                        |
|  Adam Optimizer Step             |        |  Adam Optimizer Step             |
|                                  |        |                                  |
+----------------------------------+        +----------------------------------+
                              |
                              v
                 +---------------------------+
                 | Golden Reference Output   |
                 +---------------------------+
                 | - checkpoint.pt           |
                 | - loss_curve.json         |
                 | - metrics.json (F1, EM)   |
                 | - activations/*.pt        |
                 +---------------------------+
```

### Pass 2: WarpForge Validation

```
+------------------------------------------------------------------+
|  PyTorch BERT-large Model (nn.Module)                            |
+------------------------------------------------------------------+
                              |
                              v torch.fx.symbolic_trace
+------------------------------------------------------------------+
|  SnakeGrinder                                                    |
|  - Trace BERT encoder layers                                     |
|  - Convert attention: matmul -> stablehlo.dot_general            |
|  - Convert LayerNorm -> stablehlo.reduce + broadcast             |
|  - Convert GELU -> stablehlo.custom_call                         |
|  Output: bert_large.mlir                                         |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  SnakeBurger                                                     |
|  - Parse StableHLO MLIR                                          |
|  - Type check and validate                                       |
|  - Convert to Babylon Code Reflection IR                         |
+------------------------------------------------------------------+
                              |
                              v
+----------------------------------+        +----------------------------------+
|  WarpForge Backend (NVIDIA)      |        |  WarpForge Backend (AMD)         |
|  - cuBLAS for attention matmuls  |        |  - hipBLAS for attention matmuls |
|  - Custom CUDA kernels           |        |  - Custom HIP kernels            |
+----------------------------------+        +----------------------------------+
          |                                           |
          +-------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  warpforge-io Collectives (UCC over Mellanox RDMA)               |
|  - AllReduce for gradient synchronization                        |
|  - ~56 Gbps actual throughput                                    |
+------------------------------------------------------------------+
                              |
                              v
                 +---------------------------+
                 | Comparison vs Golden      |
                 +---------------------------+
                 | torch.allclose(golden,    |
                 |   warpforge, rtol=1e-4)   |
                 | Loss curves within 1%     |
                 | F1 within 0.5%            |
                 +---------------------------+
```

### Expected Results

```
+----------------------+------------------+------------------+
| Metric               | Golden (PyTorch) | WarpForge Delta  |
+----------------------+------------------+------------------+
| F1 Score             | > 90.0%          | within 0.5%      |
| Exact Match          | > 84.0%          | within 0.5%      |
| Final Loss           | < 0.5            | within 1%        |
| Training Throughput  | baseline         | within 15%       |
+----------------------+------------------+------------------+
```

## Benchmark 2: Faster R-CNN on COCO (Saturday Overnight)

### Overview

```
+-------------------+----------------------------------------------------------+
| Item              | Details                                                  |
+-------------------+----------------------------------------------------------+
| Model             | Faster R-CNN with ResNet-50 FPN backbone                 |
| Dataset           | COCO 2017 (118K train, 5K val images)                    |
| Task              | Object Detection (80 classes)                            |
| Target Metric     | mAP > 37% (standard baseline)                            |
| Training Time     | ~5-6 hours per pass (2 passes = ~10-12 hours)            |
| VRAM Required     | ~10 GB (batch_size=2 per GPU)                            |
| Dataset Size      | ~25 GB                                                   |
| License           | CC BY 4.0 (commercial OK with attribution)               |
| Download          | https://cocodataset.org/#download                        |
+-------------------+----------------------------------------------------------+
```

### Why Faster R-CNN/COCO (not ImageNet)

```
+-------------------+------------------+------------------------------------------+
| Consideration     | ImageNet         | COCO                                     |
+-------------------+------------------+------------------------------------------+
| License           | Non-commercial   | CC BY 4.0 (commercial OK)                |
| Dataset Size      | 150 GB           | 25 GB                                    |
| Task              | Classification   | Detection (more complex)                 |
| Industry Use      | Research only    | Production deployments OK                |
+-------------------+------------------+------------------------------------------+
```

### Operation Coverage

This benchmark exercises different operations than BERT:

```
+----------------------+------------------+----------------------+
| Operation Type       | BERT-large       | Faster R-CNN         |
+----------------------+------------------+----------------------+
| Attention (matmul)   | Primary          | None                 |
| Convolutions         | None             | Primary (ResNet FPN) |
| Batch Normalization  | Layer Norm       | Batch Norm           |
| Pooling              | None             | ROI Pooling, MaxPool |
| Non-Max Suppression  | None             | NMS post-processing  |
| Anchor Generation    | None             | RPN anchors          |
| Skip Connections     | Residual in FFN  | ResNet skip blocks   |
| Activation           | GELU             | ReLU                 |
+----------------------+------------------+----------------------+
```

### Training Configuration

```
+----------------------+------------------------------------------+
| Parameter            | Value                                    |
+----------------------+------------------------------------------+
| Epochs               | 12 (1x schedule)                         |
| Batch size (global)  | 4 (2 per GPU x 2 GPUs)                   |
| Optimizer            | SGD with momentum (0.9)                  |
| Learning rate        | 0.02 with step decay at 8, 11 epochs     |
| Weight decay         | 1e-4                                     |
| Image size           | 800 x 1333 (short side 800)              |
| Backbone             | ResNet-50 FPN (pretrained on ImageNet)   |
+----------------------+------------------------------------------+
```

Note: Using pretrained backbone is standard practice and doesn't require ImageNet license
since we use weights, not the dataset itself.

### Expected Results

```
+----------------------+------------------+------------------+
| Metric               | Golden (PyTorch) | WarpForge Delta  |
+----------------------+------------------+------------------+
| mAP @ IoU=0.50:0.95  | > 37.0%          | within 0.3%      |
| mAP @ IoU=0.50       | > 58.0%          | within 0.3%      |
| mAP @ IoU=0.75       | > 40.0%          | within 0.3%      |
| Final Loss           | < 0.5            | within 1%        |
| Images/sec           | baseline         | within 15%       |
+----------------------+------------------+------------------+
```

## Benchmark 3: Llama 3.1 8B Fine-tuning (Sunday Overnight)

### Overview

```
+-------------------+----------------------------------------------------------+
| Item              | Details                                                  |
+-------------------+----------------------------------------------------------+
| Model             | Llama 3.1 8B (8 billion parameters)                      |
| Method            | QLoRA (4-bit quantization + LoRA adapters)               |
| Dataset           | Alpaca-style instruction tuning (~52K examples)          |
| Task              | Instruction following fine-tuning                        |
| Target Metric     | Loss convergence, perplexity improvement                 |
| Training Time     | ~4-5 hours per pass (2 passes = ~8-10 hours)             |
| VRAM Required     | ~12-14 GB with QLoRA (fits in 16 GB)                     |
| License           | Meta Llama 3.1 Community License (commercial OK)         |
| Download          | https://huggingface.co/meta-llama/Llama-3.1-8B           |
+-------------------+----------------------------------------------------------+
```

### Why QLoRA

Full fine-tuning of 8B parameters requires ~64 GB VRAM. QLoRA enables training on consumer GPUs:

```
+----------------------+------------------+------------------+
| Approach             | VRAM Required    | Trainable Params |
+----------------------+------------------+------------------+
| Full Fine-tuning     | ~64 GB           | 8B (100%)        |
| LoRA (rank=64)       | ~24 GB           | ~40M (0.5%)      |
| QLoRA (4-bit + LoRA) | ~12 GB           | ~40M (0.5%)      |
+----------------------+------------------+------------------+
```

### Training Configuration

```
+----------------------+------------------------------------------+
| Parameter            | Value                                    |
+----------------------+------------------------------------------+
| Quantization         | 4-bit NormalFloat (NF4)                  |
| LoRA rank            | 64                                       |
| LoRA alpha           | 16                                       |
| LoRA target modules  | q_proj, k_proj, v_proj, o_proj           |
| Batch size (global)  | 8 (4 per GPU x 2 GPUs)                   |
| Gradient accumulation| 4 steps (effective batch 32)             |
| Optimizer            | Paged AdamW 8-bit                        |
| Learning rate        | 2e-4                                     |
| Epochs               | 3                                        |
| Max sequence length  | 2048                                     |
+----------------------+------------------------------------------+
```

### Expected Results

```
+----------------------+------------------+------------------+
| Metric               | Golden (PyTorch) | WarpForge Delta  |
+----------------------+------------------+------------------+
| Final Loss           | < 1.0            | within 2%        |
| Perplexity           | baseline         | within 2%        |
| Training Throughput  | baseline         | within 20%       |
+----------------------+------------------+------------------+
```

## Comparison Tolerances

Numerical comparison between PyTorch golden reference and WarpForge output:

```
+----------------------+-----------+-----------+--------------------------+
| Comparison Type      | rtol      | atol      | Notes                    |
+----------------------+-----------+-----------+--------------------------+
| Weights              | 1e-4      | 1e-6      | After training           |
| Activations          | 1e-3      | 1e-5      | Per-layer checkpoints    |
| Gradients            | 1e-3      | 1e-5      | Before AllReduce         |
| Loss values          | 1e-4      | 1e-6      | Per iteration            |
| Final metrics        | 5e-3      | -         | F1, mAP, perplexity      |
+----------------------+-----------+-----------+--------------------------+
```

Tolerance rationale:
- Floating-point operations may differ in order between PyTorch and WarpForge
- Different GEMM implementations (cuBLAS vs hipBLAS) have different rounding
- AllReduce algorithms may sum in different orders
- These tolerances are tight enough to catch real bugs while allowing for legitimate variation

## CI Integration

### Nightly Workflow (BERT-large/SQuAD) - Monday through Friday

```yaml
# Runs at 2 AM EST
schedule:
  - cron: '0 7 * * 1-5'  # 2 AM EST, Mon-Fri

steps:
  - name: BERT-large Golden Reference (Pass 1)
    run: |
      ./gradlew :warpforge-benchmark:bertSquadGolden \
        --model bert-large-uncased \
        --epochs 2 \
        --batch-size 8

  - name: BERT-large WarpForge Validation (Pass 2)
    run: |
      ./gradlew :warpforge-benchmark:bertSquadWarpforge \
        --model bert-large-uncased \
        --epochs 2 \
        --batch-size 8 \
        --compare-golden results/bert-golden/

  - name: Validate Results Match
    run: |
      ./gradlew :warpforge-benchmark:validateResults \
        --golden results/bert-golden/ \
        --warpforge results/bert-warpforge/ \
        --fail-on-mismatch
```

### Saturday Workflow (Faster R-CNN/COCO)

```yaml
# Runs Saturday at 6 PM EST
schedule:
  - cron: '0 23 * * 6'  # 6 PM EST Saturday

steps:
  - name: Faster R-CNN Golden Reference (Pass 1)
    run: |
      ./gradlew :warpforge-benchmark:cocoDetectionGolden \
        --model faster-rcnn-resnet50-fpn \
        --epochs 12 \
        --batch-size 2

  - name: Faster R-CNN WarpForge Validation (Pass 2)
    run: |
      ./gradlew :warpforge-benchmark:cocoDetectionWarpforge \
        --model faster-rcnn-resnet50-fpn \
        --epochs 12 \
        --batch-size 2 \
        --compare-golden results/coco-golden/

  - name: Validate Results Match
    run: |
      ./gradlew :warpforge-benchmark:validateResults \
        --golden results/coco-golden/ \
        --warpforge results/coco-warpforge/ \
        --fail-on-mismatch
```

### Sunday Workflow (Llama 3.1 8B QLoRA)

```yaml
# Runs Sunday at 12 PM EST
schedule:
  - cron: '0 17 * * 0'  # 12 PM EST Sunday

steps:
  - name: Llama 3.1 8B Golden Reference (Pass 1)
    run: |
      ./gradlew :warpforge-benchmark:llamaQloraGolden \
        --model meta-llama/Llama-3.1-8B \
        --epochs 3 \
        --batch-size 4

  - name: Llama 3.1 8B WarpForge Validation (Pass 2)
    run: |
      ./gradlew :warpforge-benchmark:llamaQloraWarpforge \
        --model meta-llama/Llama-3.1-8B \
        --epochs 3 \
        --batch-size 4 \
        --compare-golden results/llama-golden/

  - name: Validate Results Match
    run: |
      ./gradlew :warpforge-benchmark:validateResults \
        --golden results/llama-golden/ \
        --warpforge results/llama-warpforge/ \
        --fail-on-mismatch
```

## Dataset Management

```
+------------------+----------+------------------+---------------------------+
| Dataset          | Size     | License          | Storage Location          |
+------------------+----------+------------------+---------------------------+
| SQuAD v1.1       | 35 MB    | CC BY-SA 4.0     | ~/.cache/huggingface/     |
| COCO 2017        | 25 GB    | CC BY 4.0        | /data/coco/               |
| Alpaca (instruct)| 50 MB    | CC BY-NC 4.0     | ~/.cache/huggingface/     |
| Llama 3.1 8B     | 16 GB    | Meta Community   | ~/.cache/huggingface/     |
+------------------+----------+------------------+---------------------------+
```

### COCO Setup (One-time)

```bash
# Download COCO 2017 dataset
mkdir -p /data/coco
cd /data/coco

# Training images (18 GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Validation images (1 GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Annotations (241 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### Llama 3.1 Access (One-time)

```bash
# 1. Accept license at https://huggingface.co/meta-llama/Llama-3.1-8B
# 2. Login with HuggingFace CLI
huggingface-cli login

# 3. Model downloads automatically on first use
```

## Implementation Roadmap

### Phase 1: Infrastructure (Week 1-2)

```
+----+--------------------------------------------------+
| #  | Task                                             |
+----+--------------------------------------------------+
| 1  | Create warpforge-benchmark Gradle subproject     |
| 2  | Implement golden reference runner (PyTorch DDP)  |
| 3  | Implement result serialization (checkpoints,     |
|    | loss curves, activations)                        |
| 4  | Implement comparison framework with tolerances   |
| 5  | Add CI workflow stubs for each schedule tier     |
+----+--------------------------------------------------+
```

### Phase 2: BERT/SQuAD Nightly (Week 3-4)

```
+----+--------------------------------------------------+
| #  | Task                                             |
+----+--------------------------------------------------+
| 6  | Implement BERT golden reference training         |
| 7  | Integrate SnakeGrinder for BERT tracing          |
| 8  | Integrate SnakeBurger for BERT IR generation     |
| 9  | Implement WarpForge BERT execution               |
| 10 | Validate end-to-end on Mark1 hardware            |
| 11 | Enable nightly CI                                |
+----+--------------------------------------------------+
```

### Phase 3: Faster R-CNN/COCO Saturday (Week 5-6)

```
+----+--------------------------------------------------+
| #  | Task                                             |
+----+--------------------------------------------------+
| 12 | Download and set up COCO on Mark1 storage        |
| 13 | Implement Faster R-CNN golden reference          |
| 14 | Extend SnakeGrinder for conv/batchnorm/pool/NMS  |
| 15 | Implement WarpForge detection execution          |
| 16 | Validate on Mark1, enable Saturday CI            |
+----+--------------------------------------------------+
```

### Phase 4: Llama 3.1 8B Sunday (Week 7-8)

```
+----+--------------------------------------------------+
| #  | Task                                             |
+----+--------------------------------------------------+
| 17 | Set up Llama 3.1 8B access and download          |
| 18 | Implement QLoRA golden reference training        |
| 19 | Extend SnakeGrinder for LLM-specific ops         |
| 20 | Implement WarpForge Llama execution              |
| 21 | Validate on Mark1, enable Sunday CI              |
+----+--------------------------------------------------+
```

## Success Criteria

The benchmark suite is successful when:

```
+----+------------------------------------------------------------------+
| #  | Criterion                                                        |
+----+------------------------------------------------------------------+
| 1  | All three benchmarks complete both passes without crashes        |
| 2  | WarpForge results match PyTorch golden reference within          |
|    | specified tolerances for all benchmarks                          |
| 3  | Nightly BERT benchmark completes in < 4 hours                    |
| 4  | Saturday COCO benchmark completes in < 14 hours                  |
| 5  | Sunday Llama benchmark completes in < 12 hours                   |
| 6  | CI automatically detects regressions and fails the build         |
| 7  | All datasets use commercially-compatible licenses                |
+----+------------------------------------------------------------------+
```

## References

- [COCO Dataset](https://cocodataset.org/) - CC BY 4.0 license
- [SQuAD Dataset](https://huggingface.co/datasets/rajpurkar/squad) - CC BY-SA 4.0 license
- [Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B) - Meta Community License
- [MLPerf Training Benchmark](https://mlcommons.org/benchmarks/training/)
- [DeepSpeed BERT Training](https://www.deepspeed.ai/2020/05/27/fastest-bert-training.html)
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Unsloth Llama Fine-tuning](https://unsloth.ai/blog/llama3-1)
