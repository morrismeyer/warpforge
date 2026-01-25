# IntelliJ IDEA Configuration

This directory contains shared IntelliJ IDEA configuration files for WarpForge development.

## Live Templates

The `WarpForge.xml` file contains live templates for dimension-typed tensor operations.

### Installation

1. Open IntelliJ IDEA
2. Go to **File > Manage IDE Settings > Import Settings...**
3. Select `WarpForge.xml` from this directory
4. Check "Live templates" and click OK

Or manually:
1. Copy `WarpForge.xml` to your IntelliJ templates directory:
   - **macOS**: `~/Library/Application Support/JetBrains/IntelliJIdea2024.x/templates/`
   - **Linux**: `~/.config/JetBrains/IntelliJIdea2024.x/templates/`
   - **Windows**: `%APPDATA%\JetBrains\IntelliJIdea2024.x\templates\`
2. Restart IntelliJ IDEA

### Available Templates

Type the abbreviation and press **Tab** to expand:

| Abbreviation | Description |
|--------------|-------------|
| `dimdef` | Define a dimension marker interface |
| `dimdefs` | Define multiple dimension markers |
| `dimvec` | Create a DimVector tensor |
| `dimmat` | Create a DimMatrix tensor |
| `dimr3` | Create a DimRank3 tensor |
| `dimr4` | Create a DimRank4 tensor |
| `matmul` | Matrix multiplication with shape comment |
| `bmatmul` | Batched matmul (rank-3) |
| `bmatmul4` | Batched matmul (rank-4, multi-head attention) |
| `matvec` | Matrix-vector multiplication |
| `transpose` | Matrix transpose |
| `attention` | Full multi-head attention pattern |
| `mlp` | MLP layer pattern |
| `dimops-imports` | Import all DimOps classes |

### Example Usage

1. Type `dimmat` in a Java file
2. Press **Tab**
3. Fill in the placeholders (dimension types, variable name, sizes)
4. Press **Tab** to move between placeholders
5. Press **Enter** when done

The template expands to:
```java
TypedTensor<DimMatrix<M, N>, F32, Cpu> matrix = TypedTensor.zeros(
        new DimMatrix<>(32, 768), F32.INSTANCE, Cpu.INSTANCE);
```
