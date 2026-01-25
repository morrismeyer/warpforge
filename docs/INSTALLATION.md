# WarpForge Installation Guide

WarpForge is a zero-configuration ML inference toolkit. Download, extract, and run - no environment variables, no PATH manipulation, no dependency management required.

## Quick Install (Recommended)

One-liner installation:

```bash
curl -sSL https://raw.githubusercontent.com/surfworks/warpforge/main/warpforge-dist/scripts/install.sh | bash
```

This will:
1. Detect your platform (Linux/macOS, AMD64/ARM64)
2. Download the latest release
3. Install to `~/.warpforge`
4. Print instructions for adding to your PATH

### Install Options

```bash
# Install specific version
WARPFORGE_VERSION=v0.1.0 curl -sSL .../install.sh | bash

# Install to custom location
WARPFORGE_HOME=/opt/warpforge curl -sSL .../install.sh | bash

# Uninstall
~/.warpforge/bin/install.sh --uninstall
```

## Manual Installation

### Step 1: Download

Download the archive for your platform from the [Releases page](https://github.com/surfworks/warpforge/releases):

+--------------------+------------------------------------------+
| Platform           | Archive Name                             |
+--------------------+------------------------------------------+
| Linux AMD64        | warpforge-VERSION-linux-amd64.tar.gz     |
| Linux ARM64        | warpforge-VERSION-linux-arm64.tar.gz     |
| macOS Apple Silicon| warpforge-VERSION-macos-arm64.tar.gz     |
| macOS Intel        | warpforge-VERSION-macos-amd64.tar.gz     |
+--------------------+------------------------------------------+

Example for Linux AMD64:

```bash
VERSION="0.1.0"
curl -LO "https://github.com/surfworks/warpforge/releases/download/v${VERSION}/warpforge-${VERSION}-linux-amd64.tar.gz"
```

### Step 2: Extract

```bash
# Create installation directory
mkdir -p ~/.warpforge

# Extract
tar -xzf warpforge-${VERSION}-linux-amd64.tar.gz -C ~/.warpforge --strip-components=1
```

### Step 3: Add to PATH

Add to your shell configuration file:

```bash
# For bash (~/.bashrc or ~/.bash_profile)
export PATH="$HOME/.warpforge/bin:$PATH"

# For zsh (~/.zshrc)
export PATH="$HOME/.warpforge/bin:$PATH"

# For fish (~/.config/fish/config.fish)
set -gx PATH $HOME/.warpforge/bin $PATH
```

Reload your shell:

```bash
source ~/.bashrc  # or ~/.zshrc
```

### Step 4: Verify Installation

```bash
warpforge --version
warpforge gpu-info
```

## What's Included

The WarpForge distribution is completely self-contained:

```
~/.warpforge/
+-- bin/
|   +-- warpforge           # Main CLI for job submission
|   +-- warpforge-trace     # PyTorch model tracing (StableHLO output)
|   +-- warpforge-burger    # Babylon code reflection tools
|   +-- download-backend.sh # GPU backend installer
+-- lib/
|   +-- python/             # Bundled Python runtime (GraalPy)
|   +-- ucx/                # UCX/UCC libraries (Linux only)
|   +-- backends/           # GPU backends (downloaded on demand)
|       +-- cuda/           # NVIDIA CUDA libraries
|       +-- rocm/           # AMD ROCm libraries
+-- backends/
|   +-- cuda/manifest.json  # CUDA backend metadata
|   +-- rocm/manifest.json  # ROCm backend metadata
+-- conf/
    +-- warpforge.conf      # Configuration file
```

## GPU Support

GPU backends are **not** included in the base distribution to keep download size manageable. They are downloaded on first use.

### Auto-Detection

WarpForge automatically detects your GPU and prompts to download the appropriate backend:

```bash
# Check GPU detection
warpforge gpu-info

# Auto-detect and install backend
warpforge gpu-info --install
```

### Manual Backend Installation

```bash
# NVIDIA (CUDA)
download-backend.sh cuda

# AMD (ROCm)
download-backend.sh rocm

# List available backends
download-backend.sh list
```

### Backend Requirements

+----------+-------------------------------+---------------------------+
| Backend  | Hardware                      | Driver Requirements       |
+----------+-------------------------------+---------------------------+
| CUDA     | NVIDIA GPU (Compute Cap 6.0+) | NVIDIA Driver 535+        |
| ROCm     | AMD GPU (RDNA2+, CDNA)        | ROCm 6.0+ / amdgpu driver |
+----------+-------------------------------+---------------------------+

## Container Images

Pre-built container images are available for deployment:

### Docker

```bash
# Base image (CPU only)
docker pull ghcr.io/surfworks/warpforge:latest

# CUDA image (NVIDIA GPU)
docker pull ghcr.io/surfworks/warpforge:latest-cuda

# Run with GPU
docker run --gpus all ghcr.io/surfworks/warpforge:latest-cuda gpu-info
```

### Singularity (HPC)

```bash
# Pull image
singularity pull warpforge-cuda.sif docker://ghcr.io/surfworks/warpforge:latest-cuda

# Run on HPC with GPU
singularity exec --nv warpforge-cuda.sif warpforge gpu-info
```

## Platform Support

+----------+--------------+------------+---------------------------+
| Platform | Architecture | Status     | Notes                     |
+----------+--------------+------------+---------------------------+
| Linux    | AMD64        | Full       | Primary development       |
| Linux    | ARM64        | Full       | Graviton, Ampere          |
| macOS    | ARM64        | Full       | Apple Silicon (M1/M2/M3)  |
| macOS    | AMD64        | Full       | Intel Macs                |
| Windows  | -            | Not yet    | Use WSL2 with Linux build |
+----------+--------------+------------+---------------------------+

## Troubleshooting

### "command not found: warpforge"

Ensure WarpForge is in your PATH:

```bash
# Check if in PATH
echo $PATH | tr ':' '\n' | grep warpforge

# Add to PATH (adjust for your shell)
export PATH="$HOME/.warpforge/bin:$PATH"
```

### "Permission denied" on macOS

macOS may quarantine downloaded executables. Remove the quarantine attribute:

```bash
xattr -d com.apple.quarantine ~/.warpforge/bin/*
```

### GPU not detected

1. Verify driver installation:
   ```bash
   # NVIDIA
   nvidia-smi

   # AMD
   rocm-smi
   ```

2. Check if device nodes exist:
   ```bash
   # NVIDIA
   ls -la /dev/nvidia*

   # AMD
   ls -la /dev/kfd /dev/dri/render*
   ```

3. Ensure user has GPU access (Linux):
   ```bash
   # Add user to video/render groups
   sudo usermod -aG video,render $USER
   # Log out and back in
   ```

### Backend download fails

1. Check internet connectivity
2. Verify GitHub is accessible
3. Try manual download from [Releases](https://github.com/surfworks/warpforge/releases)

### Performance issues

1. Verify GPU backend is installed:
   ```bash
   download-backend.sh list
   ```

2. Check GPU utilization during inference:
   ```bash
   # NVIDIA
   watch -n 1 nvidia-smi

   # AMD
   watch -n 1 rocm-smi
   ```

## Upgrading

### Using install script

```bash
# Reinstall latest version (removes previous)
curl -sSL .../install.sh | bash
```

### Manual upgrade

```bash
# Download new version
VERSION="0.2.0"
curl -LO "https://github.com/surfworks/warpforge/releases/download/v${VERSION}/warpforge-${VERSION}-linux-amd64.tar.gz"

# Remove old installation
rm -rf ~/.warpforge/bin ~/.warpforge/lib

# Extract new version
tar -xzf warpforge-${VERSION}-linux-amd64.tar.gz -C ~/.warpforge --strip-components=1
```

Note: GPU backends in `~/.warpforge/lib/backends/` are preserved during upgrade.

## Uninstalling

```bash
# Using install script
~/.warpforge/bin/install.sh --uninstall

# Or manually
rm -rf ~/.warpforge

# Remove from shell config
# Edit ~/.bashrc or ~/.zshrc and remove the PATH line
```

## Building from Source

See the [Developer Guide](../CLAUDE.md) for building WarpForge from source.

Quick start:

```bash
git clone https://github.com/surfworks/warpforge.git
cd warpforge
./gradlew :warpforge-dist:distTarGz
# Distribution at: warpforge-dist/build/warpforge-*.tar.gz
```
