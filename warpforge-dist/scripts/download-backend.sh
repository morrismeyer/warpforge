#!/bin/bash
# WarpForge Backend Downloader
# Downloads GPU backend libraries on demand
#
# Usage:
#   download-backend.sh cuda    # Download CUDA backend
#   download-backend.sh rocm    # Download ROCm backend
#   download-backend.sh auto    # Auto-detect and download appropriate backend
#   download-backend.sh list    # List available backends

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$(dirname "$SCRIPT_DIR")"
BACKENDS_DIR="$DIST_DIR/lib/backends"

# Platform detection
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    case "$arch" in
        x86_64|amd64) arch="amd64" ;;
        aarch64|arm64) arch="aarch64" ;;
    esac

    echo "${os}-${arch}"
}

# GPU detection
detect_nvidia() {
    if [[ -e /dev/nvidia0 ]] || command -v nvidia-smi &>/dev/null; then
        return 0
    fi
    return 1
}

detect_amd() {
    if [[ -e /dev/kfd ]] || command -v rocm-smi &>/dev/null; then
        return 0
    fi
    return 1
}

# Read manifest field using basic parsing (no jq dependency)
read_manifest() {
    local manifest="$1"
    local field="$2"
    local platform="$3"

    if [[ -n "$platform" ]]; then
        # Read platform-specific field
        grep -A10 "\"$platform\"" "$manifest" | grep "\"$field\"" | head -1 | cut -d'"' -f4
    else
        # Read top-level field
        grep "\"$field\"" "$manifest" | head -1 | cut -d'"' -f4
    fi
}

# Download and extract backend
download_backend() {
    local backend="$1"
    local manifest="$DIST_DIR/backends/$backend/manifest.json"

    if [[ ! -f "$manifest" ]]; then
        echo "Error: Backend '$backend' manifest not found at $manifest"
        exit 1
    fi

    local platform=$(detect_platform)
    local url=$(read_manifest "$manifest" "url" "$platform")
    local sha256=$(read_manifest "$manifest" "sha256" "$platform")
    local size_mb=$(read_manifest "$manifest" "size_mb" "$platform")
    local version=$(read_manifest "$manifest" "version")

    if [[ -z "$url" ]]; then
        echo "Error: Backend '$backend' not available for platform '$platform'"
        exit 1
    fi

    local dest_dir="$BACKENDS_DIR/$backend"
    local archive="/tmp/warpforge-${backend}-${version}.tar.gz"

    echo "WarpForge Backend Downloader"
    echo "============================"
    echo "Backend:  $backend $version"
    echo "Platform: $platform"
    echo "Size:     ~${size_mb} MB"
    echo "Target:   $dest_dir"
    echo ""

    # Check if already installed
    if [[ -d "$dest_dir" ]]; then
        echo "Backend '$backend' is already installed at $dest_dir"
        read -p "Re-download and reinstall? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping download."
            return 0
        fi
        rm -rf "$dest_dir"
    fi

    # Download
    echo "Downloading $backend backend..."
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$archive" "$url"
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "$archive" "$url"
    else
        echo "Error: Neither curl nor wget found"
        exit 1
    fi

    # Verify checksum (if not placeholder)
    if [[ "$sha256" != "placeholder"* ]] && [[ -n "$sha256" ]]; then
        echo "Verifying checksum..."
        local computed
        if command -v sha256sum &>/dev/null; then
            computed=$(sha256sum "$archive" | cut -d' ' -f1)
        elif command -v shasum &>/dev/null; then
            computed=$(shasum -a 256 "$archive" | cut -d' ' -f1)
        fi

        if [[ "$computed" != "$sha256" ]]; then
            echo "Error: Checksum mismatch!"
            echo "  Expected: $sha256"
            echo "  Got:      $computed"
            rm -f "$archive"
            exit 1
        fi
        echo "Checksum verified."
    fi

    # Extract
    echo "Extracting..."
    mkdir -p "$dest_dir"
    tar -xzf "$archive" -C "$dest_dir" --strip-components=1
    rm -f "$archive"

    echo ""
    echo "Backend '$backend' installed successfully!"
    echo "Libraries installed to: $dest_dir"
    ls -la "$dest_dir"/*.so* 2>/dev/null | head -10 || true
}

# Auto-detect and download
auto_download() {
    echo "Auto-detecting GPU hardware..."

    if detect_nvidia; then
        echo "NVIDIA GPU detected."
        download_backend "cuda"
    elif detect_amd; then
        echo "AMD GPU detected."
        download_backend "rocm"
    else
        echo "No supported GPU detected."
        echo ""
        echo "Supported GPUs:"
        echo "  - NVIDIA GPUs (CUDA)"
        echo "  - AMD GPUs (ROCm)"
        echo ""
        echo "If you have a GPU but it's not detected, try:"
        echo "  $0 cuda   # For NVIDIA"
        echo "  $0 rocm   # For AMD"
        exit 1
    fi
}

# List available backends
list_backends() {
    echo "Available WarpForge GPU Backends"
    echo "================================"
    echo ""

    local platform=$(detect_platform)

    for manifest in "$DIST_DIR/backends"/*/manifest.json; do
        [[ -f "$manifest" ]] || continue

        local backend=$(read_manifest "$manifest" "backend")
        local version=$(read_manifest "$manifest" "version")
        local description=$(read_manifest "$manifest" "description")
        local url=$(read_manifest "$manifest" "url" "$platform")
        local installed="no"

        if [[ -d "$BACKENDS_DIR/$backend" ]]; then
            installed="yes"
        fi

        echo "$backend $version"
        echo "  Description: $description"
        echo "  Platform:    $platform $([ -n "$url" ] && echo "(available)" || echo "(not available)")"
        echo "  Installed:   $installed"
        echo ""
    done

    echo "GPU Detection:"
    echo "  NVIDIA: $(detect_nvidia && echo "detected" || echo "not detected")"
    echo "  AMD:    $(detect_amd && echo "detected" || echo "not detected")"
}

# Check backend status
check_backend() {
    local backend="$1"
    local dest_dir="$BACKENDS_DIR/$backend"

    if [[ -d "$dest_dir" ]]; then
        echo "Backend '$backend' is installed at $dest_dir"
        return 0
    else
        echo "Backend '$backend' is not installed"
        return 1
    fi
}

# Main
case "${1:-}" in
    cuda|rocm)
        download_backend "$1"
        ;;
    auto)
        auto_download
        ;;
    list)
        list_backends
        ;;
    check)
        if [[ -z "${2:-}" ]]; then
            echo "Usage: $0 check <backend>"
            exit 1
        fi
        check_backend "$2"
        ;;
    *)
        echo "WarpForge Backend Downloader"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  cuda           Download NVIDIA CUDA backend"
        echo "  rocm           Download AMD ROCm backend"
        echo "  auto           Auto-detect GPU and download appropriate backend"
        echo "  list           List available backends and detection status"
        echo "  check <name>   Check if a backend is installed"
        echo ""
        exit 1
        ;;
esac
