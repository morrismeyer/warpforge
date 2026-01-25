#!/bin/bash
# WarpForge Container Build Script
# Builds Docker and Singularity images for WarpForge distribution
#
# Usage:
#   ./build-containers.sh [docker|singularity|all] [base|cpu|cuda|rocm|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$DIST_DIR/build"

# Default version
WARPFORGE_VERSION="${WARPFORGE_VERSION:-0.1.0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[BUILD]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check prerequisites
check_docker() {
    if ! command -v docker &>/dev/null; then
        warn "Docker not found. Skipping Docker builds."
        return 1
    fi
    return 0
}

check_singularity() {
    if ! command -v singularity &>/dev/null; then
        warn "Singularity not found. Skipping Singularity builds."
        return 1
    fi
    return 0
}

check_distribution() {
    if [[ ! -d "$BUILD_DIR/warpforge-$WARPFORGE_VERSION" ]]; then
        error "WarpForge distribution not found at $BUILD_DIR/warpforge-$WARPFORGE_VERSION"
        echo "Run: ./gradlew :warpforge-dist:assembleUnifiedDist"
        exit 1
    fi
}

# Docker builds
build_docker_base() {
    log "Building Docker base image..."
    docker build \
        -f "$SCRIPT_DIR/Dockerfile.base" \
        -t "warpforge:$WARPFORGE_VERSION" \
        -t "warpforge:latest" \
        --build-arg "WARPFORGE_VERSION=$WARPFORGE_VERSION" \
        "$DIST_DIR"
    log "Built: warpforge:$WARPFORGE_VERSION"
}

build_docker_cpu() {
    log "Building Docker CPU image..."
    docker build \
        -f "$SCRIPT_DIR/Dockerfile.cpu" \
        -t "warpforge:$WARPFORGE_VERSION-cpu" \
        --build-arg "WARPFORGE_VERSION=$WARPFORGE_VERSION" \
        "$DIST_DIR"
    log "Built: warpforge:$WARPFORGE_VERSION-cpu"
}

build_docker_cuda() {
    log "Building Docker CUDA image..."
    docker build \
        -f "$SCRIPT_DIR/Dockerfile.cuda" \
        -t "warpforge:$WARPFORGE_VERSION-cuda" \
        --build-arg "WARPFORGE_VERSION=$WARPFORGE_VERSION" \
        "$DIST_DIR"
    log "Built: warpforge:$WARPFORGE_VERSION-cuda"
}

build_docker_rocm() {
    log "Building Docker ROCm image..."
    docker build \
        -f "$SCRIPT_DIR/Dockerfile.rocm" \
        -t "warpforge:$WARPFORGE_VERSION-rocm" \
        --build-arg "WARPFORGE_VERSION=$WARPFORGE_VERSION" \
        "$DIST_DIR"
    log "Built: warpforge:$WARPFORGE_VERSION-rocm"
}

# Singularity builds
build_singularity_cuda() {
    log "Building Singularity CUDA image..."
    cd "$DIST_DIR"
    singularity build \
        "$BUILD_DIR/warpforge-$WARPFORGE_VERSION-cuda.sif" \
        "$SCRIPT_DIR/singularity/warpforge-cuda.def"
    log "Built: $BUILD_DIR/warpforge-$WARPFORGE_VERSION-cuda.sif"
}

build_singularity_rocm() {
    log "Building Singularity ROCm image..."
    cd "$DIST_DIR"
    singularity build \
        "$BUILD_DIR/warpforge-$WARPFORGE_VERSION-rocm.sif" \
        "$SCRIPT_DIR/singularity/warpforge-rocm.def"
    log "Built: $BUILD_DIR/warpforge-$WARPFORGE_VERSION-rocm.sif"
}

# Main
main() {
    local runtime="${1:-all}"
    local variant="${2:-all}"

    log "WarpForge Container Builder"
    log "Version: $WARPFORGE_VERSION"
    log "Runtime: $runtime"
    log "Variant: $variant"
    echo ""

    check_distribution

    case "$runtime" in
        docker)
            check_docker || exit 1
            case "$variant" in
                base) build_docker_base ;;
                cpu) build_docker_cpu ;;
                cuda) build_docker_cuda ;;
                rocm) build_docker_rocm ;;
                all)
                    build_docker_base
                    build_docker_cpu
                    build_docker_cuda
                    build_docker_rocm
                    ;;
                *) error "Unknown variant: $variant" ;;
            esac
            ;;
        singularity)
            check_singularity || exit 1
            case "$variant" in
                cuda) build_singularity_cuda ;;
                rocm) build_singularity_rocm ;;
                all)
                    build_singularity_cuda
                    build_singularity_rocm
                    ;;
                *) error "Unknown variant: $variant (singularity supports: cuda, rocm, all)" ;;
            esac
            ;;
        all)
            if check_docker; then
                case "$variant" in
                    base) build_docker_base ;;
                    cpu) build_docker_cpu ;;
                    cuda) build_docker_cuda ;;
                    rocm) build_docker_rocm ;;
                    all)
                        build_docker_base
                        build_docker_cpu
                        build_docker_cuda
                        build_docker_rocm
                        ;;
                esac
            fi
            if check_singularity; then
                case "$variant" in
                    cuda) build_singularity_cuda ;;
                    rocm) build_singularity_rocm ;;
                    all)
                        build_singularity_cuda
                        build_singularity_rocm
                        ;;
                esac
            fi
            ;;
        *)
            echo "Usage: $0 [docker|singularity|all] [base|cpu|cuda|rocm|all]"
            echo ""
            echo "Examples:"
            echo "  $0 docker all       # Build all Docker images"
            echo "  $0 docker cuda      # Build CUDA Docker image only"
            echo "  $0 singularity all  # Build all Singularity images"
            echo "  $0 all all          # Build everything"
            exit 1
            ;;
    esac

    echo ""
    log "Container build complete!"
}

main "$@"
