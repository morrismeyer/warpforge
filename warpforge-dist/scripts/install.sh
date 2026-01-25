#!/bin/bash
# WarpForge Installer
# One-liner installation for WarpForge distribution
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/surfworks/warpforge/main/warpforge-dist/scripts/install.sh | bash
#
# Environment variables:
#   WARPFORGE_VERSION  - Version to install (default: latest)
#   WARPFORGE_HOME     - Installation directory (default: ~/.warpforge)
#
# Examples:
#   # Install latest version
#   curl -sSL https://warpforge.dev/install.sh | bash
#
#   # Install specific version
#   WARPFORGE_VERSION=v0.1.0 curl -sSL https://warpforge.dev/install.sh | bash
#
#   # Install to custom location
#   WARPFORGE_HOME=/opt/warpforge curl -sSL https://warpforge.dev/install.sh | bash

set -e

# Configuration
WARPFORGE_VERSION="${WARPFORGE_VERSION:-latest}"
WARPFORGE_HOME="${WARPFORGE_HOME:-$HOME/.warpforge}"
GITHUB_REPO="surfworks/warpforge"
GITHUB_API="https://api.github.com/repos/$GITHUB_REPO"
GITHUB_RELEASES="https://github.com/$GITHUB_REPO/releases"

# Colors (disabled if not a terminal)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

log() {
    echo -e "${GREEN}[warpforge]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[warpforge]${NC} $1"
}

error() {
    echo -e "${RED}[warpforge]${NC} $1"
    exit 1
}

# Detect platform
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    case "$os" in
        darwin) os="macos" ;;
        linux) os="linux" ;;
        *)
            error "Unsupported operating system: $os"
            ;;
    esac

    case "$arch" in
        x86_64|amd64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *)
            error "Unsupported architecture: $arch"
            ;;
    esac

    echo "${os}-${arch}"
}

# Get latest release version from GitHub
get_latest_version() {
    local latest
    if command -v curl &>/dev/null; then
        latest=$(curl -sS "$GITHUB_API/releases/latest" 2>/dev/null | grep '"tag_name"' | head -1 | cut -d'"' -f4)
    elif command -v wget &>/dev/null; then
        latest=$(wget -qO- "$GITHUB_API/releases/latest" 2>/dev/null | grep '"tag_name"' | head -1 | cut -d'"' -f4)
    fi

    if [[ -z "$latest" ]]; then
        error "Failed to fetch latest version from GitHub. Check your internet connection."
    fi

    echo "$latest"
}

# Download file
download() {
    local url="$1"
    local dest="$2"

    if command -v curl &>/dev/null; then
        curl -fsSL --progress-bar -o "$dest" "$url"
    elif command -v wget &>/dev/null; then
        wget --show-progress -q -O "$dest" "$url"
    else
        error "Neither curl nor wget found. Please install one of them."
    fi
}

# Main installation
main() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           WarpForge Installer                     ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════╝${NC}"
    echo ""

    # Detect platform
    local platform=$(detect_platform)
    log "Detected platform: $platform"

    # Resolve version
    local version="$WARPFORGE_VERSION"
    if [[ "$version" == "latest" ]]; then
        log "Fetching latest version..."
        version=$(get_latest_version)
    fi

    # Ensure version has 'v' prefix for URL
    local version_tag="$version"
    if [[ ! "$version_tag" =~ ^v ]]; then
        version_tag="v$version"
    fi

    # Version without 'v' for archive name
    local version_num="${version_tag#v}"

    log "Version: $version_tag"
    log "Install directory: $WARPFORGE_HOME"
    echo ""

    # Construct download URL
    local archive_name="warpforge-${version_num}-${platform}.tar.gz"
    local download_url="$GITHUB_RELEASES/download/${version_tag}/${archive_name}"

    # Create temp directory
    local tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" EXIT

    # Download
    log "Downloading $archive_name..."
    local archive_path="$tmp_dir/$archive_name"
    download "$download_url" "$archive_path" || {
        error "Failed to download WarpForge. URL: $download_url"
    }

    # Create installation directory
    log "Installing to $WARPFORGE_HOME..."
    mkdir -p "$WARPFORGE_HOME"

    # Extract (remove old installation if exists)
    if [[ -d "$WARPFORGE_HOME/bin" ]]; then
        warn "Removing previous installation..."
        rm -rf "$WARPFORGE_HOME/bin" "$WARPFORGE_HOME/lib" "$WARPFORGE_HOME/conf" "$WARPFORGE_HOME/backends"
    fi

    tar -xzf "$archive_path" -C "$WARPFORGE_HOME" --strip-components=1

    # Verify installation
    if [[ ! -x "$WARPFORGE_HOME/bin/warpforge" ]]; then
        error "Installation verification failed: warpforge binary not found"
    fi

    # Success message
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           Installation Complete!                  ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "WarpForge $version_tag has been installed to: $WARPFORGE_HOME"
    echo ""
    echo "Installed components:"
    ls -la "$WARPFORGE_HOME/bin/" 2>/dev/null | grep -v "^total" | awk '{print "  " $NF}' | grep -v "^\s*$"
    echo ""

    # Check if already in PATH
    if [[ ":$PATH:" == *":$WARPFORGE_HOME/bin:"* ]]; then
        log "WarpForge is already in your PATH."
    else
        echo -e "${YELLOW}Add WarpForge to your PATH:${NC}"
        echo ""

        local shell_name=$(basename "$SHELL")
        local rc_file=""

        case "$shell_name" in
            bash)
                if [[ -f "$HOME/.bash_profile" ]]; then
                    rc_file=".bash_profile"
                else
                    rc_file=".bashrc"
                fi
                ;;
            zsh)
                rc_file=".zshrc"
                ;;
            fish)
                rc_file=".config/fish/config.fish"
                ;;
            *)
                rc_file=".profile"
                ;;
        esac

        echo "  # Add this line to ~/$rc_file:"
        echo "  export PATH=\"$WARPFORGE_HOME/bin:\$PATH\""
        echo ""
        echo "  # Or run this command to add it automatically:"
        echo "  echo 'export PATH=\"$WARPFORGE_HOME/bin:\$PATH\"' >> ~/$rc_file"
        echo ""
        echo "  # Then reload your shell:"
        echo "  source ~/$rc_file"
        echo ""
    fi

    echo "Verify installation:"
    echo "  warpforge --version"
    echo "  warpforge gpu-info"
    echo ""
}

# Handle --help
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "WarpForge Installer"
    echo ""
    echo "Usage:"
    echo "  curl -sSL https://raw.githubusercontent.com/surfworks/warpforge/main/warpforge-dist/scripts/install.sh | bash"
    echo ""
    echo "Environment Variables:"
    echo "  WARPFORGE_VERSION  Version to install (default: latest)"
    echo "  WARPFORGE_HOME     Installation directory (default: ~/.warpforge)"
    echo ""
    echo "Examples:"
    echo "  # Install latest"
    echo "  curl -sSL .../install.sh | bash"
    echo ""
    echo "  # Install specific version"
    echo "  WARPFORGE_VERSION=v0.1.0 curl -sSL .../install.sh | bash"
    echo ""
    echo "  # Install to custom location"
    echo "  WARPFORGE_HOME=/opt/warpforge curl -sSL .../install.sh | bash"
    exit 0
fi

# Handle --uninstall
if [[ "${1:-}" == "--uninstall" ]]; then
    echo "Uninstalling WarpForge..."
    if [[ -d "$WARPFORGE_HOME" ]]; then
        rm -rf "$WARPFORGE_HOME"
        log "WarpForge has been uninstalled from $WARPFORGE_HOME"
        warn "Don't forget to remove the PATH entry from your shell configuration."
    else
        warn "WarpForge installation not found at $WARPFORGE_HOME"
    fi
    exit 0
fi

main "$@"
