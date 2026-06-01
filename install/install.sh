#!/bin/sh
set -eu

REPO="aep/moxgo"
INSTALL_DIR="/usr/local/bin"
BINARY="moxgo"

main() {
    os="$(detect_os)"
    arch="$(detect_arch)"

    if [ "$os" = "android" ]; then
        echo "On Android, install the APK from the latest release:"
        echo "  https://github.com/${REPO}/releases/latest"
        echo ""
        echo "Or download directly:"
        echo "  curl -fsSL \$(curl -fsSL https://api.github.com/repos/${REPO}/releases/latest | grep browser_download_url | grep android | cut -d'\"' -f4) -o moxgo-android"
        exit 0
    fi

    if [ "$os" = "windows" ]; then
        printf "\n  This script does not run natively on Windows.\n"
        printf "  Use one of the following instead:\n\n"
        printf "  PowerShell:\n"
        printf "    irm https://moxgo.ai/install.ps1 | iex\n\n"
        printf "  Or WSL / Git Bash:\n"
        printf "    curl -fsSL https://moxgo.ai/install.sh | sh\n\n"
        exit 1
    fi

    tag="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | cut -d'"' -f4)"
    if [ -z "$tag" ]; then
        echo "Error: could not determine latest release." >&2
        exit 1
    fi

    asset="${BINARY}-${tag}-${os}_${arch}"
    url="https://github.com/${REPO}/releases/download/${tag}/${asset}"

    echo "Downloading ${BINARY} ${tag} for ${os}/${arch}..."
    tmp="$(mktemp)"
    if ! curl -fsSL -o "$tmp" "$url"; then
        echo "Error: failed to download ${url}" >&2
        echo "Check available assets at https://github.com/${REPO}/releases/latest" >&2
        rm -f "$tmp"
        exit 1
    fi

    chmod +x "$tmp"

    if [ -w "$INSTALL_DIR" ]; then
        mv "$tmp" "${INSTALL_DIR}/${BINARY}"
    else
        echo "Installing to ${INSTALL_DIR} (requires sudo)..."
        sudo mv "$tmp" "${INSTALL_DIR}/${BINARY}"
    fi

    echo "Installed ${BINARY} ${tag} to ${INSTALL_DIR}/${BINARY}"
}

detect_os() {
    case "$(uname -s)" in
        Linux*)
            if [ -f /system/build.prop ] || [ -d /system/app ]; then
                echo "android"
            else
                echo "linux"
            fi
            ;;
        Darwin*)  echo "darwin" ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *)
            echo "Error: unsupported OS: $(uname -s)" >&2
            exit 1
            ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)   echo "amd64" ;;
        aarch64|arm64)   echo "arm64" ;;
        *)
            echo "Error: unsupported architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac
}

main
