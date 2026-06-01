#!/bin/sh
set -eu

REPO="aep/moxgo"
INSTALL_DIR="/usr/local/bin"
BINARY="moxgo"

status() { echo ">>> $*"; }
warning() { echo "WARNING: $*"; }
error() { echo "ERROR: $*" >&2; exit 1; }

available() { command -v "$1" >/dev/null; }

SUDO=
if [ "$(id -u)" -ne 0 ]; then
    if available sudo; then
        SUDO="sudo"
    elif available doas; then
        SUDO="doas"
    fi
fi

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
        error "could not determine latest release."
    fi

    asset="${BINARY}-${tag}-${os}_${arch}"
    url="https://github.com/${REPO}/releases/download/${tag}/${asset}"

    status "Downloading ${BINARY} ${tag} for ${os}/${arch}..."
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
        status "Installing to ${INSTALL_DIR} (requires sudo)..."
        $SUDO mv "$tmp" "${INSTALL_DIR}/${BINARY}"
    fi

    status "Installed ${BINARY} ${tag} to ${INSTALL_DIR}/${BINARY}"

    if [ "$os" = "linux" ]; then
        configure_systemd
    fi
}

configure_systemd() {
    if ! available systemctl; then
        warning "systemd not found, skipping service setup."
        warning "You can run moxgo manually: moxgo serve"
        return
    fi

    if ! id moxgo >/dev/null 2>&1; then
        status "Creating moxgo user..."
        $SUDO useradd -r -s /bin/false -U -m -d /usr/share/moxgo moxgo
    fi

    if getent group render >/dev/null 2>&1; then
        $SUDO usermod -a -G render moxgo
    fi
    if getent group video >/dev/null 2>&1; then
        $SUDO usermod -a -G video moxgo
    fi

    status "Creating moxgo systemd service..."
    $SUDO tee /etc/systemd/system/moxgo.service >/dev/null <<EOF
[Unit]
Description=Moxgo Inference Server
After=network-online.target

[Service]
ExecStart=${INSTALL_DIR}/${BINARY} serve
User=moxgo
Group=moxgo
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
EOF

    SYSTEMCTL_RUNNING="$(systemctl is-system-running || true)"
    case $SYSTEMCTL_RUNNING in
        running|degraded)
            status "Enabling and starting moxgo service..."
            $SUDO systemctl daemon-reload
            $SUDO systemctl enable moxgo

            start_service() { $SUDO systemctl restart moxgo; }
            trap start_service EXIT
            ;;
        *)
            warning "systemd is not running."
            if is_wsl; then
                warning "See https://learn.microsoft.com/en-us/windows/wsl/systemd#how-to-enable-systemd"
            fi
            ;;
    esac
}

is_wsl() {
    [ -f /proc/sys/fs/binfmt_misc/WSLInterop ] || grep -qi microsoft /proc/version 2>/dev/null
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
            error "unsupported OS: $(uname -s)"
            ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)   echo "amd64" ;;
        aarch64|arm64)   echo "arm64" ;;
        *)
            error "unsupported architecture: $(uname -m)"
            ;;
    esac
}

main
