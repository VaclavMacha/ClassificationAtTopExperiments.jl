#!/usr/bin/env bash
# One-time setup: downloads the JIN pretrained weights, clones the reference
# SRNet implementation, and installs Python dependencies via Poetry.
#
# Run from the scripts_python/ directory:
#   bash prepare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
PRETRAINED_DIR="$DATA_DIR/pretrained"
VENDOR_DIR="$SCRIPT_DIR/vendor/srnet"
SRNET_REPO="https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet"
WEIGHTS_URL="https://dde.binghamton.edu/download/feature_extractors/download/JIN_SRNet.zip"
WEIGHTS_ZIP="$DATA_DIR/JIN_SRNet.zip"
WEIGHTS_PT="$DATA_DIR/JIN_SRNet.pt"

# ---------------------------------------------------------------------------
# 1. Download JIN pretrained weights
# ---------------------------------------------------------------------------
if [ -f "$WEIGHTS_PT" ]; then
    echo "[1/3] JIN_SRNet.pt already present — skipping download."
else
    echo "[1/3] Downloading JIN pretrained SRNet weights..."
    mkdir -p "$DATA_DIR"
    curl -L --progress-bar "$WEIGHTS_URL" -o "$WEIGHTS_ZIP"
    echo "      Unzipping..."
    unzip -q "$WEIGHTS_ZIP" -d "$DATA_DIR"
    mv "$DATA_DIR/epoch=56_val_wAUC=0.8921.pt" "$WEIGHTS_PT"
    rm "$WEIGHTS_ZIP"
    if [ ! -f "$WEIGHTS_PT" ]; then
        # zip may place the .pt inside a subdirectory — find and move it
        found="$(find "$DATA_DIR" -name "JIN_SRNet.pt" | head -1)"
        if [ -z "$found" ]; then
            echo "ERROR: JIN_SRNet.pt not found after unzip."
            echo "       Unzip the archive manually into $DATA_DIR and re-run."
            exit 1
        fi
        mv "$found" "$WEIGHTS_PT"
    fi
    echo "      Saved → $WEIGHTS_PT"
fi

# ---------------------------------------------------------------------------
# 2. Clone reference SRNet implementation
# ---------------------------------------------------------------------------
if [ -d "$VENDOR_DIR/.git" ]; then
    echo "[2/3] vendor/srnet already cloned — pulling latest..."
    git -C "$VENDOR_DIR" pull --ff-only
else
    echo "[2/3] Cloning $SRNET_REPO → vendor/srnet ..."
    mkdir -p "$SCRIPT_DIR/vendor"
    git clone "$SRNET_REPO" "$VENDOR_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Install Python dependencies via uv
# ---------------------------------------------------------------------------
echo "[3/3] Installing Python dependencies via UV..."
cd "$SCRIPT_DIR"
uv sync

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Export weights and generate parity reference:"
echo "       cd scripts_python"
echo "       uv run python prepare_srnet_weights.py \\"
echo "           --input  $WEIGHTS_PT \\"
echo "           --output $PRETRAINED_DIR/srnet.h5"
echo "  2. Verify Julia implementation:"
echo "       julia scripts/test_srnet.jl"
