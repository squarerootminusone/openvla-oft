#!/usr/bin/env bash
#
# Set up conda environment for OpenVLA-OFT fine-tuning.
#
# Requires: conda, NVIDIA GPU (Ada Lovelace / RTX 6000 Pro or newer)
#
# Usage:
#   bash scripts/setup_env.sh
#
# Creates a conda env named "openvla-oft" with all dependencies.

set -euo pipefail

ENV_NAME="openvla-oft"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Creating conda environment '$ENV_NAME' with Python 3.10..."
conda create -n "$ENV_NAME" python=3.10 -y

echo "==> Installing PyTorch 2.4.0 + CUDA 12.4 (Ada Lovelace+ support)..."
conda run -n "$ENV_NAME" --no-banner pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing openvla-oft (editable, no deps — we install them manually)..."
conda run -n "$ENV_NAME" --no-banner pip install -e "$REPO_ROOT" --no-deps

echo "==> Installing core training dependencies..."
conda run -n "$ENV_NAME" --no-banner pip install \
    "accelerate>=0.25.0" \
    "draccus==0.8.0" \
    "einops" \
    "huggingface_hub<1.0" \
    "json-numpy" \
    "jsonlines" \
    "matplotlib" \
    "peft==0.11.1" \
    "protobuf" \
    "rich" \
    "sentencepiece" \
    "timm==0.9.10" \
    "tokenizers==0.19.1" \
    "wandb" \
    "diffusers==0.30.3" \
    "imageio" \
    "uvicorn" \
    "fastapi" \
    "numpy<2"

echo "==> Installing custom transformers fork (required for bidirectional attn)..."
conda run -n "$ENV_NAME" --no-banner pip install \
    "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git" --no-deps

echo "==> Installing dlimp..."
conda run -n "$ENV_NAME" --no-banner pip install \
    "dlimp @ git+https://github.com/moojink/dlimp_openvla" --no-deps

echo "==> Installing TensorFlow + TFDS (for data loading)..."
conda run -n "$ENV_NAME" --no-banner pip install \
    tensorflow tensorflow-datasets tensorflow-graphics

echo "==> Installing flash-attn (takes a few minutes)..."
conda run -n "$ENV_NAME" --no-banner pip install packaging ninja
conda run -n "$ENV_NAME" --no-banner pip install "flash-attn>=2.5.5" --no-build-isolation \
    || echo "WARNING: flash-attn failed to build. Training will still work but slower."

echo ""
echo "==> Done! Activate with:  conda activate $ENV_NAME"
echo "    Then run:  bash scripts/finetune_go_vla.sh <DATA_DIR>"
