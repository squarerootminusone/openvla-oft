#!/usr/bin/env bash
#
# Fine-tune OpenVLA-OFT on the Go VLA dataset (4-DOF).
#
# Usage:
#   bash scripts/finetune_go_vla.sh [NUM_GPUS]
#
# Defaults to 1 GPU if not specified.
# Expects to be run from the openvla-oft repo root.

set -euo pipefail

NUM_GPUS="${1:-1}"

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/datasets/rlds"
ZIP_PATH="$REPO_ROOT/datasets/go_vla_dataset.zip"
DATASET_NAME="go_vla_dataset"
FILE_ID="1ZFHYMDSctedImi8yKVIXrLWWlzmJwN-"

# ---------- Wandb ----------
WANDB_ENTITY="m-w-jarosz-team-epoch"
WANDB_PROJECT="openvla-oft"

# ---------- Download ----------
if [ ! -d "$DATA_DIR/$DATASET_NAME" ]; then
    echo "==> Downloading Go VLA dataset from Google Drive..."
    mkdir -p "$DATA_DIR"

    # Use gdown if available, otherwise fall back to curl
    if command -v gdown &>/dev/null; then
        gdown --id "$FILE_ID" -O "$ZIP_PATH"
    else
        echo "    (installing gdown...)"
        pip install -q gdown
        gdown --id "$FILE_ID" -O "$ZIP_PATH"
    fi

    echo "==> Unpacking..."
    unzip -q "$ZIP_PATH" -d "$DATA_DIR"
    rm "$ZIP_PATH"
    echo "==> Dataset ready at $DATA_DIR/$DATASET_NAME"
else
    echo "==> Dataset already exists at $DATA_DIR/$DATASET_NAME, skipping download."
fi

# ---------- Fine-tune ----------
echo "==> Starting fine-tuning with $NUM_GPUS GPU(s)..."

cd "$REPO_ROOT"

torchrun --standalone --nnodes 1 --nproc-per-node "$NUM_GPUS" vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir "$DATA_DIR" \
    --dataset_name "$DATASET_NAME" \
    --run_root_dir "$REPO_ROOT/runs" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 1 \
    --use_proprio False \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 50000 \
    --max_steps 75000 \
    --save_freq 5000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --run_id_note "go_vla_4dof"
