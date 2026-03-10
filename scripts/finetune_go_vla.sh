#!/usr/bin/env bash
#
# Fine-tune OpenVLA-OFT on the Go VLA dataset (4-DOF).
#
# Usage:
#   bash scripts/finetune_go_vla.sh <DATA_DIR> [NUM_GPUS]
#
# Arguments:
#   DATA_DIR   Path to directory containing the go_vla_dataset RLDS dataset
#   NUM_GPUS   Number of GPUs (default: 1)
#
# Expects:
#   - conda env "openvla-oft" (see scripts/setup_env.sh)
#   - Run from the openvla-oft repo root

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/finetune_go_vla.sh <DATA_DIR> [NUM_GPUS]"
    echo "  DATA_DIR: path to directory containing go_vla_dataset/"
    exit 1
fi

DATA_DIR="$1"
NUM_GPUS="${2:-1}"

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_NAME="go_vla_dataset"

# ---------- Wandb ----------
WANDB_ENTITY="m-w-jarosz-team-epoch"
WANDB_PROJECT="openvla-oft"

# ---------- Validate ----------
if [ ! -d "$DATA_DIR/$DATASET_NAME" ]; then
    echo "Error: $DATA_DIR/$DATASET_NAME not found."
    echo "Make sure the RLDS dataset is built and located at that path."
    exit 1
fi

# ---------- Fine-tune ----------
echo "==> Starting fine-tuning with $NUM_GPUS GPU(s)..."
echo "    Dataset: $DATA_DIR/$DATASET_NAME"

cd "$REPO_ROOT"

# Use torchrun for multi-GPU, env vars for single-GPU (avoids torchrun segfault on some setups)
if [ "$NUM_GPUS" -gt 1 ]; then
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
else
    MASTER_ADDR=localhost MASTER_PORT=29500 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    python vla-scripts/finetune.py \
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
fi
