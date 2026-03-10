# Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success

**Project website: https://openvla-oft.github.io/**

**Paper: https://arxiv.org/abs/2502.19645**

**Summary video: https://youtu.be/T3Zkkr_NTSA**

## System Requirements

* NVIDIA GPU with Ada Lovelace architecture or newer (RTX 6000 Pro, RTX 4090, etc.)
* 1 GPU with 48+ GB VRAM for training (RTX 6000 Pro recommended)
* conda

## Installation

```bash
bash scripts/setup_env.sh
conda activate openvla-oft
```

This creates a conda environment with Python 3.10, PyTorch 2.4 + CUDA 12.4, and all dependencies.

## Training and Evaluation

### Go VLA (4-DOF stone placement)

Fine-tune OpenVLA-OFT on the Go VLA benchmark dataset (4-DOF: dx, dy, dz, gripper).

**1. Authenticate with Weights & Biases:**

```bash
wandb login
```

Get your API key from https://wandb.ai/authorize, or set:

```bash
export WANDB_API_KEY="your-api-key-here"
```

**2. Build the RLDS dataset** (if not already built):

```bash
cd /path/to/rlds_builder/go_vla_dataset
pip install apache-beam mlcroissant
tfds build --data_dir /path/to/data_dir
```

This creates a `go_vla_dataset/` directory inside `data_dir`.

**3. Run fine-tuning:**

```bash
cd openvla-oft

# Single GPU — DATA_DIR must contain a go_vla_dataset/ subdirectory (built RLDS)
bash scripts/finetune_go_vla.sh <DATA_DIR>

# Multi-GPU (e.g., 4 GPUs)
bash scripts/finetune_go_vla.sh <DATA_DIR> 4
```

Logs are sent to the `m-w-jarosz-team-epoch/openvla-oft` wandb project.

To disable wandb (e.g., for local debugging):

```bash
WANDB_MODE=disabled bash scripts/finetune_go_vla.sh <DATA_DIR>
```

Checkpoints are saved to `runs/` every 5,000 steps. After training, merge LoRA weights:

```bash
python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint openvla/openvla-7b \
  --lora_finetuned_checkpoint_dir runs/<your_run>/
```

### LIBERO / ALOHA

See [LIBERO.md](LIBERO.md) for fine-tuning/evaluating on LIBERO simulation benchmark task suites.

See [ALOHA.md](ALOHA.md) for fine-tuning/evaluating on real-world ALOHA robot tasks.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2502.19645):

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
