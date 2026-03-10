# Setup Instructions

## Automated Setup (recommended)

```bash
bash scripts/setup_env.sh
conda activate openvla-oft
```

This creates a conda environment with Python 3.10, PyTorch 2.4 + CUDA 12.4, and all dependencies. Requires Ada Lovelace GPU or newer (RTX 6000 Pro, RTX 4090, etc.).

## Manual Setup

```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch 2.4 with CUDA 12.4 (required for Ada Lovelace+)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install openvla-oft (editable, skip deps to avoid TF version conflicts)
pip install -e . --no-deps

# Install training dependencies
pip install "accelerate>=0.25.0" "draccus==0.8.0" einops "huggingface_hub<1.0" \
    json-numpy jsonlines matplotlib "peft==0.11.1" protobuf rich sentencepiece \
    "timm==0.9.10" "tokenizers==0.19.1" wandb "diffusers==0.30.3" imageio \
    uvicorn fastapi "numpy<2"

# Install custom transformers fork (required for bidirectional attn)
pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git" --no-deps

# Install dlimp (skip deps to avoid TF 2.15 pin)
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla" --no-deps

# Install TensorFlow + TFDS (for data loading)
pip install tensorflow tensorflow-datasets tensorflow-graphics

# Install Flash Attention 2 for training
pip install packaging ninja
pip install "flash-attn>=2.5.5" --no-build-isolation
```
