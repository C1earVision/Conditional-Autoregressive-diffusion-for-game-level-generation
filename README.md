# Conditional Autoregressive Diffusion for Discrete Level Generation

A deep learning framework for generating Super Mario Bros levels using conditional diffusion models with autoregressive patch generation.

## Overview

This project implements a two-stage approach for controllable level generation:

1. **Autoencoder**: Learns a compressed latent representation of level patches (14×16 tile grids)
2. **Conditional Diffusion Model**: Generates new level patches in latent space, conditioned on:
   - Previous patches (autoregressive context)
   - Target difficulty score (0.0 - 1.0)

The system uses **Classifier-Free Guidance (CFG)** to enable control over the difficulty of generated levels.

## Project Structure

```
├── config/                  # Configuration files
│   ├── model_config.py      # Model architecture settings
│   ├── training_config.py   # Training hyperparameters
│   └── generation_config.yaml  # Generation settings
├── data/                    # Data processing modules
│   ├── parser.py            # Level parsing utilities
│   ├── dataset.py           # PyTorch dataset classes
│   └── patch_extraction.py  # Patch extraction from levels
├── models/                  # Neural network architectures
│   ├── autoencoder.py       # Patch encoder/decoder
│   ├── diffusion.py         # Diffusion U-Net with CFG
│   ├── embeddings.py        # Time/playability embeddings
│   ├── noise_scheduler.py   # Diffusion noise schedule
│   └── latent_normalizer.py # Latent space normalization
├── training/                # Training modules
│   ├── autoencoder_trainer.py
│   └── diffusion_trainer.py
├── generation/              # Level generation
│   ├── sampler.py           # Diffusion sampling with CFG
│   └── stitcher.py          # Patch-to-level assembly
├── evaluation/              # Evaluation metrics
│   └── patch_evaluation.py  # Difficulty scoring
├── scripts/                 # Runnable scripts
│   ├── prepare_data.py      # Data preprocessing
│   ├── train_autoencoder.py # Autoencoder training
│   ├── prepare_latents.py   # Encode patches to latents
│   ├── train_diffusion.py   # Diffusion model training
│   └── generate_levels.py   # Level generation
├── checkpoints/             # Saved model weights
├── output/                  # Generated outputs
└── dataset/                 # Raw level data
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Conditional Autoregressive diffusion for discreet level generation"

# Create virtual environment (recommended)
conda create -n diffusion python=3.10
conda activate diffusion

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if using GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Step 1: Prepare Data

```bash
python -m scripts.prepare_data
```

This extracts patches and computes difficulty scores, saving to `output/processed/`.

### Step 2: Train Autoencoder

```bash
python -m scripts.train_autoencoder
```

Trains the patch autoencoder. Checkpoint saved to `checkpoints/autoencoder.pth`.

### Step 3: Prepare Latents

```bash
python -m scripts.prepare_latents
```

Encodes all patches to latent space and fits the latent normalizer.

### Step 4: Train Diffusion Model

```bash
python -m scripts.train_diffusion
```

Trains the conditional diffusion model. Best checkpoint saved to `checkpoints/diffusion_best.pth`.

### Step 5: Generate Levels

Edit `config/generation_config.yaml` to set generation parameters:

```yaml
generation:
  num_levels: 5           # Number of levels to generate
  patches_per_level: 20   # Patches per level (affects length)
  difficulty_target: 0.7  # Difficulty target (0.0=easy, 1.0=hard)
  temperature: 0.5        # Sampling temperature
  guidance_scale: 3.0     # CFG strength (higher = stronger conditioning)
```

Then generate:

```bash
python -m scripts.generate_levels
```

Generated levels are saved to `output/generated_levels/`.

## License

MIT License
