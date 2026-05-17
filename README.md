# In-Context Learning for Autonomous Drone Racing

A transformer-based policy for few-shot racing-line adaptation. The model conditions on demonstration trajectories at inference time and predicts drone control actions for a target episode - no fine-tuning required.

<p align="center">
  <video src="assets/test_lemniscate_ep_000013_custom.mp4" controls width="720">
    Demo video: assets/test_lemniscate_ep_000013_custom.mp4
  </video>
</p>

<p align="center">
  <a href="assets/test_lemniscate_ep_000013_custom.mp4">Open the lemniscate demo video</a>
</p>

## Overview

Traditional imitation learning requires retraining for each new track. This project applies **In-Context Learning (ICL)** to drone racing: given a few demonstration laps, the model predicts expert-quality actions on the same track without any parameter updates.

**Key idea**: A transformer policy processes demonstration trajectories as context and cross-attends from the current query state to produce thrust and body-rate commands.

### Architecture

```
                    ┌─────────────────────────────────┐
                    │     ICL Transformer Policy       │
                    │                                  │
  Demo Laps ──────►│  ┌──────────────┐                │
  (state, action)   │  │  Trajectory  │   Context      │
  sequences         │  │  Encoder     │──Embeddings──┐ │
                    │  │  (4-layer    │               │ │
                    │  │  Transformer)│               │ │
                    │  └──────────────┘               │ │
                    │                            ┌────▼─┤
                    │                            │Cross-│
  Current State ──►│  ┌──────────────┐           │Attn  │──► Action
  (19-dim)          │  │ Observation  │  Query    │Decoder│   (thrust,
                    │  │ Encoder      │──Embed──►│(6-lyr)│    body rates)
                    │  │ (3-layer MLP)│           └──────┤
                    │  └──────────────┘                  │
                    └────────────────────────────────────┘
```

### State and Action Spaces

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Position | 3 | x, y, z coordinates |
| Orientation | 4 | Quaternion (qx, qy, qz, qw) |
| Linear velocity | 3 | vx, vy, vz |
| Angular velocity | 3 | wx, wy, wz |
| IMU acceleration | 3 | ax, ay, az |
| IMU angular velocity | 3 | wx, wy, wz |
| **Total state** | **19** | |
| **Action** | **4** | Thrust + body rates (roll, pitch, yaw) |

## Results

Trained on 28 episodes across 3 track types with rigorous episode-level train/val splitting (zero data leakage).

| Track Type | MSE | Std Dev | Val Episodes |
|------------|-----|---------|-------------|
| trackRATM | 0.121 | 0.026 | 3 |
| lemniscate | 0.115 | 0.022 | 3 |
| ellipse | 0.119 | 0.025 | 2 |
| **Overall** | **0.118** | **0.024** | **8** |

| Metric | Value |
|--------|-------|
| Model parameters | 9.96M |
| Inference (200 steps) | ~15ms on H100 |
| Effective control rate | ~67 Hz |
| Training time | ~6.3 hours (100 epochs, H100) |

## Quickstart

### Requirements

- Python 3.10+
- PyTorch >= 2.0
- CUDA-capable GPU (recommended)

### Install

```bash
pip install -r requirements.txt
```

The training and evaluation scripts expect HDF5 datasets under `data/` and model checkpoints under `checkpoints/`. Those large artifacts are intentionally ignored by Git. Demo video generation uses the original full-lap dataset, typically `data/ratm_racing_dataset.h5`.

### Train

```bash
# Prepare data in data/train_final.h5 and data/val_final.h5
bash train.sh
```

Training uses AdamW with ReduceLROnPlateau scheduling:
- Batch size: 256
- Sequence length: 512
- 3 demonstration trajectories per sample
- Embedding dimension: 256

### Run Inference / Demo

```bash
# Create a demo video from a checkpoint and validation dataset
python scripts/create_demo_video.py \
  --checkpoint checkpoints/best_model.pt \
  --data data/ratm_racing_dataset.h5 \
  --track lemniscate \
  --num-demos 3 \
  --output-dir results/demo_videos \
  --device cpu

# Evaluation mode
python scripts/evaluate_icl.py \
  --checkpoint checkpoints/best_model.pt \
  --data data/val_final.h5 \
  --output-dir results \
  --num-demos 1 3 \
  --device cpu
```

### Docker

```bash
docker build -t drone-icl .
docker run --gpus all -it drone-icl bash train.sh
```

## Project Structure

```
├── drone_icl/                 # Core model code
│   ├── models.py              # ICLTransformerPolicy
│   ├── encoders.py            # Observation, Trajectory, MultiModal encoders
│   ├── dataset_icl.py         # HDF5 dataset with sliding windows
│   └── train.py               # Training loop with checkpointing
├── scripts/                   # Evaluation and utility scripts
│   ├── evaluate_icl.py        # Model evaluation
│   ├── create_demo_video.py   # Generate trajectory visualizations
│   ├── isaac_sim_inference.py # NVIDIA Isaac Sim integration
│   ├── convert_ratm_to_hdf5.py
│   ├── augment_dataset_fast.py
│   ├── expand_dataset_sliding_window.py
│   └── ...                    # See scripts/ for full list
├── configs/                   # Track configurations
├── slurm/                     # HPC/SLURM job scripts
├── docs/                      # Documentation
│   ├── IEEE_REPORT.md         # Full IEEE-format paper
│   ├── ARCHITECTURE_ANALYSIS.md
│   ├── EVALUATION_GUIDE.md
│   ├── DATA_PIPELINE.md
│   └── ISAAC_SIM_INTEGRATION.md
└── assets/                    # README demo videos
    └── test_lemniscate_ep_000013_custom.mp4
```

## Data Pipeline

1. **Raw data**: Expert trajectories from RATM dataset (trackRATM, lemniscate, ellipse)
2. **Convert**: `scripts/convert_ratm_to_hdf5.py` — RATM format to HDF5
3. **Augment**: `scripts/augment_dataset_fast.py` — Rotation/translation augmentation
4. **Expand**: `scripts/expand_dataset_sliding_window.py` — Sliding window extraction
5. **Split**: `scripts/split_base_episodes.py` — Episode-level train/val split
6. **Verify**: `scripts/verify_no_leakage.py` — Confirm zero overlap

## Media Assets

The README demo video is stored at `assets/test_lemniscate_ep_000013_custom.mp4`. It was copied from `zaratan_results/results/demo_videos/demo_videos/test_lemniscate_ep_000013_custom.mp4` so the README does not depend on ignored generated-results directories.

## License

This project is for academic and research purposes.
