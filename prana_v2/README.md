# PRANA — Policy for Robotic Action via Neural Architecture

Flow matching with correlated noise for low-data robot manipulation on SO-101.

[![HuggingFace Model](https://img.shields.io/badge/🤗_HuggingFace-PRANA__v2-yellow)](https://huggingface.co/Siddarth09/PRANA_v2)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-PRANA-blue)](https://huggingface.co/datasets/Siddarth09/PRANA)
[![LeRobot](https://img.shields.io/badge/Framework-LeRobot_v0.5.1-green)](https://github.com/huggingface/lerobot)

---

## Overview

PRANA is a vision-action policy that picks up a screwdriver and places it in a box using a SO-101 robot arm. It takes dual-camera images (table + wrist) and joint states as input, and predicts 50-step action chunks through iterative denoising.

Three architecture iterations are included:

| Version | Action Head | Vision Backbone | Key Feature |
|---------|-----------|----------------|-------------|
| **v1** | L1 regression | ViT-Tiny (timm) | Self-attention only, deterministic |
| **v2** | Flow matching | ViT-Tiny (timm) | Cross-attention decoder, correlated noise |
| **v3** | Flow matching | DINOv2 ViT-S/14 | Self-supervised vision features |

## Architecture (v2)

<img width="1400" height="1800" alt="Image" src="https://github.com/user-attachments/assets/f5df7e8a-fd31-4774-ae8f-56f47df31309" />

**Training:** Sample noise ε from the empirical action covariance, interpolate with target actions at random timestep t, predict the velocity field. Loss = MSE(v_pred, ε - a).

**Inference:** Start from correlated noise, take 10 Euler denoising steps, output 50 actions. Execute 40, save last 10 for rolling inpainting into the next chunk.

## Setup

### Prerequisites

- Python 3.12+
- CUDA GPU (tested on RTX 5060 Laptop, 8GB VRAM)
- SO-101 robot arm with LeRobot firmware

### Installation

```bash
# Create environment
python -m venv lerobot_env
source lerobot_env/bin/activate

# Install LeRobot
pip install lerobot==0.5.1

# Install dependencies
pip install timm wandb av

# Clone this repo
git clone https://github.com/siddarth09/PRANA.git
cd PRANA
```

### CUDA / PyTorch (RTX 5060 / Blackwell GPUs)

If you have a Blackwell GPU (RTX 50xx), you need the cu128 nightly build:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

## Project Structure

```
PRANA/
├── prana_v1/                    # Deterministic action chunking (baseline)
│   └── model/
│       ├── configuration_prana.py
│       ├── encoders.py
│       ├── modeling.py
│       └── policy_prana.py
├── prana_v2/                    # Flow matching + correlated noise
│   ├── model/
│   │   ├── configuration_prana.py
│   │   ├── encoders.py
│   │   ├── modeling.py
│   │   └── policy_prana.py
│   ├── train_v2.py
│   ├── deploy_v2.py
│   └── fit_noise.py
├── prana_v3/                    # DINOv2 backbone variant
│   └── model/
│       ├── configuration_prana.py
│       ├── encoder.py
│       ├── modeling.py
│       └── policy_prana.py
└── outputs/
    └── train/
```

## Data Collection

Record teleoperation episodes using LeRobot:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{"table": {"type": "intelrealsense", "serial_number_or_name": "YOUR_SERIAL", "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --dataset.repo_id=YOUR_HF_USER/YOUR_DATASET \
    --dataset.single_task="Pick the screwdriver and place it in the box" \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=25 \
    --dataset.fps=30 \
    --display_data=true \
    --dataset.push_to_hub=false
```

We collected 123 episodes total.

## Training

### PRANA v2 (recommended)

```bash
python3 prana_v2/train_v2.py \
    --dataset.repo_id=Siddarth09/PRANA \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=false \
    --policy.type=prana_v2 \
    --policy.device=cuda \
    --policy.camera_order='["observation.images.table","observation.images.wrist"]' \
    --batch_size=8 \
    --num_workers=0 \
    --steps=100000 \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/prana_v2 \
    --wandb.enable=true
```

Training takes approximately 6-8 hours on an RTX 5060 (8GB VRAM). Loss should converge from ~3.5 to ~0.5.

### Fit Correlated Noise (required after training)

```bash
python3 prana_v2/fit_noise.py \
    --dataset Siddarth09/PRANA \
    --checkpoint outputs/train/prana_v2/checkpoints/last/pretrained_model
```

> **⚠️ This step is critical.** Without it, the robot will be jittery during deployment. The script computes the action covariance matrix from the dataset and patches the Cholesky factor into the checkpoint.

### Backbone Variants

Change `vision_backbone` in `configuration_prana.py`:

```python
# ViT-Tiny (default, 5.7M params, 197 tokens/cam)
vision_backbone: str = "vit_tiny_patch16_224"

# DINOv2 ViT-S/14 (22M params, 256 tokens/cam, self-supervised)
vision_backbone: str = "vit_small_patch14_dinov2"

# ConvNeXt-Tiny (28M params, 49 tokens/cam, CNN)
vision_backbone: str = "convnext_tiny"
```

The encoder auto-detects the backbone output format (ViT tokens vs CNN spatial maps).

## Deployment

### Deploy on SO-101

```bash
python3 prana_v2/deploy_v2.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{"table": {"type": "intelrealsense", "serial_number_or_name": "YOUR_SERIAL", "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}}' \
    --display_data=false \
    --dataset.fps=10 \
    --dataset.repo_id=YOUR_HF_USER/eval_prana_v2 \
    --dataset.num_episodes=5 \
    --dataset.single_task="Pick the screwdriver and place it in the box" \
    --dataset.push_to_hub=false \
    --policy.path=outputs/train/prana_v2/checkpoints/last/pretrained_model \
    --policy.device=cuda
```

### Deployment Tips

| Tip | Why |
|-----|-----|
| `--dataset.fps=10` | Matches actual camera throughput, prevents frame drops |
| `--display_data=false` | Saves CPU for inference |
| Denoise steps = 5 | Faster inference without retraining (edit config) |
| Fit noise sampler | **Critical** — reduces jitter significantly |

### Inference Speed

| Operation | Time |
|-----------|------|
| Chunk prediction (10 denoise steps) | ~22ms |
| Queued action (from buffer) | ~0.5ms |
| Camera capture (2 cameras) | ~100ms |

## How Flow Matching Works

Standard imitation learning predicts a single action via regression (L1/MSE loss). This learns the **average** of demonstrations, which fails when multiple valid strategies exist.

Flow matching learns the **velocity field** that transforms noise into valid action trajectories:

1. **Training:** Interpolate between noise ε and target actions a at time t: `x_t = t·ε + (1-t)·a`. Predict velocity v_t. Loss = `MSE(v_t, ε - a)`.
2. **Inference:** Start from noise, take 10 Euler steps: `x_{t-dt} = x_t - dt·v_t`. Result ≈ valid action trajectory.

**Correlated noise** makes this easier — instead of starting from random noise, we start from noise shaped like real trajectories (temporal smoothness + joint coordination), computed from the Cholesky decomposition of the empirical action covariance.

## Results

| Metric | v1 (L1 regression) | v2 (flow matching) |
|--------|--------------------|--------------------|
| Training loss | 0.334 (L1) | 0.50 (velocity MSE) |
| Reaches screwdriver | Yes | Yes |
| Grasps screwdriver | With assistance | Yes |
| Places in box | No | With assistance |
| Inference speed | ~5ms | ~22ms |

## Model Weights

- **v2 (ViT-Tiny):** [Siddarth09/PRANA_v2](https://huggingface.co/Siddarth09/PRANA_v2)
- **Dataset:** [Siddarth09/PRANA](https://huggingface.co/datasets/Siddarth09/PRANA)

## Citation

```bibtex
@misc{dayasagar2026prana,
  title={PRANA: Flow Matching with Correlated Noise for Low-Data Robot Manipulation},
  author={Dayasagar, Siddarth},
  year={2026},
  url={https://github.com/siddarth09/PRANA}
}
```

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face
- Flow matching inspired by [BEHAVIOR 2025 1st place solution](https://arxiv.org/abs/2512.06951)
- Built at Northeastern University

## License

MIT