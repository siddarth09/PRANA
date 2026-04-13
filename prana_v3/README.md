# PRANA v3 — Flow Matching with DINOv2 Vision

Flow matching action prediction with a **DINOv2 ViT-S/14** self-supervised vision backbone for richer spatial features.

## Architecture

<p align="center">
  <img src="prana_v3_architecture.png" alt="PRANA v3 Architecture" width="700"/>
</p>

PRANA v3 is identical to v2 in every component except the vision backbone:

| Component | v2 | v3 |
|---|---|---|
| **Vision backbone** | ViT-Tiny (timm, ImageNet supervised) | **DINOv2 ViT-S/14 (self-supervised)** |
| Embed dimension | 192 | **384 (2x)** |
| Tokens per camera | 197 (patch16) | **256 (patch14)** |
| Total visual tokens | 394 | **512** |
| Backbone params | 5.7M (frozen) | **22M (frozen)** |
| Pretraining | ImageNet classification | **Self-supervised (142M images)** |
| Context encoder | 4 self-attention layers | 4 self-attention layers |
| Action decoder | 7 cross-attention layers | 7 cross-attention layers |
| Flow matching | Correlated noise, 10 Euler steps | Correlated noise, 10 Euler steps |
| Rolling inpainting | Execute 40, save 10 | Execute 40, save 10 |

## Why DINOv2?

DINOv2's self-supervised training learns **spatial structure** — where objects are and what shape they have — rather than just classification signals. For a robot looking at a screwdriver on a table:

- **ImageNet ViT-Tiny** encodes: "this image contains a screwdriver" (global class signal)
- **DINOv2 ViT-S/14** encodes: "there is a screwdriver at position (x,y) with orientation θ" (local spatial features)

Each camera produces 256 tokens at 384 dimensions (vs 197 tokens at 192d for v2). That's **2.6x more visual information** flowing into the cross-attention decoder.

## Project Structure

```
prana_v3/
├── model/
│   ├── configuration_prana.py   # Config (registered as "prana_v3")
│   ├── encoder.py               # DINOv2 ViT-S/14 + camera ID + state + time + action encoders
│   ├── modeling.py              # Flow matching model (same as v2)
│   └── policy_prana.py          # LeRobot policy wrapper
├── train_v3.py                  # Training script
└── deploy_v3.py                 # Deployment on physical robot
```

## Setup

```bash
# Same as v2, plus DINOv2 loads via torch.hub on first run
pip install lerobot==0.5.1 timm wandb av
```

> **Note:** The DINOv2 backbone is loaded via `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` which requires internet on first run. Weights are cached locally after that.

## Training

```bash
python3 prana_v3/train_v3.py \
    --dataset.repo_id=Siddarth09/PRANA \
    --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=false \
    --policy.type=prana_v3 \
    --policy.device=cuda \
    --policy.camera_order='["observation.images.table","observation.images.wrist"]' \
    --batch_size=8 \
    --num_workers=0 \
    --steps=100000 \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/prana_v3 \
    --wandb.enable=true
```

### Fit Correlated Noise (required after training)

```bash
python3 prana_v3/fit_noise.py \
    --dataset Siddarth09/PRANA \
    --checkpoint outputs/train/prana_v3/checkpoints/last/pretrained_model
```

> **⚠️ Critical step.** Without fitting the noise sampler, the robot will be jittery during deployment.

## Deployment

```bash
python3 prana_v3/deploy_v3.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{"table": {"type": "intelrealsense", "serial_number_or_name": "YOUR_SERIAL", "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}}' \
    --display_data=false \
    --dataset.fps=10 \
    --dataset.repo_id=YOUR_HF_USER/eval_prana_v3 \
    --dataset.num_episodes=5 \
    --dataset.single_task="Pick the screwdriver and place it in the box" \
    --dataset.push_to_hub=false \
    --policy.path=outputs/train/prana_v3/checkpoints/last/pretrained_model \
    --policy.device=cuda
```

### VRAM Considerations

DINOv2 ViT-S/14 uses ~1.5GB more VRAM than ViT-Tiny due to larger embeddings (384d vs 192d) and more tokens (512 vs 394). On an RTX 5060 (8GB):

| Batch size | Estimated VRAM |
|---|---|
| 8 | ~5.5 GB |
| 4 | ~3.5 GB |
| 16 | ~8 GB (tight) |

## Key Differences from v2

1. **`encoder.py`** — Uses `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` instead of `timm.create_model()`. Extracts patch tokens via `forward_features()["x_norm_patchtokens"]` (excludes CLS token).
2. **`configuration_prana.py`** — Registered as `"prana_v3"`, removes `vision_backbone` string field since the backbone is hardcoded to DINOv2.
3. **Everything else** — Identical to v2 (flow matching, correlated noise, rolling inpainting, action decoder).

## Model Details

| Parameter | Value |
|---|---|
| Vision backbone | DINOv2 ViT-S/14 (frozen, 22M params) |
| Hidden dim | 256 |
| Attention heads | 8 |
| FFN dim | 512 |
| Encoder layers | 4 |
| Decoder layers | 7 |
| Chunk size | 50 |
| Action dim | 6 |
| Denoise steps | 10 |
| Trainable params | ~30M |
| Total params | ~52M |

## Citation

```bibtex
@misc{dayasagar2026prana,
  title={PRANA: Flow Matching with Correlated Noise for Low-Data Robot Manipulation},
  author={Dayasagar, Siddarth},
  year={2026},
  url={https://github.com/siddarth09/PRANA}
}
```