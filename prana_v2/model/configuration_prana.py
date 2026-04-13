"""
PRANA v2 Configuration
======================
Flow-matching action prediction with correlated noise.
Designed for single-task manipulation on UR5e with LeRobot (80 episodes).

Key design choices from BEHAVIOR 2025 1st-place adapted for low-data:
  - Flow matching instead of CVAE (better multimodal action modeling)
  - Correlated noise from empirical action covariance
  - Rolling inpainting for temporal consistency
  - Per-timestamp normalization on delta actions
  - Frozen ViT backbone (critical for 80 episodes)
"""

from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("prana_v2")
@dataclass
class PranaConfig(PreTrainedConfig):
    # ── Observation / Action chunking ──────────────────────────────────
    n_obs_steps: int = 1
    chunk_size: int = 50          # predict 50 actions per chunk
    n_action_steps: int = 40      # execute 40, save last 10 for inpainting
    inpaint_steps: int = 10       # overlap window for temporal consistency

    # ── Model dimensions ───────────────────────────────────────────────
    hidden_dim: int = 256
    n_heads: int = 8
    dim_feedforward: int = 768
    n_encoder_layers: int = 4     # context encoder (vision + state)
    n_decoder_layers: int = 8     # action decoder (cross-attention)
    dropout: float = 0.05

    # ── Flow matching ──────────────────────────────────────────────────
    num_denoise_steps: int = 10   # denoising steps at inference
    noise_beta: float = 0.5       
    # Beta distribution for biased time sampling (harder early steps)
    time_sampling_alpha: float = 1.5
    time_sampling_beta: float = 1.0

    # ── Delta actions + per-timestamp normalization ────────────────────
    use_delta_actions: bool = True
    use_per_timestamp_norm: bool = True

    # ── Vision backbone ────────────────────────────────────────────────
    vision_backbone: str = "vit_tiny_patch16_224"
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0

    # ── Camera setup ───────────────────────────────────────────────────
    camera_order: list[str] = field(
        default_factory=lambda: [
            "observation.images.table",
            "observation.images.wrist",
        ]
    )

    # ── Normalization ──────────────────────────────────────────────────
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size.")
        if self.inpaint_steps > self.chunk_size - self.n_action_steps:
            raise ValueError(
                f"inpaint_steps ({self.inpaint_steps}) too large. "
                f"Max is chunk_size - n_action_steps = {self.chunk_size - self.n_action_steps}"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None  # cosine scheduler applied in training loop

    def validate_features(self) -> None:
        pass

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None