"""
PRANA v3 Policy
===============
LeRobot-compatible policy with:
  - DINOv2 ViT-S/14 frozen vision encoder
  - Flow matching training (velocity prediction + MSE loss)
  - Correlated noise initialization from dataset statistics
  - Rolling inpainting for temporal consistency at inference
  - Differential learning rates for frozen backbone
  - Proper action denormalization
"""

import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from collections import deque
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from prana_v3.model.modeling import PranaVLA
from prana_v3.model.configuration_prana import PranaConfig


class PranaPolicy(PreTrainedPolicy):
    config_class = PranaConfig
    name = "prana_v3"

    def __init__(self, config: PranaConfig, dataset_stats: dict | None = None, **kwargs):
        super().__init__(config)
        self.config = config
        self.dataset_stats = dataset_stats

        # ── Resolve dimensions from dataset features ───────────────────
        state_dim = 6
        action_dim = 6

        features = getattr(config, "input_features", None) or getattr(config, "features", {})
        if OBS_STATE in features:
            f = features[OBS_STATE]
            state_dim = f["shape"][0] if isinstance(f, dict) else f.shape[0]

        out_features = getattr(config, "output_features", None) or features
        if ACTION in out_features:
            f = out_features[ACTION]
            action_dim = f["shape"][0] if isinstance(f, dict) else f.shape[0]

        self.action_dim = action_dim
        num_cameras = len(config.camera_order)

        # ── Build model ────────────────────────────────────────────────
        self.model = PranaVLA(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=config.chunk_size,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            dropout=config.dropout,
            num_denoise_steps=config.num_denoise_steps,
            noise_beta=config.noise_beta,
            time_sampling_alpha=config.time_sampling_alpha,
            time_sampling_beta=config.time_sampling_beta,
            freeze_backbone=config.freeze_backbone,
            unfreeze_last_n_blocks=config.unfreeze_last_n_blocks,
            num_cameras=num_cameras,
        )

        self._saved_tail_actions = None
        self.reset()

    def get_optim_params(self) -> list[dict]:
        """Differential LR: backbone at 1/10th if unfrozen."""
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "vision_encoder.backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        groups = [{"params": other_params, "lr": self.config.optimizer_lr}]
        if backbone_params:
            groups.append(
                {"params": backbone_params, "lr": self.config.optimizer_lr * 0.1}
            )
        return groups

    def reset(self):
        """Reset between episodes."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._saved_tail_actions = None

    # ──────────────────────────────────────────────────────────────────
    # NOISE SAMPLER INITIALIZATION
    # ──────────────────────────────────────────────────────────────────
    def fit_noise_sampler(self, dataloader):
        """
        Call ONCE before training to initialize the correlated noise sampler.

        Usage:
            policy.fit_noise_sampler(train_dataloader)
        """
        all_actions = []
        for batch in dataloader:
            if ACTION in batch:
                all_actions.append(batch[ACTION])
        if len(all_actions) == 0:
            print("[PRANA v3] Warning: no actions found, using uncorrelated noise")
            return

        all_actions = torch.cat(all_actions, dim=0)
        print(f"[PRANA v3] Fitting noise sampler on {all_actions.shape[0]} action chunks...")
        self.model.noise_sampler.fit(all_actions)
        print("[PRANA v3] Noise sampler fitted successfully.")

    # ──────────────────────────────────────────────────────────────────
    # IMAGE PREPROCESSING
    # ──────────────────────────────────────────────────────────────────
    def _prepare_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        images = []
        for cam_key in self.config.camera_order:
            img = TF.resize(batch[cam_key], [224, 224], antialias=True)
            images.append(img)
        return images

    # ──────────────────────────────────────────────────────────────────
    # TRAINING FORWARD
    # ──────────────────────────────────────────────────────────────────
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training: compute flow matching loss."""
        images = self._prepare_images(batch)
        states = batch.get(
            OBS_STATE,
            torch.zeros(images[0].shape[0], 6, device=images[0].device),
        )
        target_actions = batch[ACTION]

        # Handle action shape: [B, action_dim] → [B, chunk_size, action_dim]
        if target_actions.dim() == 2:
            target_actions = target_actions.unsqueeze(1).expand(
                -1, self.config.chunk_size, -1
            ).contiguous()

        # Truncate or pad to match chunk_size
        if target_actions.shape[1] > self.config.chunk_size:
            target_actions = target_actions[:, :self.config.chunk_size, :]
        elif target_actions.shape[1] < self.config.chunk_size:
            pad_size = self.config.chunk_size - target_actions.shape[1]
            last_action = target_actions[:, -1:, :].expand(-1, pad_size, -1)
            target_actions = torch.cat([target_actions, last_action], dim=1)

        output = self.model(images, states, target_actions)
        loss = output["loss"]

        # Ignore padded actions at episode boundaries
        if "action_is_pad" in batch and batch["action_is_pad"].any():
            v_pred = output.get("v_pred")
            v_target = output.get("v_target")
            if v_pred is not None and v_target is not None:
                mask = ~batch["action_is_pad"].unsqueeze(-1)
                loss = ((v_pred - v_target) ** 2 * mask).sum() / mask.sum().clamp(min=1)

        return loss, {"flow_loss": loss.item()}

    # ──────────────────────────────────────────────────────────────────
    # INFERENCE
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a full action chunk via iterative denoising."""
        self.eval()
        images = self._prepare_images(batch)
        states = batch.get(
            OBS_STATE,
            torch.zeros(images[0].shape[0], 6, device=images[0].device),
        )

        inpaint_actions = None
        inpaint_mask = None

        if self._saved_tail_actions is not None:
            B = states.shape[0]
            inpaint_steps = self.config.inpaint_steps

            inpaint_actions = torch.zeros(
                B, self.config.chunk_size, self.action_dim, device=states.device
            )
            inpaint_mask = torch.zeros(
                B, self.config.chunk_size, device=states.device, dtype=torch.bool
            )
            inpaint_actions[:, :inpaint_steps, :] = self._saved_tail_actions
            inpaint_mask[:, :inpaint_steps] = True

        return self.model.denoise(
            images, states,
            inpaint_actions=inpaint_actions,
            inpaint_mask=inpaint_mask,
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Select next action with rolling inpainting.

        Flow:
        1. Queue empty → predict new chunk
        2. Save tail for next chunk's inpainting
        3. Queue n_action_steps actions
        4. Pop one per call
        """
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch).squeeze(0)  # [chunk_size, action_dim]

            tail_start = self.config.n_action_steps
            tail_end = tail_start + self.config.inpaint_steps
            if tail_end <= self.config.chunk_size:
                self._saved_tail_actions = actions[tail_start:tail_end].unsqueeze(0)

            for t in range(self.config.n_action_steps):
                self._action_queue.append(actions[t])

        action = self._action_queue.popleft()

        # ── Denormalize ────────────────────────────────────────────────
        if self.dataset_stats is not None:
            action_stats = self.dataset_stats.get(ACTION) or self.dataset_stats.get("action")
            if action_stats is not None:
                mean = action_stats["mean"].to(action.device)
                std = action_stats["std"].to(action.device)
                action = action * std + mean

        return action