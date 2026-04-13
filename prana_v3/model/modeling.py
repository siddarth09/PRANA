"""
PRANA v3 — Flow Matching with Correlated Noise + DINOv2
========================================================

Architecture:

    ┌─────────────────────────────────────────────────────────────────┐
    │  TRAINING: sample t ~ Beta(1.5,1), ε ~ N(0, Σ_reg)              │
    │  x_t = t*ε + (1-t)*a_target   (noisy interpolation)             │
    │  Model predicts velocity v_t   (the "flow direction")           │
    │  Loss = ||v_t - (ε - a_target)||²                               │
    │                                                                 │
    │  INFERENCE: start from x_1 = ε ~ N(0, Σ_reg)                    │
    │  Iteratively denoise: x_{t-dt} = x_t - dt * v_t                 │
    │  After K steps, x_0 ≈ predicted actions                         │
    └─────────────────────────────────────────────────────────────────┘

    Images ──► VisionEncoder (DINOv2 ViT-S/14) ──┐
                                                  ├──► Context Encoder (self-attn, 4 layers)
    State  ──► StateEncoder  ────────────────────┘         │
                                                           │ memory
                                                           ▼
    x_t + t_emb ──► ActionEncoder ──► Action Decoder (cross-attn, 7 layers)
                                                           │
                                                           ▼
                                                  Velocity Head ──► v_t [B, chunk, action_dim]
"""

import torch
import torch.nn as nn
from prana_v3.model.encoder import (
    VisionEncoder,
    StateEncoder,
    SinusoidalTimeEmbedding,
    ActionEncoder,
)


class CorrelatedNoiseSampler(nn.Module):
    """
    Samples noise from N(0, Σ_reg) where Σ_reg = β*Σ + (1-β)*I.

    Σ is the empirical covariance of action chunks from the training data.
    Must be initialized by calling fit(action_chunks) before training.
    """

    def __init__(self, action_dim: int = 6, chunk_size: int = 50, beta: float = 0.5):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.beta = beta
        self.total_dim = action_dim * chunk_size

        self.register_buffer(
            "cholesky_L", torch.eye(self.total_dim, dtype=torch.float32)
        )
        self.fitted = False

    def fit(self, action_chunks: torch.Tensor):
        """
        Args:
            action_chunks: [N, chunk_size, action_dim]
        """
        N = action_chunks.shape[0]
        flat = action_chunks.reshape(N, -1).float()

        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean
        sigma = (centered.T @ centered) / max(N - 1, 1)

        eye = torch.eye(self.total_dim, device=sigma.device, dtype=sigma.dtype)
        sigma_reg = self.beta * sigma + (1.0 - self.beta) * eye

        L = torch.linalg.cholesky(sigma_reg)
        self.cholesky_L.copy_(L)
        self.fitted = True

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns: [B, chunk_size, action_dim]"""
        z = torch.randn(batch_size, self.total_dim, device=device)
        L = self.cholesky_L.to(device)
        correlated = z @ L.T
        return correlated.reshape(batch_size, self.chunk_size, self.action_dim)

    def sample_uncorrelated(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Fallback before fit() is called."""
        return torch.randn(batch_size, self.chunk_size, self.action_dim, device=device)


class PranaVLA(nn.Module):
    """PRANA v3: Flow matching + DINOv2 vision for robot action prediction."""

    def __init__(
        self,
        action_dim: int = 6,
        state_dim: int = 6,
        chunk_size: int = 50,
        hidden_dim: int = 256,
        n_heads: int = 8,
        dim_feedforward: int = 512,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 7,
        dropout: float = 0.1,
        num_denoise_steps: int = 10,
        noise_beta: float = 0.5,
        time_sampling_alpha: float = 1.5,
        time_sampling_beta: float = 1.0,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        num_cameras: int = 2,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_denoise_steps = num_denoise_steps
        self.time_alpha = time_sampling_alpha
        self.time_beta = time_sampling_beta

        # ══════════════════════════════════════════════════════════════
        # ENCODERS
        # ══════════════════════════════════════════════════════════════
        self.vision_encoder = VisionEncoder(
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            num_cameras=num_cameras,
        )
        self.state_encoder = StateEncoder(state_dim=state_dim, hidden_dim=hidden_dim)
        self.time_encoder = SinusoidalTimeEmbedding(hidden_dim=hidden_dim)
        self.action_encoder = ActionEncoder(
            action_dim=action_dim, chunk_size=chunk_size, hidden_dim=hidden_dim
        )

        # ══════════════════════════════════════════════════════════════
        # CORRELATED NOISE SAMPLER
        # ══════════════════════════════════════════════════════════════
        self.noise_sampler = CorrelatedNoiseSampler(
            action_dim=action_dim, chunk_size=chunk_size, beta=noise_beta
        )

        # ══════════════════════════════════════════════════════════════
        # CONTEXT ENCODER (self-attention over vision + state)
        # ══════════════════════════════════════════════════════════════
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers
        )

        # ══════════════════════════════════════════════════════════════
        # ACTION DECODER (cross-attention: noisy actions attend to context)
        # ══════════════════════════════════════════════════════════════
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_decoder_layers
        )

        # ══════════════════════════════════════════════════════════════
        # VELOCITY HEAD
        # ══════════════════════════════════════════════════════════════
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for non-pretrained parameters."""
        for name, p in self.named_parameters():
            if "vision_encoder.backbone" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_context(
        self, images: list[torch.Tensor] | torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode visual + state observations into context memory.
        Returns: [B, context_len, hidden_dim]
        """
        if isinstance(images, list):
            all_v = [self.vision_encoder(img, cam_id) for cam_id, img in enumerate(images)]
            v_tokens = torch.cat(all_v, dim=1)
        else:
            v_tokens = self.vision_encoder(images, camera_id=0)

        s_tokens = self.state_encoder(state)
        context = torch.cat([v_tokens, s_tokens], dim=1)
        return self.context_encoder(context)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:     [B, chunk_size, action_dim]
            t:       [B]
            context: [B, ctx_len, hidden_dim]
        Returns:
                     [B, chunk_size, action_dim]
        """
        action_tokens = self.action_encoder(x_t)
        t_emb = self.time_encoder(t)
        action_tokens = action_tokens + t_emb.unsqueeze(1)

        decoded = self.action_decoder(tgt=action_tokens, memory=context)
        return self.velocity_head(decoded)

    def forward(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        state: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> dict:
        """
        Training forward: compute flow matching loss.

          x_t = t * ε + (1 - t) * a
          target_velocity = ε - a
          loss = ||v_pred - target_velocity||²
        """
        B = target_actions.shape[0]
        device = target_actions.device

        context = self.encode_context(images, state)

        t = torch.distributions.Beta(self.time_alpha, self.time_beta).sample((B,)).to(device)

        if self.noise_sampler.fitted:
            epsilon = self.noise_sampler.sample(B, device)
        else:
            epsilon = self.noise_sampler.sample_uncorrelated(B, device)

        t_expanded = t[:, None, None]
        x_t = t_expanded * epsilon + (1.0 - t_expanded) * target_actions

        v_pred = self.predict_velocity(x_t, t, context)

        target_velocity = epsilon - target_actions
        loss = nn.functional.mse_loss(v_pred, target_velocity)

        return {
            "loss": loss,
            "velocity_mse": loss.item(),
            "v_pred": v_pred,
            "v_target": target_velocity,
        }

    @torch.no_grad()
    def denoise(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        state: torch.Tensor,
        inpaint_actions: torch.Tensor | None = None,
        inpaint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Iterative denoising from t=1 to t=0 (Euler method).
        Returns: [B, chunk_size, action_dim]
        """
        B = state.shape[0]
        device = state.device
        K = self.num_denoise_steps

        context = self.encode_context(images, state)

        if self.noise_sampler.fitted:
            x_t = self.noise_sampler.sample(B, device)
        else:
            x_t = self.noise_sampler.sample_uncorrelated(B, device)

        z_init = x_t.clone() if inpaint_actions is not None else None

        dt = 1.0 / K
        for step in range(K):
            t_val = 1.0 - step * dt
            t = torch.full((B,), t_val, device=device)

            v_pred = self.predict_velocity(x_t, t, context)
            x_t = x_t - dt * v_pred

            if inpaint_actions is not None and inpaint_mask is not None and t_val > 0.3:
                t_next = t_val - dt
                desired = t_next * z_init + (1.0 - t_next) * inpaint_actions
                mask = inpaint_mask.unsqueeze(-1).float()
                x_t = mask * desired + (1.0 - mask) * x_t

        return x_t