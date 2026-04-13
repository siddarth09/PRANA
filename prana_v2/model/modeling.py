"""
PRANA v2 — Flow Matching with Correlated Noise
===============================================

Architecture:

    ┌─────────────────────────────────────────────────────────────────┐
    │  TRAINING: sample t ~ Beta(1.5,1), ε ~ N(0, Σ_reg)           │
    │  x_t = t*ε + (1-t)*a_target   (noisy interpolation)           │
    │  Model predicts velocity v_t   (the "flow direction")          │
    │  Loss = ||v_t - (ε - a_target)||²                              │
    │                                                                 │
    │  INFERENCE: start from x_1 = ε ~ N(0, Σ_reg)                  │
    │  Iteratively denoise: x_{t-dt} = x_t - dt * v_t               │
    │  After K steps, x_0 ≈ predicted actions                        │
    └─────────────────────────────────────────────────────────────────┘

    Images ──► VisionEncoder ──┐
                               ├──► Context Encoder (self-attn, 4 layers)
    State  ──► StateEncoder  ──┘         │
                                         │ memory
                                         ▼
    x_t + t_emb ──► ActionEncoder ──► Action Decoder (cross-attn, 7 layers)
                                         │
                                         ▼
                                    Velocity Head ──► v_t [B, chunk, action_dim]

Why flow matching over CVAE:
  - Handles multimodal action distributions naturally (multiple valid ways
    to pick up the screwdriver)
  - No KL balancing headaches — just MSE on velocity
  - Correlated noise makes early denoising steps easier (the BEHAVIOR
    paper's key insight: noise that already looks like action structure
    means less work for the model)
  - At inference, iterative refinement is more robust than single-shot
"""

import torch
import torch.nn as nn
import numpy as np

from prana_v2.model.encoder import (
    VisionEncoder,
    StateEncoder,
    SinusoidalTimeEmbedding,
    ActionEncoder,
)


class CorrelatedNoiseSampler(nn.Module):
    """
    Samples noise from N(0, Σ_reg) where Σ_reg = β*Σ + (1-β)*I.

    Σ is the empirical covariance of action chunks from the training data.
    The correlated noise already has the temporal smoothness and cross-joint
    coordination structure of real robot trajectories, making the flow
    matching problem easier — especially at the noisiest steps (t ≈ 1).

    Must be initialized after loading the dataset by calling
    `fit(action_chunks)` with the training data.
    """

    def __init__(self, action_dim: int = 6, chunk_size: int = 50, beta: float = 0.5):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.beta = beta
        self.total_dim = action_dim * chunk_size

        # Cholesky factor of Σ_reg — registered as buffer (not a parameter)
        # Initialized as identity (standard noise) until fit() is called
        self.register_buffer(
            "cholesky_L", torch.eye(self.total_dim, dtype=torch.float32)
        )
        self.fitted = False

    def fit(self, action_chunks: torch.Tensor):
        """
        Compute Σ_reg from training action chunks and cache the Cholesky factor.

        Args:
            action_chunks: [N, chunk_size, action_dim] — all training action chunks
        """
        N = action_chunks.shape[0]
        # Flatten each chunk: [N, chunk_size * action_dim]
        flat = action_chunks.reshape(N, -1).float()

        # Empirical covariance (zero-mean since data should be normalized)
        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean
        sigma = (centered.T @ centered) / max(N - 1, 1)

        # Shrinkage regularization: Σ_reg = β*Σ + (1-β)*I
        eye = torch.eye(self.total_dim, device=sigma.device, dtype=sigma.dtype)
        sigma_reg = self.beta * sigma + (1.0 - self.beta) * eye

        # Cholesky decomposition for efficient sampling
        L = torch.linalg.cholesky(sigma_reg)
        self.cholesky_L.copy_(L)
        self.fitted = True

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample correlated noise ε ~ N(0, Σ_reg).

        Returns: [B, chunk_size, action_dim]
        """
        z = torch.randn(batch_size, self.total_dim, device=device)
        L = self.cholesky_L.to(device)
        correlated = z @ L.T  # equivalent to L @ z for each sample
        return correlated.reshape(batch_size, self.chunk_size, self.action_dim)

    def sample_uncorrelated(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Fallback: standard Gaussian noise (before fit() is called)."""
        return torch.randn(batch_size, self.chunk_size, self.action_dim, device=device)


class PranaVLA(nn.Module):
    """PRANA v2: Flow matching with correlated noise for robot action prediction."""

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
        # Flow matching
        num_denoise_steps: int = 10,
        noise_beta: float = 0.5,
        time_sampling_alpha: float = 1.5,
        time_sampling_beta: float = 1.0,
        # Vision
        vision_backbone: str = "vit_tiny_patch16_224",
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
            model_name=vision_backbone,
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
        # VELOCITY HEAD — predicts the flow direction v_t
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

    # ──────────────────────────────────────────────────────────────────
    # CONTEXT ENCODING (shared between training and inference)
    # ──────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────
    # SINGLE DENOISE STEP (velocity prediction)
    # ──────────────────────────────────────────────────────────────────
    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the flow velocity v_t given noisy actions x_t and time t.

        Args:
            x_t:     [B, chunk_size, action_dim]  noisy actions
            t:       [B]                           flow time
            context: [B, ctx_len, hidden_dim]      encoded observations
        Returns:
            v_t:     [B, chunk_size, action_dim]   predicted velocity
        """
        # Encode noisy actions as tokens
        action_tokens = self.action_encoder(x_t)  # [B, chunk_size, hidden_dim]

        # Add time conditioning to every action token
        t_emb = self.time_encoder(t)  # [B, hidden_dim]
        action_tokens = action_tokens + t_emb.unsqueeze(1)

        # Cross-attend to context
        decoded = self.action_decoder(
            tgt=action_tokens,
            memory=context,
        )  # [B, chunk_size, hidden_dim]

        # Predict velocity
        return self.velocity_head(decoded)  # [B, chunk_size, action_dim]

    # ──────────────────────────────────────────────────────────────────
    # TRAINING FORWARD
    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        state: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> dict:
        """
        Training forward: compute flow matching loss.

        The flow matching objective:
          x_t = t * ε + (1 - t) * a        (interpolate between noise and target)
          target_velocity = ε - a            (the true flow direction)
          loss = ||v_t_predicted - target_velocity||²

        Args:
            images:         list of [B, 3, 224, 224] per camera
            state:          [B, state_dim]
            target_actions: [B, chunk_size, action_dim]
        Returns:
            dict with "loss", "velocity_mse"
        """
        B = target_actions.shape[0]
        device = target_actions.device

        # 1. Encode context (one forward pass, shared)
        context = self.encode_context(images, state)

        # 2. Sample flow time t ~ Beta(α, β) biased toward harder steps
        t = torch.distributions.Beta(self.time_alpha, self.time_beta).sample((B,)).to(device)

        # 3. Sample correlated noise
        if self.noise_sampler.fitted:
            epsilon = self.noise_sampler.sample(B, device)
        else:
            epsilon = self.noise_sampler.sample_uncorrelated(B, device)

        # 4. Compute noisy actions via flow interpolation
        t_expanded = t[:, None, None]  # [B, 1, 1]
        x_t = t_expanded * epsilon + (1.0 - t_expanded) * target_actions

        # 5. Predict velocity
        v_pred = self.predict_velocity(x_t, t, context)

        # 6. Compute loss: target velocity is (ε - a)
        target_velocity = epsilon - target_actions
        loss = nn.functional.mse_loss(v_pred, target_velocity)

        return {"loss": loss, "velocity_mse": loss.item(), "v_pred": v_pred, "v_target": target_velocity}

    # ──────────────────────────────────────────────────────────────────
    # INFERENCE: iterative denoising
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def denoise(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        state: torch.Tensor,
        inpaint_actions: torch.Tensor | None = None,
        inpaint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate action chunk via iterative denoising (Euler method).

        Start from noise x_1 and step backward to x_0.

        Args:
            images:          camera observations
            state:           robot state
            inpaint_actions: [B, chunk_size, action_dim] — known actions for overlap
            inpaint_mask:    [B, chunk_size] — True where actions are known (inpaint)
        Returns:
            [B, chunk_size, action_dim] denoised action chunk
        """
        B = state.shape[0]
        device = state.device
        K = self.num_denoise_steps

        # Encode context once
        context = self.encode_context(images, state)

        # Start from correlated noise
        if self.noise_sampler.fitted:
            x_t = self.noise_sampler.sample(B, device)
        else:
            x_t = self.noise_sampler.sample_uncorrelated(B, device)

        # Save the initial noise for inpainting consistency
        z_init = x_t.clone() if inpaint_actions is not None else None

        # Step from t=1 to t=0
        dt = 1.0 / K
        for step in range(K):
            t_val = 1.0 - step * dt  # current time
            t = torch.full((B,), t_val, device=device)

            # Predict velocity
            v_pred = self.predict_velocity(x_t, t, context)

            # Euler step: x_{t-dt} = x_t - dt * v_t
            x_t = x_t - dt * v_pred

            # ── Inpainting: soft constraint on known actions ───────────
            # Apply only during early steps (t > 0.3) — let model be free
            # near the end so it can adapt to current observations
            if inpaint_actions is not None and inpaint_mask is not None and t_val > 0.3:
                t_next = t_val - dt
                # Where the inpainted actions should be at time t_next
                desired = t_next * z_init + (1.0 - t_next) * inpaint_actions
                # Hard-set known dimensions
                mask = inpaint_mask.unsqueeze(-1).float()  # [B, chunk_size, 1]
                x_t = mask * desired + (1.0 - mask) * x_t

        return x_t