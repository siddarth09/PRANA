"""
PRANA v2 Encoders (DINOv2)
===========================
- VisionEncoder:        DINOv2 ViT-S/14 with camera-ID embeddings
- StateEncoder:         2-layer MLP projecting joint state to token space
- SinusoidalTimeEmb:    Sinusoidal encoding of flow-matching timestep t ∈ [0,1]
- ActionEncoder:        Projects noisy action chunk into token space for denoising

Dimensions:
  DINOv2 ViT-S/14 → embed_dim=384, 256 patch tokens per camera (patch14)
"""

import math
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """
    Frozen DINOv2 ViT-S/14 with camera-ID embeddings.
    Each camera gets a learned embedding added to all its patch tokens
    so the transformer knows which viewpoint produced each token.

    Input:  [B, 3, 224, 224]
    Output: [B, 256, hidden_dim]   (256 patch tokens per camera)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        num_cameras: int = 2,
    ):
        super().__init__()

        # ── Load DINOv2 ViT-S/14 ──────────────────────────────────────
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True,
        )
        vit_embed_dim = 384   # ViT-S/14 native embed dim

        # ── Freeze / selective unfreeze ────────────────────────────────
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            if unfreeze_last_n_blocks > 0:
                for block in self.backbone.blocks[-unfreeze_last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True

                if hasattr(self.backbone, "norm"):
                    for param in self.backbone.norm.parameters():
                        param.requires_grad = True

        # ── Camera identity embedding ──────────────────────────────────
        self.camera_embed = nn.Embedding(num_cameras, vit_embed_dim)

        # ── Projection to shared hidden_dim ───────────────────────────
        self.proj = nn.Linear(vit_embed_dim, hidden_dim)  # 384 → 256
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, image: torch.Tensor, camera_id: int = 0) -> torch.Tensor:
        """
        Args:
            image:     [B, 3, 224, 224]
            camera_id: 0 for table cam, 1 for wrist cam
        Returns:
                       [B, 256, hidden_dim]
        """
        # Extract patch tokens — excludes CLS token
        out = self.backbone.forward_features(image)
        features = out["x_norm_patchtokens"]                    # [B, 256, 384]

        # Add camera identity to every patch token
        cam_emb = self.camera_embed(
            torch.tensor(camera_id, device=image.device)
        )                                                        # [384]
        features = features + cam_emb.unsqueeze(0).unsqueeze(0) # [B, 256, 384]

        return self.norm(self.proj(features))                   # [B, 256, hidden_dim]


class StateEncoder(nn.Module):
    """
    Projects robot joint state → single token.
    2-layer MLP for expressiveness.
    """

    def __init__(self, state_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
        Returns:
               [B, 1, hidden_dim]
        """
        return self.net(state).unsqueeze(1)


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal encoding of flow-matching time t ∈ [0,1].
    Tells the denoising network how noisy the input is right now.

    t=1.0 → fully noisy
    t=0.0 → clean actions
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B]  flow time values in [0, 1]
        Returns:
               [B, hidden_dim]
        """
        half_dim = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, hidden_dim]
        return self.mlp(emb)


class ActionEncoder(nn.Module):
    """
    Projects noisy action chunk into token space for the decoder.
    Each timestep becomes one token with a learned positional embedding.
    """

    def __init__(self, action_dim: int = 6, chunk_size: int = 50, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(action_dim, hidden_dim)
        self.pos_embed = nn.Embedding(chunk_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_actions: [B, chunk_size, action_dim]
        Returns:
                           [B, chunk_size, hidden_dim]
        """
        B, T, _ = noisy_actions.shape
        pos_ids = torch.arange(T, device=noisy_actions.device)
        tokens = self.proj(noisy_actions) + self.pos_embed(pos_ids)
        return self.norm(tokens)