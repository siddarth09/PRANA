"""
PRANA v2 Encoders
=================
- VisionEncoder:  Frozen ViT-Tiny with camera-ID embeddings
- StateEncoder:   2-layer MLP projecting joint state to token space
- TimeEmbedding:  Sinusoidal encoding of flow-matching timestep t ∈ [0,1]
- ActionEncoder:  Projects noisy action chunk into token space for denoising
"""

import math
import torch
import torch.nn as nn
import timm


class VisionEncoder(nn.Module):
    """
    Frozen pretrained vision backbone with camera-ID embeddings.
    Supports both ViT-style (token output) and CNN-style (spatial output) backbones.
    """

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        hidden_dim: int = 256,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        num_cameras: int = 2,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )

        # ── Detect backbone embed dim ──────────────────────────────────
        # Run a dummy forward to get the output shape
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            dummy_out = self.backbone(dummy)

        if dummy_out.dim() == 3:
            # ViT-style: [B, num_tokens, embed_dim]
            self.backbone_type = "vit"
            backbone_dim = dummy_out.shape[-1]
        elif dummy_out.dim() == 4:
            if dummy_out.shape[1] > dummy_out.shape[-1]:
                # CNN-style: [B, C, H, W] — channels first
                self.backbone_type = "cnn_chw"
                backbone_dim = dummy_out.shape[1]
            else:
                # Swin-style: [B, H, W, C] — channels last
                self.backbone_type = "cnn_hwc"
                backbone_dim = dummy_out.shape[-1]
        else:
            raise ValueError(f"Unexpected backbone output shape: {dummy_out.shape}")

        self.backbone_dim = backbone_dim

        # ── Freeze / selective unfreeze ────────────────────────────────
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_n_blocks > 0 and hasattr(self.backbone, "blocks"):
                for block in self.backbone.blocks[-unfreeze_last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True
                if hasattr(self.backbone, "norm"):
                    for param in self.backbone.norm.parameters():
                        param.requires_grad = True

        # ── Camera identity ────────────────────────────────────────────
        self.camera_embed = nn.Embedding(num_cameras, backbone_dim)

        # ── Projection ─────────────────────────────────────────────────
        self.proj = nn.Linear(backbone_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, image: torch.Tensor, camera_id: int = 0) -> torch.Tensor:
        """
        [B, 3, 224, 224] → [B, num_tokens, hidden_dim]
        Works with ViT (token output) and CNN (spatial output) backbones.
        """
        raw = self.backbone(image)

        # Reshape to [B, num_tokens, backbone_dim]
        if self.backbone_type == "vit":
            features = raw                                      # [B, N, D] already
        elif self.backbone_type == "cnn_chw":
            B, C, H, W = raw.shape
            features = raw.flatten(2).transpose(1, 2)           # [B, H*W, C]
        elif self.backbone_type == "cnn_hwc":
            B, H, W, C = raw.shape
            features = raw.reshape(B, H * W, C)                 # [B, H*W, C]

        # Add camera identity
        cam_emb = self.camera_embed(
            torch.tensor(camera_id, device=image.device)
        )
        features = features + cam_emb.unsqueeze(0).unsqueeze(0)

        return self.norm(self.proj(features))


class StateEncoder(nn.Module):
    """Projects robot state → single token. 2-layer MLP for expressiveness."""

    def __init__(self, state_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """[B, state_dim] → [B, 1, hidden_dim]"""
        return self.net(state).unsqueeze(1)


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for the flow-matching time t ∈ [0,1].
    Then projected through a 2-layer MLP (AdaRMS-style conditioning).

    This tells the denoising network "how noisy is the input right now?"
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
            t: [B] flow time values in [0, 1]
        Returns:
            [B, hidden_dim] time embedding
        """
        half_dim = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        # t is [B], freqs is [half_dim] → outer product
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0  # scale up for resolution
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, hidden_dim]
        return self.mlp(emb)


class ActionEncoder(nn.Module):
    """
    Projects noisy action chunk x_t into token space for the decoder.
    Each timestep in the chunk becomes one token.

    Includes learned positional embeddings so the decoder knows
    which timestep each action token represents.
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