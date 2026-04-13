"""
Fit Correlated Noise Sampler & Patch Checkpoint
=================================================
Reads all actions from the dataset, computes the covariance matrix,
fits the Cholesky factor, and saves it into the existing checkpoint.

Usage:
    python3 prana_v2/fit_noise.py \
        --dataset Siddarth09/PRANA \
        --checkpoint outputs/train/prana_v2/checkpoints/last/pretrained_model
"""

import os
import sys
import argparse
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prana_v2.model.configuration_prana import PranaConfig
from prana_v2.model.policy_prana import PranaPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Siddarth09/PRANA")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/train/prana_v2/checkpoints/last/pretrained_model")
    parser.add_argument("--chunk_size", type=int, default=50)
    args = parser.parse_args()

    # ── Load actions from dataset ──────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    ds = LeRobotDataset(repo_id=args.dataset, video_backend="pyav")
    raw = ds.hf_dataset["action"]

    # Convert Arrow Column → tensor
    print(f"  Total frames: {len(raw)}")
    action_list = raw.to_pylist() if hasattr(raw, "to_pylist") else [a.tolist() for a in raw]
    actions_flat = torch.tensor(action_list, dtype=torch.float32)
    print(f"  Actions shape: {actions_flat.shape}")

    # Reshape into chunks
    chunk = args.chunk_size
    n_chunks = actions_flat.shape[0] // chunk
    chunks = actions_flat[:n_chunks * chunk].reshape(n_chunks, chunk, -1)
    print(f"  Chunks: {chunks.shape} ({n_chunks} chunks of {chunk}x{chunks.shape[2]})")

    # ── Compute covariance and Cholesky ────────────────────────────
    action_dim = chunks.shape[2]
    total_dim = chunk * action_dim
    flat = chunks.reshape(n_chunks, -1)

    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    sigma = (centered.T @ centered) / max(n_chunks - 1, 1)

    beta = 0.5
    eye = torch.eye(total_dim, dtype=torch.float32)
    sigma_reg = beta * sigma + (1.0 - beta) * eye

    print(f"  Computing Cholesky of {total_dim}x{total_dim} matrix...")
    L = torch.linalg.cholesky(sigma_reg)
    print(f"  Cholesky factor computed successfully!")

    # ── Load checkpoint and patch ──────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    policy = PranaPolicy.from_pretrained(args.checkpoint)

    # Patch the noise sampler
    policy.model.noise_sampler.cholesky_L.copy_(L)
    policy.model.noise_sampler.fitted = True
    print(f"  Noise sampler patched (fitted=True)")

    # ── Save back ──────────────────────────────────────────────────
    print(f"  Saving to: {args.checkpoint}")
    policy.save_pretrained(args.checkpoint)
    print(f"\n  Done! Correlated noise sampler is now in the checkpoint.")
    print(f"  Re-deploy and the jitter should be significantly reduced.")


if __name__ == "__main__":
    main()