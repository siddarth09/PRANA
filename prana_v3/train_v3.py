"""
PRANA v3 Training — LeRobot native training with monkey-patch registration.

Usage:
    python3 prana_v3/train_v3.py \
        --dataset.repo_id=Siddarth09/PRANA \
        --dataset.video_backend=pyav \
        --dataset.image_transforms.enable=false \
        --policy.type=prana_v3 \
        --policy.device=cuda \
        --policy.camera_order='["observation.images.table","observation.images.wrist"]' \
        --batch_size=8 \
        --num_workers=0 \
        --steps=85000 \
        --policy.push_to_hub=false \
        --output_dir=outputs/train/prana_v3 \
        --wandb.enable=true
"""

import os
import sys

# ── Ensure project root is on path ─────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Register PRANA v3 with LeRobot's factory ───────────────────────
from prana_v3.model.configuration_prana import PranaConfig
from prana_v3.model.policy_prana import PranaPolicy
import lerobot.policies.factory as factory
from lerobot.policies.act.processor_act import make_act_pre_post_processors


original_get_policy_class = factory.get_policy_class

def custom_get_policy_class(name: str):
    if name == "prana_v3":
        return PranaPolicy
    return original_get_policy_class(name)

factory.get_policy_class = custom_get_policy_class


original_make_pre_post = factory.make_pre_post_processors

def custom_make_pre_post(policy_cfg, pretrained_path=None, **kwargs):
    if pretrained_path is not None:
        return original_make_pre_post(policy_cfg, pretrained_path=pretrained_path, **kwargs)
    if isinstance(policy_cfg, PranaConfig) or getattr(policy_cfg, "type", None) == "prana_v3":
        dataset_stats = kwargs.get("dataset_stats")
        return make_act_pre_post_processors(policy_cfg, dataset_stats)
    return original_make_pre_post(policy_cfg, pretrained_path=pretrained_path, **kwargs)

factory.make_pre_post_processors = custom_make_pre_post


# ── Hand off to LeRobot's training entrypoint ──────────────────────
if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import train
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.parser import parse_arg

    config = parse_arg(TrainPipelineConfig)
    train(config)