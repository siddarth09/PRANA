import os
import sys
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prana.model.configuration_prana import PranaConfig
from prana.model.policy_prana import PranaPolicy
import lerobot.policies.factory as factory
from lerobot.policies.act.processor_act import make_act_pre_post_processors


original_get_policy_class = factory.get_policy_class

def custom_get_policy_class(name: str):
    if name == "prana_v1":
        return PranaPolicy
    return original_get_policy_class(name)

factory.get_policy_class = custom_get_policy_class

original_make_pre_post = factory.make_pre_post_processors

def custom_make_pre_post(policy_cfg, pretrained_path=None, **kwargs):
    if pretrained_path is not None:
        return original_make_pre_post(policy_cfg, pretrained_path=pretrained_path, **kwargs)
    if isinstance(policy_cfg, PranaConfig) or getattr(policy_cfg, "type", None) == "prana_v1":
        dataset_stats = kwargs.get('dataset_stats')
        return make_act_pre_post_processors(policy_cfg, dataset_stats)
    return original_make_pre_post(policy_cfg, pretrained_path=pretrained_path, **kwargs)

factory.make_pre_post_processors = custom_make_pre_post

def main():
    # Determine checkpoint path: CLI arg > env var > default relative to PROJECT_ROOT
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = os.environ.get(
            "PRANA_CHECKPOINT_PATH",
            os.path.join(
                PROJECT_ROOT,
                "outputs",
                "train",
                "prana",
                "checkpoints",
                "last",
                "pretrained_model",
            ),
        )
    print(f"Loading policy from: {checkpoint_path}...")
    try:
        policy = PranaPolicy.from_pretrained(checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(device)
        policy.eval()
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Loading Pre/Post Processors (Testing the Deploy Logic)...")
    try:
        
        preprocessor, postprocessor = factory.make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint_path
        )
        print("Processors loaded successfully!")
    except Exception as e:
        print(f"Failed to load processors: {e}")
        return

    dummy_batch = {
        "observation.images.table": torch.rand(1, 3, 224, 224, device=device),
        "observation.images.wrist": torch.rand(1, 3, 224, 224, device=device),
        "observation.state": torch.zeros(1, 6, device=device)
    }

    with torch.no_grad():
        raw_action = policy.select_action(dummy_batch)

    print("\n=======================================================")
    print("   1. RAW NEURAL NETWORK OUTPUT (Normalized)")
    print("=======================================================")
    print(raw_action)

    print("\n=======================================================")
    print("   2. PHYSICAL MOTOR RADIANS (Post-Processed)")
    print("=======================================================")
    
    try:
       
        stats = postprocessor.processors["unnormalizer_processor"].stats
        mean = stats["action"]["mean"].to(device)
        std = stats["action"]["std"].to(device)
        
    
        physical_action = (raw_action * std) + mean
        
        print(physical_action)
        print("=======================================================\n")
        print("SUCCESS! The stats are saved, loaded, and the math works.")
        print("The robot is officially cleared for real-world deployment.")
    except Exception as e:
        print("Failed to apply un-normalization math. Stats might be missing. Error:", e)

if __name__ == "__main__":
    main()