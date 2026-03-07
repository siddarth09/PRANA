import os
import sys
import torch
from safetensors.torch import load_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prana.model.configuration_prana import PranaConfig
from prana.model.policy_prana import PranaPolicy

def main():
    checkpoint_path = "/home/sid/projects25/src/PRANA/prana_v1/outputs/train/prana/checkpoints/085000/pretrained_model"
    
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
    print("   2. PHYSICAL MOTOR RADIANS (Un-Normalized)")
    print("=======================================================")
    
    try:
       
        stats_file = os.path.join(checkpoint_path, "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
        print(f"Loading stats from: {stats_file}")
        
        
        tensors = load_file(stats_file)
        
        
        action_mean = tensors.get("stats.action.mean")
        if action_mean is None:
             action_mean = tensors.get("action.mean") 
        action_std = tensors.get("stats.action.std")
        if action_std is None:
             action_std = tensors.get("action.std")

        if action_mean is None or action_std is None:
             print("Could not locate 'action.mean' or 'action.std' keys in the safetensors file.")
             print("Keys found:", list(tensors.keys()))
             return

        action_mean = action_mean.to(device)
        action_std = action_std.to(device)

       
        physical_action = (raw_action * action_std) + action_mean
        
        print(physical_action)
        print("=======================================================\n")
        print("SUCCESS! The un-normalization math is sound.")

    except Exception as e:
        print(f"Failed to un-normalize manually: {e}")

if __name__ == "__main__":
    main()