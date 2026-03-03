import os
import sys

# 1. PATH FIX
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 2. Import everything we need to patch
from prana.model.configuration_prana import PranaConfig
from prana.model.policy_prana import PranaPolicy

import lerobot.policies.factory as factory
import lerobot.scripts.lerobot_train as train_script # Import the train script itself!
from lerobot.policies.act.processor_act import make_act_pre_post_processors

# --- 3. THE DEEP PATCH: Inject PranaPolicy into the lowest level of the factory ---
original_get_policy_class = factory.get_policy_class

def custom_get_policy_class(name: str):
    if name == "prana_v1":
        return PranaPolicy
    return original_get_policy_class(name)

factory.get_policy_class = custom_get_policy_class


# --- 4. THE NAMESPACE PATCH: Force the Train script to use our custom processors ---
original_make_pre_post = factory.make_pre_post_processors

def custom_make_pre_post(policy_cfg, **kwargs):
    if isinstance(policy_cfg, PranaConfig) or policy_cfg.type == "prana_v1":
        dataset_stats = kwargs.get('dataset_stats')
        return make_act_pre_post_processors(policy_cfg, dataset_stats)
    return original_make_pre_post(policy_cfg, **kwargs)

# Overwrite it in the factory AND directly in the train script's memory
factory.make_pre_post_processors = custom_make_pre_post
train_script.make_pre_post_processors = custom_make_pre_post


if __name__ == "__main__":
    print("\n=======================================================")
    print("   PRANA PHASE 1.5 LEROBOT INJECTION SUCCESSFUL")
    print("=======================================================\n")
    
    # Run the main training loop
    train_script.main()