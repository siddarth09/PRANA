import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prana_v2.model.policy_prana import PranaPolicy
import lerobot.policies.factory as factory
import lerobot.scripts.lerobot_record as record_script

original_get_policy_class = factory.get_policy_class

def custom_get_policy_class(name: str):
    if name == "prana_v2":
        return PranaPolicy
    return original_get_policy_class(name)

factory.get_policy_class = custom_get_policy_class


if __name__ == "__main__":
    print("\n=======================================================")
    print("   PRANA PHASE 2.0")
    print("=======================================================\n")
    record_script.main()