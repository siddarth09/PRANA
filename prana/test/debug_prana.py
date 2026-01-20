#!/usr/bin/env python3
import sys

# IMPORTANT: load third-party plugins *before* importing lerobot_train (which builds CLI choices)
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()

# Now import the real CLI entry and run it
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    sys.exit(main())
