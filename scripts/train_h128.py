"""Warm-start an H128 model from the finalized H64 checkpoint."""

import sys
from train_curriculum import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "h128"]
    main()
