"""Fine-tune H64 with Stage-2 data and smoothness supervision."""

import sys
from train_curriculum import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "stage2_h64"]
    main()
