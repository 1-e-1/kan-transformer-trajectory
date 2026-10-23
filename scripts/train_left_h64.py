"""Train the independent left H64 policy."""

import sys
from train_single_h64 import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--side", "left"]
    main()
