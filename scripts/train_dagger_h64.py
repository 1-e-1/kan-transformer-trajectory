"""Fine-tune H64 on aggregated DAgger corrective samples."""

import sys
from train_curriculum import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--mode", "dagger_h64"]
    main()
