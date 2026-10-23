"""Run one DAgger collection iteration, then supervised H64 fine-tuning."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    report = ROOT / "results" / f"dagger_collection_iter{args.iteration}.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "collect_dagger.py"), "--headless",
        "--iteration", str(args.iteration), "--max-steps", str(args.rollout_steps),
    ], cwd=ROOT, check=True)
    if not report.exists():
        raise RuntimeError(f"DAgger collection did not produce {report}")
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "train_dagger_h64.py"),
        "--epochs", str(args.epochs),
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
