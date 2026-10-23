"""Generate simultaneous collision-filtered left/right Stage-1 datasets in Isaac Sim."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--trajectory-steps", type=int, default=80)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-sampling-attempts", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

from src.data.generator import Stage1DataGenerator
from src.env.dual_arm_scene import DualArmScene


def main():
    scene = DualArmScene(ROOT / "finally.usd", device=args.device, joint_stiffness=4000.0, joint_damping=200.0, joint_effort_limit=500.0)
    generator = Stage1DataGenerator(scene, ROOT / "datasets", history_length=64, trajectory_steps=args.trajectory_steps, seed=args.seed, max_sampling_attempts=args.max_sampling_attempts)
    report = generator.generate(args.episodes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)

