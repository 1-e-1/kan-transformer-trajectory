"""Generate a small-amplitude live-FK dataset solely for the tiny-overfit gate."""

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
parser.add_argument("--trajectory-steps", type=int, default=160)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np

from src.data.generator import Stage1DataGenerator
from src.env.dual_arm_scene import DualArmScene


def main():
    scene = DualArmScene(ROOT / "finally.usd", device=args.device, joint_stiffness=4000.0, joint_damping=200.0, joint_effort_limit=500.0)
    generator = Stage1DataGenerator(scene, ROOT / "datasets" / "tiny", history_length=64, trajectory_steps=args.trajectory_steps, seed=args.seed)
    q_start = {"left": np.zeros(8, dtype=np.float32), "right": np.zeros(8, dtype=np.float32)}
    q_goal = {
        "left": np.array([0.30, -0.25, 0.20, -0.15, 0.10, -0.08, 0.05, -0.03], dtype=np.float32),
        "right": np.array([-0.25, 0.20, -0.18, 0.12, -0.10, 0.07, -0.04, 0.02], dtype=np.float32),
    }
    report = generator.generate_fixed_tiny(q_start, q_goal)
    (ROOT / "results" / "tiny_data_generation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)

