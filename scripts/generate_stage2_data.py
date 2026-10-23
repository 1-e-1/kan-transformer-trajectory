"""Generate Stage-2 demonstrations with a PhysX-backed RRT-Connect teacher."""

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
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=142)
parser.add_argument("--max-attempts", type=int, default=500)
parser.add_argument("--planner-iterations", type=int, default=1600)
parser.add_argument("--max-planner-failures", type=int, default=5)
parser.add_argument("--accept-direct-safe", action="store_true", help="Build diverse planner-validated Stage-2 data instead of hard-only tasks")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from src.data.stage2_generator import Stage2DataGenerator
from src.env.dual_arm_scene import DualArmScene


def main():
    scene = DualArmScene(ROOT / "finally.usd", device=args.device, execution_mode="kinematic")
    report = Stage2DataGenerator(scene, ROOT / "datasets", seed=args.seed, max_sampling_attempts=args.max_attempts, planner_iterations=args.planner_iterations, max_planner_failures=args.max_planner_failures, require_direct_collision=not args.accept_direct_safe).generate(args.episodes)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    status = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(status)
