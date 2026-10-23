"""Collect corrective labels at H64 policy-visited states (supervised DAgger)."""

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
parser.add_argument("--max-steps", type=int, default=8)
parser.add_argument("--iteration", type=int, default=0)
parser.add_argument("--seed", type=int, default=243)
parser.add_argument("--planner-iterations", type=int, default=600)
parser.add_argument("--checkpoint", default="dagger_h64_best.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np
import torch

from src.env.dual_arm_scene import DualArmScene
from src.models.trajectory_model import TrajectoryModel, TrajectoryModelConfig, assert_models_independent
from src.training.dagger import DAggerCollector


def main():
    task = {}
    for side in ("left", "right"):
        with np.load(ROOT / "datasets" / side / "stage2_rrt_seed43_ep1.npz") as data:
            task[side] = {key: data[key][0] for key in ("q_start", "q_goal", "x_start", "x_goal")}
    models = {}
    for side in ("left", "right"):
        checkpoint = torch.load(ROOT / "checkpoints" / side / args.checkpoint, map_location=args.device, weights_only=False)
        config = TrajectoryModelConfig(**checkpoint["config"])
        models[side] = TrajectoryModel(config).to(args.device)
        models[side].load_compatible_state_dict(checkpoint["model"])
        models[side].eval()
    assert_models_independent(models["left"], models["right"])
    scene = DualArmScene(ROOT / "finally.usd", device=args.device, joint_stiffness=4000.0, joint_damping=200.0, joint_effort_limit=500.0)
    report = DAggerCollector(scene, models, ROOT / "datasets", args.seed, args.planner_iterations).collect(task, args.max_steps, args.iteration)
    (ROOT / "results" / f"dagger_collection_iter{args.iteration}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
