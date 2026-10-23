"""Run two independent tiny-overfit models in the required one-step closed loop."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--max-steps", type=int, default=180)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch

from src.data.dataset import ArmTrajectoryDataset
from src.env.dual_arm_scene import DualArmScene
from src.evaluation.metrics import trajectory_metrics
from src.evaluation.rollout import dual_closed_loop_rollout
from src.models.trajectory_model import TrajectoryModel, TrajectoryModelConfig, assert_models_independent


def write_row(path, row):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main():
    datasets = {side: ArmTrajectoryDataset(ROOT / "datasets" / "tiny" / side) for side in ("left", "right")}
    task = {}
    for side in ("left", "right"):
        sample = datasets[side][0]
        task[side] = {key: sample[key].numpy() for key in ("q_start", "q_goal", "x_start", "x_goal")}
    models = {}
    for side in ("left", "right"):
        checkpoint = torch.load(ROOT / "checkpoints" / side / "tiny_overfit_best.pt", map_location=args.device, weights_only=False)
        config = TrajectoryModelConfig(**checkpoint["config"])
        models[side] = TrajectoryModel(config).to(args.device)
        models[side].load_compatible_state_dict(checkpoint["model"])
        models[side].eval()
    assert_models_independent(models["left"], models["right"])
    scene = DualArmScene(ROOT / "finally.usd", device=args.device, joint_stiffness=4000.0, joint_damping=200.0, joint_effort_limit=500.0)
    rollout = dual_closed_loop_rollout(scene, models, task, max_steps=args.max_steps)
    metrics = {}
    for side in ("left", "right"):
        trace = rollout[side]
        metrics[side] = trajectory_metrics(trace["q"], trace["tcp_position"], scene.control_dt, trace["joint_error"], trace["position_error"], trace["orientation_error"], trace["success"], trace["self_collisions"], trace["inter_collisions"], trace["latency_ms"])
        write_row(ROOT / "results" / f"{side}_metrics.csv", metrics[side])
    scene_metrics = {"dual_success": int(metrics["left"]["success"] and metrics["right"]["success"]), "steps": len(rollout["left"]["q"]), "joint_tolerance_rad": 0.15, "tcp_tolerance_m": 0.10, "orientation_tolerance_rad": 0.25}
    write_row(ROOT / "results" / "scene_metrics.csv", scene_metrics)
    report = {"left": metrics["left"], "right": metrics["right"], "scene": scene_metrics}
    (ROOT / "results" / "closed_loop_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not scene_metrics["dual_success"]:
        raise RuntimeError("Closed-loop smoke gate failed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
