"""Collect DAgger corrections across randomized independent episode tasks."""

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
parser.add_argument("--tasks", type=int, default=20)
parser.add_argument("--steps-per-task", type=int, default=32)
parser.add_argument("--iteration", type=int, default=10)
parser.add_argument("--seed", type=int, default=442)
parser.add_argument("--planner-iterations", type=int, default=600)
parser.add_argument("--checkpoint", default="dagger_h64_best.pt")
parser.add_argument("--allow-repeat-tasks", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np
import torch

from src.data.dataset import ArmTrajectoryDataset
from src.env.dual_arm_scene import DualArmScene
from src.models.trajectory_model import TrajectoryModel, TrajectoryModelConfig, assert_models_independent
from src.training.dagger import DAggerCollector


def task_catalog():
    datasets = {side: ArmTrajectoryDataset([*sorted((ROOT / "datasets" / side).glob("stage1*.npz")), *sorted((ROOT / "datasets" / side).glob("stage2*.npz"))]) for side in ("left", "right")}
    if not np.array_equal(datasets["left"].group_ids, datasets["right"].group_ids):
        raise RuntimeError("Left/right task catalogs are not aligned")
    tasks = []
    for group in np.unique(datasets["left"].group_ids):
        index = int(np.flatnonzero(datasets["left"].group_ids == group)[0])
        task = {}
        for side in ("left", "right"):
            task[side] = {key: datasets[side].arrays[key][index] for key in ("q_start", "q_goal", "x_start", "x_goal")}
        tasks.append((group, task))
    return tasks


def main():
    models = {}
    for side in ("left", "right"):
        checkpoint = torch.load(ROOT / "checkpoints" / side / args.checkpoint, map_location=args.device, weights_only=False)
        models[side] = TrajectoryModel(TrajectoryModelConfig(**checkpoint["config"])).to(args.device)
        models[side].load_compatible_state_dict(checkpoint["model"])
        models[side].eval()
    assert_models_independent(models["left"], models["right"])
    scene = DualArmScene(ROOT / "finally.usd", device=args.device, joint_stiffness=4000.0, joint_damping=200.0, joint_effort_limit=500.0)
    manifest = json.loads((ROOT / "results" / "episode_split_manifest.json").read_text(encoding="utf-8"))
    train_groups = set(manifest["groups"]["train"])
    catalog = [(group, task) for group, task in task_catalog() if group in train_groups]
    if not args.allow_repeat_tasks:
        used_groups = set()
        for report_path in (ROOT / "results").glob("dagger_randomized_iter*.json"):
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            used_groups.update(item["task_group"] for item in previous.get("reports", []))
        catalog = [(group, task) for group, task in catalog if group not in used_groups]
    if not catalog:
        raise RuntimeError("No train-split task groups are available for DAgger")
    order = np.random.default_rng(args.seed).permutation(len(catalog))[: min(args.tasks, len(catalog))]
    reports = []
    for task_index, catalog_index in enumerate(order):
        group, task = catalog[int(catalog_index)]
        output_iteration = args.iteration * 10000 + task_index
        try:
            report = DAggerCollector(scene, models, ROOT / "datasets", args.seed + task_index, args.planner_iterations).collect(task, args.steps_per_task, output_iteration)
            reports.append({"task_group": group, **report})
            print(f"dagger_random_task={task_index + 1}/{len(order)} samples={report['samples_per_side']}", flush=True)
        except Exception as error:
            reports.append({"task_group": group, "error": repr(error)})
    summary = {"requested_tasks": args.tasks, "selected_tasks": len(order), "steps_per_task": args.steps_per_task, "reports": reports, "successful_tasks": sum("outputs" in report for report in reports)}
    (ROOT / "results" / f"dagger_randomized_iter{args.iteration}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"successful_tasks": summary["successful_tasks"], "selected_tasks": len(order)}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
