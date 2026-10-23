"""Hard gate: independently overfit 100-1000 live-simulation Stage-1 samples per arm."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from src.data.dataset import ArmTrajectoryDataset
from src.models.trajectory_model import TrajectoryModel, TrajectoryModelConfig, assert_models_independent
from src.training.losses import LossWeights
from src.training.trainer import append_csv, evaluate_action_loss, lock_seed, make_loader, save_checkpoint, to_device, train_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--target-loss", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets" / "tiny")
    args = parser.parse_args()
    if not 100 <= args.samples <= 1000:
        raise ValueError("tiny overfit must use 100-1000 samples")
    lock_seed(args.seed)
    datasets = {side: ArmTrajectoryDataset(args.dataset_root / side, limit=args.samples) for side in ("left", "right")}
    loaders = {side: make_loader(datasets[side], args.batch_size, shuffle=True, seed=args.seed + i) for i, side in enumerate(("left", "right"))}
    eval_loaders = {side: make_loader(datasets[side], args.batch_size, shuffle=False, seed=args.seed) for side in ("left", "right")}
    config = TrajectoryModelConfig(history_length=64)
    models = {side: TrajectoryModel(config).to(args.device) for side in ("left", "right")}
    assert_models_independent(models["left"], models["right"])
    optimizers = {side: torch.optim.AdamW(models[side].parameters(), lr=args.lr, weight_decay=1e-6) for side in ("left", "right")}
    if optimizers["left"] is optimizers["right"]:
        raise AssertionError("Optimizers must be independent")
    weights = LossWeights(action=1.0, velocity=0.0, acceleration=0.0, jerk=0.0, self_collision=0.0, inter_collision=0.0)
    best = {"left": float("inf"), "right": float("inf")}
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        running = {side: [] for side in ("left", "right")}
        # Same GPU utilization window, but each side has its own forward/backward/step.
        for batch_left, batch_right in zip(loaders["left"], loaders["right"]):
            loss_left = train_step(models["left"], optimizers["left"], to_device(batch_left, args.device), weights)
            loss_right = train_step(models["right"], optimizers["right"], to_device(batch_right, args.device), weights)
            running["left"].append(loss_left)
            running["right"].append(loss_right)
        evaluation = {side: evaluate_action_loss(models[side], eval_loaders[side], args.device) for side in ("left", "right")}
        for side in ("left", "right"):
            row = {"epoch": epoch, "action_loss": evaluation[side], "total_loss": sum(item["total_loss"] for item in running[side]) / len(running[side])}
            append_csv(ROOT / "logs" / f"{side}_tiny_overfit.csv", row)
            if evaluation[side] < best[side]:
                best[side] = evaluation[side]
                save_checkpoint(ROOT / "checkpoints" / side / "tiny_overfit_best.pt", models[side], optimizers[side], epoch, row, config.__dict__)
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} left={evaluation['left']:.8f} right={evaluation['right']:.8f}", flush=True)
        if max(evaluation.values()) <= args.target_loss:
            break
    for side in ("left", "right"):
        save_checkpoint(ROOT / "checkpoints" / side / "tiny_overfit_last.pt", models[side], optimizers[side], epoch, {"action_loss": evaluation[side]}, config.__dict__)
    report = {
        "status": "PASS" if max(evaluation.values()) <= args.target_loss else "FAIL",
        "samples_per_side": args.samples,
        "epochs_completed": epoch,
        "target_action_loss": args.target_loss,
        "left_action_loss": evaluation["left"],
        "right_action_loss": evaluation["right"],
        "elapsed_seconds": time.perf_counter() - start_time,
        "models_independent": True,
        "optimizers_independent": True,
    }
    path = ROOT / "results" / "tiny_overfit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Tiny overfit gate failed; do not start formal training")


if __name__ == "__main__":
    main()
