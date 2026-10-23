"""Train two independent H64 BC models with separate optimizers/checkpoints/statistics."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    lock_seed(args.seed)
    datasets = {side: ArmTrajectoryDataset(ROOT / "datasets" / side) for side in ("left", "right")}
    train_loaders = {side: make_loader(datasets[side], args.batch_size, True, args.seed + i) for i, side in enumerate(("left", "right"))}
    eval_loaders = {side: make_loader(datasets[side], args.batch_size, False, args.seed) for side in ("left", "right")}
    config = TrajectoryModelConfig(history_length=64)
    models = {side: TrajectoryModel(config).to(args.device) for side in ("left", "right")}
    assert_models_independent(models["left"], models["right"])
    optimizers = {side: torch.optim.AdamW(models[side].parameters(), lr=args.lr, weight_decay=1e-6) for side in ("left", "right")}
    weights = LossWeights(action=1.0, velocity=0.0, acceleration=0.0, jerk=0.0, self_collision=0.0, inter_collision=0.0)
    best = {side: float("inf") for side in ("left", "right")}
    for epoch in range(1, args.epochs + 1):
        for batch_left, batch_right in zip(train_loaders["left"], train_loaders["right"]):
            # Deliberately no combined loss/backward and no shared optimizer.
            train_step(models["left"], optimizers["left"], to_device(batch_left, args.device), weights)
            train_step(models["right"], optimizers["right"], to_device(batch_right, args.device), weights)
        for side in ("left", "right"):
            action_loss = evaluate_action_loss(models[side], eval_loaders[side], args.device)
            row = {"epoch": epoch, "action_loss": action_loss}
            append_csv(ROOT / "logs" / f"{side}_h64_train.csv", row)
            if action_loss < best[side]:
                best[side] = action_loss
                save_checkpoint(ROOT / "checkpoints" / side / "bc_h64_best.pt", models[side], optimizers[side], epoch, row, config.__dict__)
            save_checkpoint(ROOT / "checkpoints" / side / "bc_h64_last.pt", models[side], optimizers[side], epoch, row, config.__dict__)
        print(f"epoch={epoch} left={best['left']:.8f} right={best['right']:.8f}", flush=True)
    report = {"epochs": args.epochs, "samples": {side: len(datasets[side]) for side in datasets}, "best_action_loss": best}
    (ROOT / "results" / "h64_training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
