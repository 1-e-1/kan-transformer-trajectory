"""Train exactly one independent H64 arm policy (left or right)."""

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
from src.models.trajectory_model import TrajectoryModel, TrajectoryModelConfig
from src.training.losses import LossWeights
from src.training.trainer import append_csv, evaluate_action_loss, lock_seed, make_loader, save_checkpoint, to_device, train_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    lock_seed(args.seed)
    dataset = ArmTrajectoryDataset(ROOT / "datasets" / args.side, history_length=64)
    train_loader = make_loader(dataset, args.batch_size, True, args.seed)
    eval_loader = make_loader(dataset, args.batch_size, False, args.seed)
    config = TrajectoryModelConfig(history_length=64)
    model = TrajectoryModel(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    weights = LossWeights(action=1.0, velocity=0.0, acceleration=0.0, jerk=0.0, self_collision=0.0, inter_collision=0.0)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            train_step(model, optimizer, to_device(batch, args.device), weights)
        loss = evaluate_action_loss(model, eval_loader, args.device)
        append_csv(ROOT / "logs" / f"{args.side}_h64_single_train.csv", {"epoch": epoch, "action_loss": loss})
        if loss < best:
            best = loss
            save_checkpoint(ROOT / "checkpoints" / args.side / "bc_h64_best.pt", model, optimizer, epoch, {"action_loss": loss}, config.__dict__)
        save_checkpoint(ROOT / "checkpoints" / args.side / "bc_h64_last.pt", model, optimizer, epoch, {"action_loss": loss}, config.__dict__)
    print(json.dumps({"side": args.side, "samples": len(dataset), "best_action_loss": best}, indent=2))


if __name__ == "__main__":
    main()
