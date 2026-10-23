"""Retrain the left auxiliary collision head on equal PhysX boundary classes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.data.dataset import ArmTrajectoryDataset
from src.models.trajectory_model import TrajectoryModel, TrajectoryModelConfig
from src.training.losses import LossWeights
from src.training.trainer import append_csv, evaluate_supervised, lock_seed, save_checkpoint, to_device, train_step


def stratified_episode_split(dataset: ArmTrajectoryDataset, seed: int) -> dict[str, list[int]]:
    labels = np.stack(
        (dataset.arrays["self_collision_label"], dataset.arrays["inter_collision_label"]), axis=1
    )
    # Full provenance string prevents equal episode numbers in different NPZ
    # sources from leaking across splits.
    episode_names = dataset.group_ids
    rng = np.random.default_rng(seed)
    result = {name: [] for name in ("train", "validation", "test")}
    for target in range(2):
        positive_indices = np.flatnonzero(labels[:, target] > 0.5)
        episodes = np.unique(episode_names[positive_indices])
        rng.shuffle(episodes)
        n_train = int(round(0.70 * len(episodes)))
        n_validation = int(round(0.15 * len(episodes)))
        split_episodes = {
            "train": episodes[:n_train],
            "validation": episodes[n_train : n_train + n_validation],
            "test": episodes[n_train + n_validation :],
        }
        for split, names in split_episodes.items():
            for name in names:
                result[split].extend(np.flatnonzero(episode_names == name).tolist())
    for split in result:
        result[split] = sorted(set(result[split]))
    return result


def binary_metrics(probability: np.ndarray, label: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probability >= threshold
    actual = label >= 0.5
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
    }


def choose_shared_threshold(predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = [
            binary_metrics(1.0 / (1.0 + np.exp(-predictions[name][0])), predictions[name][1], threshold)
            for name in ("self", "inter")
        ]
        # Equal arithmetic mean: neither collision type gets a larger weight.
        score = float(np.mean([metric["f1"] for metric in metrics]))
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold


def evaluate_metrics(model, loader, device, threshold):
    weights = LossWeights(action=0.0, self_collision=1.0, inter_collision=1.0)
    losses, predictions = evaluate_supervised(
        model, loader, device, weights, self_pos_weight=3.0, inter_pos_weight=3.0
    )
    metrics = {}
    for model_name, report_name in (("self", "self_collision"), ("inter", "external_collision")):
        logits, labels = predictions[model_name]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        metrics[report_name] = binary_metrics(probabilities, labels, threshold)
    return losses, predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=18350)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline", default="dagger_h64_best.pt")
    args = parser.parse_args()
    lock_seed(args.seed)

    dataset = ArmTrajectoryDataset(ROOT / "datasets_safety" / "left", history_length=64)
    indices = stratified_episode_split(dataset, args.seed)
    train_labels = np.stack(
        (
            dataset.arrays["self_collision_label"][indices["train"]],
            dataset.arrays["inter_collision_label"][indices["train"]],
        ),
        axis=1,
    )
    categories = np.where(train_labels[:, 0] > 0.5, 1, np.where(train_labels[:, 1] > 0.5, 2, 0))
    category_mass = {0: 0.50, 1: 0.25, 2: 0.25}
    sample_weights = np.asarray(
        [category_mass[int(category)] / max(1, np.sum(categories == category)) for category in categories],
        dtype=np.float64,
    )
    train_sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=max(2048, len(indices["train"])),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loaders = {
        "train": DataLoader(
            Subset(dataset, indices["train"]), batch_size=args.batch_size,
            sampler=train_sampler, num_workers=0,
        ),
        "validation": DataLoader(
            Subset(dataset, indices["validation"]), batch_size=args.batch_size,
            shuffle=False, num_workers=0,
        ),
        "test": DataLoader(
            Subset(dataset, indices["test"]), batch_size=args.batch_size,
            shuffle=False, num_workers=0,
        ),
    }
    checkpoint = torch.load(
        ROOT / "checkpoints" / "left" / args.baseline,
        map_location=args.device,
        weights_only=False,
    )
    model = TrajectoryModel(TrajectoryModelConfig(**checkpoint["config"])).to(args.device)
    model.load_compatible_state_dict(checkpoint["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.collision_head.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(model.collision_head.parameters(), lr=args.lr, weight_decay=1e-5)
    weights = LossWeights(action=0.0, velocity=0.0, acceleration=0.0, jerk=0.0, self_collision=1.0, inter_collision=1.0)
    log_path = ROOT / "logs" / "left_collision_head_train.csv"
    if log_path.exists():
        log_path.unlink()
    output_path = ROOT / "checkpoints" / "left" / "left_collision_aware_h64_best.pt"
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            train_step(
                model,
                optimizer,
                to_device(batch, args.device),
                weights,
                self_pos_weight=3.0,
                inter_pos_weight=3.0,
            )
        validation_losses, _ = evaluate_supervised(
            model,
            loaders["validation"],
            args.device,
            weights,
            self_pos_weight=3.0,
            inter_pos_weight=3.0,
        )
        score = 0.5 * (
            validation_losses["self_collision_aux_loss"]
            + validation_losses["inter_collision_aux_loss"]
        )
        append_csv(
            log_path,
            {
                "epoch": epoch,
                "equal_weight_validation_loss": score,
                "self_validation_loss": validation_losses["self_collision_aux_loss"],
                "external_validation_loss": validation_losses["inter_collision_aux_loss"],
            },
        )
        if score < best:
            best = score
            save_checkpoint(
                output_path,
                model,
                optimizer,
                epoch,
                {"equal_weight_validation_loss": score},
                model.config.__dict__,
            )

    best_checkpoint = torch.load(output_path, map_location=args.device, weights_only=False)
    model.load_compatible_state_dict(best_checkpoint["model"])
    _, validation_predictions = evaluate_supervised(
        model, loaders["validation"], args.device, weights, self_pos_weight=3.0, inter_pos_weight=3.0
    )
    threshold = choose_shared_threshold(validation_predictions)
    validation_losses, _, validation_metrics = evaluate_metrics(
        model, loaders["validation"], args.device, threshold
    )
    test_losses, _, test_metrics = evaluate_metrics(model, loaders["test"], args.device, threshold)

    # The core and action head were frozen. Verify the retraining cannot alter
    # nominal policy actions for an identical observation batch.
    baseline = TrajectoryModel(TrajectoryModelConfig(**checkpoint["config"])).to(args.device)
    baseline.load_compatible_state_dict(checkpoint["model"])
    baseline.eval()
    model.eval()
    batch = to_device(next(iter(loaders["test"])), args.device)
    with torch.no_grad():
        action_difference = float(torch.max(torch.abs(model(batch)["action"] - baseline(batch)["action"])).cpu())

    # Store the validation-selected common threshold in checkpoint metadata.
    best_checkpoint["metrics"].update(
        {
            "collision_threshold": threshold,
            "validation": validation_metrics,
            "test": test_metrics,
            "max_nominal_action_difference_rad": action_difference,
        }
    )
    torch.save(best_checkpoint, output_path)
    report = {
        "checkpoint": str(output_path),
        "baseline": args.baseline,
        "seed": args.seed,
        "split_samples": {split: len(values) for split, values in indices.items()},
        "equal_loss_weights": {"self_collision": 1.0, "external_collision": 1.0},
        "shared_positive_weight": 3.0,
        "shared_collision_threshold": threshold,
        "validation_loss": validation_losses,
        "validation_metrics": validation_metrics,
        "test_loss": test_losses,
        "test_metrics": test_metrics,
        "trainable_parameters": [name for name, value in model.named_parameters() if value.requires_grad],
        "frozen_policy_core": True,
        "max_nominal_action_difference_rad": action_difference,
    }
    report_path = ROOT / "results" / "left_collision_head_training.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
