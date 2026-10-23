from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import LossWeights, supervised_loss


def lock_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_step(model, optimizer, batch, weights, gradient_clip_norm=1.0, self_pos_weight=1.0, inter_pos_weight=1.0):
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    losses = supervised_loss(output, batch, weights, self_pos_weight=self_pos_weight, inter_pos_weight=inter_pos_weight)
    losses["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def append_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_loader(dataset, batch_size=32, shuffle=True, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


@torch.no_grad()
def evaluate_action_loss(model, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        loss = torch.nn.functional.smooth_l1_loss(model(batch)["action"], batch["expert_action"], reduction="sum")
        total += float(loss.cpu())
        count += batch["expert_action"].numel()
    model.train()
    return total / max(1, count)


@torch.no_grad()
def evaluate_supervised(model, loader, device, weights, self_pos_weight=1.0, inter_pos_weight=1.0):
    model.eval()
    totals, batches = {}, 0
    logits = {"self": [], "inter": []}
    labels = {"self": [], "inter": []}
    for batch in loader:
        batch = to_device(batch, device)
        output = model(batch)
        losses = supervised_loss(output, batch, weights, self_pos_weight=self_pos_weight, inter_pos_weight=inter_pos_weight)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.cpu())
        for name in ("self", "inter"):
            logits[name].append(output[f"{name}_collision_logit"].cpu().numpy())
            labels[name].append(batch[f"{name}_collision_label"].cpu().numpy())
        batches += 1
    model.train()
    return {key: value / max(1, batches) for key, value in totals.items()}, {name: (np.concatenate(logits[name]), np.concatenate(labels[name])) for name in logits}


def save_checkpoint(path, model, optimizer, epoch, metrics, config) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )
