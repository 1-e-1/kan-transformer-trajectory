"""Stage-2 H64 fine-tuning or H128 warm-start for two independent policies."""

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
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import ArmTrajectoryDataset
from src.data.splits import episode_group_split, extend_episode_group_split
from src.data.normalization import compute_normalization
from src.evaluation.metrics import binary_classification_metrics
from src.models.trajectory_model import TrajectoryModel, TrajectoryModelConfig, assert_models_independent
from src.training.losses import LossWeights, supervised_loss, teacher_preservation_loss
from src.training.trainer import append_csv, evaluate_supervised, lock_seed, make_loader, save_checkpoint, to_device, train_step


def positive_weight(values):
    positives = float(np.sum(values > 0.5))
    negatives = float(len(values) - positives)
    return negatives / positives if positives else 1.0


def teacher_view(batch, history_length):
    view = dict(batch)
    view["history_action"] = batch["history_action"][:, -history_length:]
    view["history_mask"] = batch["history_mask"][:, -history_length:]
    return view


@torch.no_grad()
def evaluate_teacher_alignment(
    model, teacher, loader, device, weights, minimum_weight, include_token_loss
):
    model.eval()
    total_action, total_token, batches = 0.0, 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        teacher_batch = teacher_view(batch, teacher.config.history_length)
        output = model(batch)
        teacher_action = teacher(teacher_batch)["action"]
        total_action += float(
            teacher_preservation_loss(
                output["action"], teacher_action, batch,
                minimum_weight=minimum_weight,
                terminal_joint_scale=weights.terminal_joint_scale,
                terminal_tcp_scale=weights.terminal_tcp_scale,
            ).cpu()
        )
        if include_token_loss:
            student_tokens, _ = model.tokenizer(model.normalizer(batch))
            teacher_tokens, _ = teacher.tokenizer(teacher.normalizer(teacher_batch))
            total_token += float(
                F.smooth_l1_loss(student_tokens[:, 1:10], teacher_tokens[:, 1:10]).cpu()
            )
        batches += 1
    model.train()
    return {
        "teacher_preservation_loss": total_action / max(1, batches),
        "teacher_semantic_token_loss": total_token / max(1, batches),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage2_h64", "dagger_h64", "h128"), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-checkpoint", help="Override the mode's warm-start checkpoint filename")
    parser.add_argument(
        "--teacher-checkpoint",
        help="Optional frozen teacher checkpoint filename, independent of the student warm start",
    )
    parser.add_argument("--output-name", help="Checkpoint/result stem; preserves existing baselines when overridden")
    parser.add_argument("--normalization-source", choices=("recompute", "checkpoint"), default="recompute")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / 'datasets')
    parser.add_argument("--split-manifest", type=Path, default=ROOT / 'results/episode_split_manifest.json')
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--terminal-error-boost", type=float, default=0.0)
    parser.add_argument("--terminal-joint-scale", type=float, default=0.25)
    parser.add_argument("--terminal-tcp-scale", type=float, default=0.15)
    parser.add_argument("--collision-loss-weight", type=float, default=0.01)
    parser.add_argument("--freeze-collision-head", action="store_true")
    parser.add_argument("--semantic-encoder", choices=("kan", "mlp"), default="kan")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--train-semantic-encoder-only", action="store_true",
        help="Freeze Transformer/history/action modules and train only the five semantic encoders",
    )
    parser.add_argument(
        "--teacher-loss-weight", type=float, default=0.0,
        help="Behavior-preservation weight for a frozen copy of the warm-start checkpoint",
    )
    parser.add_argument(
        "--teacher-min-weight", type=float, default=0.1,
        help="Residual teacher weight at the goal; far-from-goal weight is always 1",
    )
    parser.add_argument(
        "--teacher-token-loss-weight", type=float, default=0.0,
        help="Distill the nine KAN task/environment tokens into a replacement encoder",
    )
    args = parser.parse_args()
    if args.teacher_loss_weight < 0.0:
        raise ValueError("--teacher-loss-weight must be non-negative")
    if args.teacher_token_loss_weight < 0.0:
        raise ValueError("--teacher-token-loss-weight must be non-negative")
    if not 0.0 <= args.teacher_min_weight <= 1.0:
        raise ValueError("--teacher-min-weight must be in [0, 1]")
    lock_seed(args.seed)

    history_length = 128 if args.mode == "h128" else 64
    input_name = args.input_checkpoint or {"stage2_h64": "bc_h64_best.pt", "dagger_h64": "stage2_h64_best.pt", "h128": "dagger_h64_best.pt"}[args.mode]
    teacher_name = args.teacher_checkpoint or input_name
    output_name = args.output_name or args.mode.replace("_h64", "_h64")
    if Path(input_name).name != input_name or not input_name.endswith(".pt"):
        raise ValueError("--input-checkpoint must be a plain .pt filename")
    if Path(teacher_name).name != teacher_name or not teacher_name.endswith(".pt"):
        raise ValueError("--teacher-checkpoint must be a plain .pt filename")
    if Path(output_name).name != output_name or not output_name.replace("_", "").isalnum():
        raise ValueError("--output-name must be a plain alphanumeric/underscore stem")
    sides = ("left", "right") if args.side == "both" else (args.side,)
    datasets = {side: ArmTrajectoryDataset(args.dataset_root / side, history_length=history_length) for side in sides}
    reference_side = sides[0]
    sample_count = len(datasets[reference_side])
    if len(sides) == 2 and len(datasets["right"]) != sample_count:
        raise RuntimeError("Left/right simultaneous datasets must have equal sample counts")
    if len(sides) == 2 and not np.array_equal(datasets["left"].group_ids, datasets["right"].group_ids):
        raise RuntimeError("Left/right episode group alignment mismatch")
    manifest_path = args.split_manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))["groups"]
        split_indices, split_groups = extend_episode_group_split(datasets[reference_side].group_ids, existing, seed=args.seed)
    else:
        split_indices, split_groups = episode_group_split(datasets[reference_side].group_ids, seed=args.seed)
    train_indices = split_indices["train"]
    eval_indices = split_indices["validation"]
    loaders = {}
    for side_index, side in enumerate(sides):
        loaders[side] = {
            "train": make_loader(Subset(datasets[side], train_indices), args.batch_size, True, args.seed + side_index),
            "eval": make_loader(Subset(datasets[side], eval_indices), args.batch_size, False, args.seed),
            "test": make_loader(Subset(datasets[side], split_indices["test"]), args.batch_size, False, args.seed),
        }

    models, optimizers, writers, configs, teachers = {}, {}, {}, {}, {}
    computed_normalization = {side: compute_normalization(datasets[side], train_indices) for side in sides}
    normalization = {}
    run_tag = f"{output_name}_g{len(np.unique(datasets[reference_side].group_ids))}_seed{args.seed}"
    for side in sides:
        checkpoint = torch.load(ROOT / "checkpoints" / side / input_name, map_location=args.device, weights_only=False)
        normalization[side] = checkpoint["config"].get("normalization") if args.normalization_source == "checkpoint" else computed_normalization[side]
        if normalization[side] is None:
            raise RuntimeError(f"Checkpoint {input_name} has no normalization statistics for {side}")
        config = TrajectoryModelConfig(
            history_length=history_length,
            semantic_encoder=args.semantic_encoder,
            mlp_hidden_dim=args.mlp_hidden_dim,
            normalization=normalization[side],
        )
        configs[side] = config
        model = TrajectoryModel(config).to(args.device)
        checkpoint_encoder = checkpoint["config"].get("semantic_encoder", "kan")
        model.load_compatible_state_dict(
            checkpoint["model"],
            allow_semantic_encoder_reinit=args.semantic_encoder != checkpoint_encoder,
        )
        if args.train_semantic_encoder_only:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for module in (
                model.tokenizer.joint,
                model.tokenizer.pose,
                model.tokenizer.joint_error,
                model.tokenizer.pose_error,
                model.tokenizer.other_arm,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        if args.freeze_collision_head and model.collision_head is not None:
            for parameter in model.collision_head.parameters():
                parameter.requires_grad_(False)
        models[side] = model
        optimizers[side] = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.lr,
            weight_decay=1e-6,
        )
        writers[side] = SummaryWriter(ROOT / "logs" / "tensorboard" / side / run_tag)
        if args.teacher_loss_weight > 0.0 or args.teacher_token_loss_weight > 0.0:
            teacher_checkpoint = torch.load(
                ROOT / "checkpoints" / side / teacher_name,
                map_location=args.device,
                weights_only=False,
            )
            teacher_config = TrajectoryModelConfig(**teacher_checkpoint["config"])
            teacher = TrajectoryModel(teacher_config).to(args.device)
            teacher.load_compatible_state_dict(teacher_checkpoint["model"])
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            teachers[side] = teacher
    if len(sides) == 2:
        assert_models_independent(models["left"], models["right"])

    # Keep auxiliary/smoothness gradients below the one-step expert objective.
    # Previous weights made collision BCE dominate action loss on rare positives.
    weights = LossWeights(
        action=args.action_loss_weight,
        terminal_error_boost=args.terminal_error_boost,
        terminal_joint_scale=args.terminal_joint_scale,
        terminal_tcp_scale=args.terminal_tcp_scale,
        velocity=2e-4,
        acceleration=2e-5,
        jerk=2e-6,
        self_collision=args.collision_loss_weight,
        inter_collision=args.collision_loss_weight,
    )
    pos_weights = {
        side: {
            "self": positive_weight(datasets[side].arrays["self_collision_label"][train_indices]),
            "inter": positive_weight(datasets[side].arrays["inter_collision_label"][train_indices]),
        }
        for side in sides
    }
    best = {side: float("inf") for side in sides}
    for epoch in range(1, args.epochs + 1):
        iterators = [loaders[side]["train"] for side in sides]
        for batches in zip(*iterators):
            for side, batch in zip(sides, batches):
                device_batch = to_device(batch, args.device)
                if args.teacher_loss_weight <= 0.0 and args.teacher_token_loss_weight <= 0.0:
                    metrics = train_step(
                        models[side], optimizers[side], device_batch, weights,
                        self_pos_weight=pos_weights[side]["self"], inter_pos_weight=pos_weights[side]["inter"],
                    )
                else:
                    models[side].train()
                    optimizers[side].zero_grad(set_to_none=True)
                    output = models[side](device_batch)
                    losses = supervised_loss(
                        output, device_batch, weights,
                        self_pos_weight=pos_weights[side]["self"],
                        inter_pos_weight=pos_weights[side]["inter"],
                    )
                    teacher_batch = teacher_view(
                        device_batch, teachers[side].config.history_length
                    )
                    with torch.no_grad():
                        teacher_action = teachers[side](teacher_batch)["action"]
                    teacher_loss = teacher_preservation_loss(
                        output["action"], teacher_action, device_batch,
                        minimum_weight=args.teacher_min_weight,
                        terminal_joint_scale=args.terminal_joint_scale,
                        terminal_tcp_scale=args.terminal_tcp_scale,
                    )
                    token_loss = output["action"].new_zeros(())
                    if args.teacher_token_loss_weight > 0.0:
                        student_tokens, _ = models[side].tokenizer(
                            models[side].normalizer(device_batch)
                        )
                        with torch.no_grad():
                            teacher_tokens, _ = teachers[side].tokenizer(
                                teachers[side].normalizer(teacher_batch)
                            )
                        token_loss = F.smooth_l1_loss(
                            student_tokens[:, 1:10], teacher_tokens[:, 1:10]
                        )
                    optimized_total = (
                        losses["total_loss"]
                        + args.teacher_loss_weight * teacher_loss
                        + args.teacher_token_loss_weight * token_loss
                    )
                    optimized_total.backward()
                    torch.nn.utils.clip_grad_norm_(models[side].parameters(), 1.0)
                    optimizers[side].step()
                    metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
                    metrics["teacher_preservation_loss"] = float(teacher_loss.detach().cpu())
                    metrics["teacher_semantic_token_loss"] = float(token_loss.detach().cpu())
                    metrics["optimized_total_loss"] = float(optimized_total.detach().cpu())
                for key, value in metrics.items():
                    writers[side].add_scalar(f"train/{key}", value, epoch)
        for side in sides:
            losses, predictions = evaluate_supervised(models[side], loaders[side]["eval"], args.device, weights, pos_weights[side]["self"], pos_weights[side]["inter"])
            collision = {name: binary_classification_metrics(*predictions[name]) for name in ("self", "inter")}
            alignment = {"teacher_preservation_loss": 0.0, "teacher_semantic_token_loss": 0.0}
            if side in teachers:
                alignment = evaluate_teacher_alignment(
                    models[side], teachers[side], loaders[side]["eval"], args.device,
                    weights, args.teacher_min_weight,
                    args.teacher_token_loss_weight > 0.0,
                )
            selection_loss = (
                losses["total_loss"]
                + args.teacher_loss_weight * alignment["teacher_preservation_loss"]
                + args.teacher_token_loss_weight * alignment["teacher_semantic_token_loss"]
            )
            row = {"epoch": epoch, **losses, **alignment, "selection_loss": selection_loss, "self_f1": collision["self"]["f1"], "inter_f1": collision["inter"]["f1"]}
            append_csv(ROOT / "logs" / f"{side}_{run_tag}_train.csv", row)
            for key, value in losses.items():
                writers[side].add_scalar(f"eval/{key}", value, epoch)
            if selection_loss < best[side]:
                best[side] = selection_loss
                save_checkpoint(ROOT / "checkpoints" / side / f"{output_name}_best.pt", models[side], optimizers[side], epoch, row, configs[side].__dict__)
            save_checkpoint(ROOT / "checkpoints" / side / f"{output_name}_last.pt", models[side], optimizers[side], epoch, row, configs[side].__dict__)
        summary = " ".join(f"{side}={best[side]:.8f}" for side in sides)
        print(f"epoch={epoch} {summary}", flush=True)
    for writer in writers.values():
        writer.close()
    test_metrics = {}
    for side in sides:
        best_checkpoint = torch.load(ROOT / "checkpoints" / side / f"{output_name}_best.pt", map_location=args.device, weights_only=False)
        models[side].load_compatible_state_dict(best_checkpoint["model"])
        test_losses, test_predictions = evaluate_supervised(models[side], loaders[side]["test"], args.device, weights, pos_weights[side]["self"], pos_weights[side]["inter"])
        test_metrics[side] = {
            "losses": test_losses,
            "collision_classification": {name: binary_classification_metrics(*test_predictions[name]) for name in ("self", "inter")},
        }
    split_manifest = {
        "seed": args.seed, "groups": split_groups,
        "sample_counts": {name: len(indices) for name, indices in split_indices.items()},
    }
    manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    normalization_path = ROOT / "results" / f"{output_name}_normalization_stats.json"
    normalization_path.write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    report = {"mode": args.mode, "sides": list(sides), "run_tag": run_tag, "history_length": history_length, "semantic_encoder": args.semantic_encoder, "mlp_hidden_dim": args.mlp_hidden_dim if args.semantic_encoder == "mlp" else None, "train_semantic_encoder_only": args.train_semantic_encoder_only, "trainable_parameters": {side: sum(parameter.numel() for parameter in models[side].parameters() if parameter.requires_grad) for side in sides}, "total_parameters": {side: sum(parameter.numel() for parameter in models[side].parameters()) for side in sides}, "warm_start": input_name, "normalization_source": args.normalization_source, "epochs": args.epochs, "train_samples": len(train_indices), "validation_samples": len(eval_indices), "test_samples": len(split_indices["test"]), "train_groups": len(split_groups["train"]), "validation_groups": len(split_groups["validation"]), "test_groups": len(split_groups["test"]), "best_selection_loss": best, "test_metrics": test_metrics, "loss_weights": weights.__dict__, "positive_weights": pos_weights, "collision_head_frozen": args.freeze_collision_head, "teacher_preservation": {"enabled": args.teacher_loss_weight > 0.0, "loss_weight": args.teacher_loss_weight, "minimum_terminal_weight": args.teacher_min_weight, "semantic_token_loss_weight": args.teacher_token_loss_weight, "checkpoint": teacher_name if args.teacher_loss_weight > 0.0 or args.teacher_token_loss_weight > 0.0 else None}}
    (ROOT / "results" / f"{output_name}_training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
