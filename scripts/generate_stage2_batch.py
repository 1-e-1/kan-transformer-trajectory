"""Resumable multi-seed Stage-2 generation in one Isaac Sim session."""

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
parser.add_argument("--seed-start", type=int, default=3000)
parser.add_argument("--seed-count", type=int, default=45)
parser.add_argument("--seeds", nargs="+", type=int, help="Explicit seed list; overrides --seed-start/--seed-count")
parser.add_argument("--episodes-per-seed", type=int, default=1)
parser.add_argument("--max-attempts", type=int, default=600)
parser.add_argument("--planner-iterations", type=int, default=400)
parser.add_argument("--max-planner-failures", type=int, default=5)
parser.add_argument(
    "--hard-only", action=argparse.BooleanOptionalAction, default=True,
    help="Require a collision on the direct path so every new episode needs an RRT detour (default: true)",
)
parser.add_argument("--pool-noise-std", type=float, default=0.50)
parser.add_argument("--report-name", default="stage2_batch_generation.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from src.data.stage2_generator import Stage2DataGenerator
from src.env.dual_arm_scene import DualArmScene


def load_safe_endpoint_pool():
    import numpy as np

    pools = {"left": [], "right": []}
    for side in ("left", "right"):
        for path in sorted((ROOT / "datasets" / side).glob("stage1*.npz")):
            with np.load(path) as data:
                # One row per episode endpoint is enough and avoids timestep bias.
                first = np.r_[True, data["episode_id"][1:] != data["episode_id"][:-1]]
                pools[side].extend((data["q_start"][first], data["q_goal"][first]))
        pools[side] = np.concatenate(pools[side], axis=0)
    if len(pools["left"]) != len(pools["right"]):
        raise RuntimeError("Safe endpoint pool alignment mismatch")
    return pools


def main():
    report_path = ROOT / "results" / args.report_name
    if report_path.name != args.report_name or report_path.suffix.lower() != ".json":
        raise ValueError("--report-name must be a plain .json filename")
    scene = DualArmScene(ROOT / "finally.usd", device=args.device, execution_mode="kinematic")
    endpoint_pool = load_safe_endpoint_pool()
    reports, skipped, failures = [], [], []
    seed_values = args.seeds if args.seeds else range(args.seed_start, args.seed_start + args.seed_count)
    for seed in seed_values:
        expected = ROOT / "datasets" / "left" / f"stage2_rrt_seed{seed}_ep{args.episodes_per_seed}.npz"
        if expected.exists():
            skipped.append(seed)
            continue
        try:
            report = Stage2DataGenerator(
                scene, ROOT / "datasets", seed=seed,
                max_sampling_attempts=args.max_attempts,
                planner_iterations=args.planner_iterations,
                max_planner_failures=args.max_planner_failures,
                require_direct_collision=args.hard_only,
                safe_endpoint_pool=endpoint_pool,
                pool_noise_std=args.pool_noise_std,
            ).generate(args.episodes_per_seed)
            reports.append({"seed": seed, **report})
            print(f"stage2_batch_completed={len(reports)} seed={seed}", flush=True)
        except Exception as error:
            failures.append({"seed": seed, "error": repr(error)})
            print(f"stage2_batch_failed seed={seed} error={error!r}", flush=True)
        summary = {
            "seed_start": args.seed_start, "seed_count": args.seed_count,
            "requested_seeds": list(seed_values) if args.seeds else None,
            "episodes_per_seed": args.episodes_per_seed, "hard_only": args.hard_only,
            "execution_mode": scene.execution_mode,
            "completed": reports, "skipped_seeds": skipped, "failures": failures,
        }
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"completed_seeds": len(reports), "skipped": len(skipped), "failures": len(failures)}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
