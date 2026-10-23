from __future__ import annotations

import time

import numpy as np
import torch

from src.data.dagger_buffer import ActionHistory
from src.geometry.periodic import periodic_delta
from src.geometry.rotations import geodesic_distance, pose_error6, rotation_6d_to_matrix
from src.evaluation.collision_boundary import boundary_tangent_action


def _tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, *np.asarray(value).shape)


def build_observation(scene, side, q_start, q_goal, x_start, x_goal, history):
    snapshot = scene.state.snapshot(side)
    history_action, history_mask = history.padded()
    x_current = snapshot["tcp_pose9_local"]
    x_goal_tensor = _tensor(x_goal, scene.device)
    delta_x = pose_error6(x_current[:, :3], rotation_6d_to_matrix(x_current[:, 3:]), x_goal_tensor[:, :3], rotation_6d_to_matrix(x_goal_tensor[:, 3:]))
    return {
        "q_start": _tensor(q_start, scene.device),
        "q_current": snapshot["q"],
        "q_goal": _tensor(q_goal, scene.device),
        "delta_q_goal": periodic_delta(_tensor(q_goal, scene.device), snapshot["q"]),
        "x_start": _tensor(x_start, scene.device),
        "x_current": x_current,
        "x_goal": x_goal_tensor,
        "delta_x_goal": delta_x,
        "other_arm_geometry": snapshot["other_arm_geometry_local"],
        "history_action": _tensor(history_action, scene.device),
        "history_mask": torch.as_tensor(history_mask, dtype=torch.bool, device=scene.device).unsqueeze(0),
    }


@torch.no_grad()
def single_arm_closed_loop_rollout(
    scene,
    model,
    task,
    side="left",
    max_steps=180,
    joint_tolerance=0.15,
    tcp_tolerance=0.10,
    orientation_tolerance=0.25,
    collision_avoidance=False,
    collision_threshold=0.5,
):
    """Roll out one controller while the other arm remains at its task start pose.

    Only collisions involving the evaluated arm can terminate the rollout.  This
    prevents a stationary other-arm self-contact from censoring single-arm
    performance while retaining the original scene geometry and observations.
    """
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    other = "right" if side == "left" else "left"
    scene.teleport(
        _tensor(task["left"]["q_start"], scene.device)[0],
        _tensor(task["right"]["q_start"], scene.device)[0],
        settle_steps=4,
    )
    history = ActionHistory(model.config.history_length)
    trace = {"q": [], "tcp_position": [], "latency_ms": [], "self_collisions": 0, "inter_collisions": 0, "environment_collisions": 0, "collision_corrections": 0, "proposed_collision_risk": [], "executed_collision_risk": []}
    zero_delta = torch.zeros((1, 8), dtype=torch.float32, device=scene.device)
    spec = task[side]
    for _ in range(max_steps):
        observation = build_observation(
            scene, side, spec["q_start"], spec["q_goal"], spec["x_start"], spec["x_goal"], history
        )
        start = time.perf_counter()
        output = model(observation)
        action = output["action"]
        if collision_avoidance:
            action, correction = boundary_tangent_action(
                model,
                output["control_hidden"],
                observation["q_current"],
                action,
                threshold=collision_threshold,
            )
            trace["collision_corrections"] += int(correction.corrected)
            trace["proposed_collision_risk"].append(correction.proposed_risk)
            trace["executed_collision_risk"].append(correction.corrected_risk)
        if str(scene.device).startswith("cuda"):
            torch.cuda.synchronize(scene.device)
        trace["latency_ms"].append((time.perf_counter() - start) * 1000.0)
        deltas = {side: action, other: zero_delta}
        result = scene.step_deltas(deltas["left"], deltas["right"])
        history.append(result[side]["actual_delta"][0].detach().cpu().numpy())
        snapshot = scene.state.snapshot(side)
        trace["q"].append(snapshot["q"][0].cpu().numpy())
        trace["tcp_position"].append(snapshot["tcp_pose9_local"][0, :3].cpu().numpy())
        trace["self_collisions"] += int(result["labels"][f"{side}_self"])
        trace["inter_collisions"] += int(result["labels"]["inter_arm"])
        trace["environment_collisions"] += int(result["labels"][f"{side}_environment"])
        if trace["self_collisions"] or trace["inter_collisions"] or trace["environment_collisions"] or result[side]["emergency_stop"]:
            break
        joint_ok = torch.max(
            torch.abs(periodic_delta(_tensor(spec["q_goal"], scene.device), scene.state.q(side)))
        ).item() < joint_tolerance
        current_pose = scene.state.tcp_pose9_local(side)
        goal_pose = _tensor(spec["x_goal"], scene.device)
        position_ok = torch.linalg.vector_norm(goal_pose[:, :3] - current_pose[:, :3]).item() < tcp_tolerance
        orientation_ok = geodesic_distance(
            rotation_6d_to_matrix(current_pose[:, 3:]), rotation_6d_to_matrix(goal_pose[:, 3:])
        ).item() < orientation_tolerance
        if joint_ok and position_ok and orientation_ok:
            break
    current = scene.state.tcp_pose9_local(side)
    goal = _tensor(spec["x_goal"], scene.device)
    joint_error = periodic_delta(_tensor(spec["q_goal"], scene.device), scene.state.q(side))[0]
    position_error = torch.linalg.vector_norm(goal[:, :3] - current[:, :3]).item()
    orientation_error = geodesic_distance(
        rotation_6d_to_matrix(current[:, 3:]), rotation_6d_to_matrix(goal[:, 3:])
    ).item()
    success = bool(
        torch.max(torch.abs(joint_error)).item() < joint_tolerance
        and position_error < tcp_tolerance
        and orientation_error < orientation_tolerance
        and trace["self_collisions"] == 0
        and trace["inter_collisions"] == 0
        and trace["environment_collisions"] == 0
    )
    return {
        "joint_error": joint_error.cpu().numpy(),
        "position_error": position_error,
        "orientation_error": orientation_error,
        "success": success,
        **trace,
    }


@torch.no_grad()
def dual_closed_loop_rollout(scene, models, task, max_steps=180, joint_tolerance=0.15, tcp_tolerance=0.10, orientation_tolerance=0.25):
    scene.teleport(_tensor(task["left"]["q_start"], scene.device)[0], _tensor(task["right"]["q_start"], scene.device)[0], settle_steps=4)
    histories = {side: ActionHistory(models[side].config.history_length) for side in ("left", "right")}
    traces = {side: {"q": [], "tcp_position": [], "latency_ms": [], "self_collisions": 0, "inter_collisions": 0, "environment_collisions": 0} for side in ("left", "right")}
    for _ in range(max_steps):
        observations, actions = {}, {}
        for side in ("left", "right"):
            spec = task[side]
            observations[side] = build_observation(scene, side, spec["q_start"], spec["q_goal"], spec["x_start"], spec["x_goal"], histories[side])
            start = time.perf_counter()
            actions[side] = models[side](observations[side])["action"]
            if str(scene.device).startswith("cuda"):
                torch.cuda.synchronize(scene.device)
            traces[side]["latency_ms"].append((time.perf_counter() - start) * 1000.0)
        result = scene.step_deltas(actions["left"], actions["right"])
        labels = result["labels"]
        for side in ("left", "right"):
            histories[side].append(result[side]["actual_delta"][0].detach().cpu().numpy())
            snapshot = scene.state.snapshot(side)
            traces[side]["q"].append(snapshot["q"][0].cpu().numpy())
            traces[side]["tcp_position"].append(snapshot["tcp_pose9_local"][0, :3].cpu().numpy())
            traces[side]["self_collisions"] += int(labels[f"{side}_self"])
            traces[side]["inter_collisions"] += int(labels["inter_arm"])
            traces[side]["environment_collisions"] += int(labels[f"{side}_environment"])
        if result["safety_stop"]:
            break
        done = []
        for side in ("left", "right"):
            joint_ok = torch.max(torch.abs(periodic_delta(_tensor(task[side]["q_goal"], scene.device), scene.state.q(side)))).item() < joint_tolerance
            current_pose = scene.state.tcp_pose9_local(side)
            goal_pose = _tensor(task[side]["x_goal"], scene.device)
            position_ok = torch.linalg.vector_norm(goal_pose[:, :3] - current_pose[:, :3]).item() < tcp_tolerance
            orientation_ok = geodesic_distance(rotation_6d_to_matrix(current_pose[:, 3:]), rotation_6d_to_matrix(goal_pose[:, 3:])).item() < orientation_tolerance
            done.append(joint_ok and position_ok and orientation_ok)
        if all(done):
            break
    final = {}
    for side in ("left", "right"):
        current = scene.state.tcp_pose9_local(side)
        goal = _tensor(task[side]["x_goal"], scene.device)
        joint_error = periodic_delta(_tensor(task[side]["q_goal"], scene.device), scene.state.q(side))[0]
        position_error = torch.linalg.vector_norm(goal[:, :3] - current[:, :3]).item()
        orientation_error = geodesic_distance(rotation_6d_to_matrix(current[:, 3:]), rotation_6d_to_matrix(goal[:, 3:])).item()
        success = bool(torch.max(torch.abs(joint_error)).item() < joint_tolerance and position_error < tcp_tolerance and orientation_error < orientation_tolerance and traces[side]["self_collisions"] == 0 and traces[side]["inter_collisions"] == 0 and traces[side]["environment_collisions"] == 0)
        final[side] = {"joint_error": joint_error.cpu().numpy(), "position_error": position_error, "orientation_error": orientation_error, "success": success, **traces[side]}
    return final
