from __future__ import annotations

import numpy as np


def binary_classification_metrics(logits, labels, threshold=0.5):
    """Precision/recall/F1/AUROC without reporting misleading accuracy."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auroc = None
    if np.unique(labels).size == 2:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(labels, probabilities))
    return {"precision": precision, "recall": recall, "f1": f1, "auroc": auroc, "positive_count": int(labels.sum()), "sample_count": int(labels.size)}


def _rms_max(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    magnitudes = np.linalg.norm(values, axis=-1)
    return float(np.sqrt(np.mean(magnitudes**2))), float(np.max(magnitudes))


def trajectory_metrics(q, tcp_positions, dt, final_joint_error, final_tcp_position_error, final_tcp_orientation_error, success, self_collisions, inter_collisions, inference_latency_ms, environment_collisions=0):
    q = np.asarray(q)
    tcp_positions = np.asarray(tcp_positions)
    velocity = np.diff(q, axis=0) / dt if len(q) > 1 else np.empty((0, q.shape[-1]))
    acceleration = np.diff(velocity, axis=0) / dt if len(velocity) > 1 else np.empty((0, q.shape[-1]))
    jerk = np.diff(acceleration, axis=0) / dt if len(acceleration) > 1 else np.empty((0, q.shape[-1]))
    velocity_rms, velocity_max = _rms_max(velocity)
    acceleration_rms, acceleration_max = _rms_max(acceleration)
    jerk_rms, jerk_max = _rms_max(jerk)
    return {
        "tcp_position_error_m": float(final_tcp_position_error),
        "tcp_orientation_error_rad": float(final_tcp_orientation_error),
        "periodic_joint_error_rms_rad": float(np.sqrt(np.mean(np.asarray(final_joint_error) ** 2))),
        "periodic_joint_error_max_rad": float(np.max(np.abs(final_joint_error))),
        "success": int(bool(success)),
        "self_collision_rate": float(self_collisions / max(1, len(q) - 1)),
        "inter_arm_collision_rate": float(inter_collisions / max(1, len(q) - 1)),
        "environment_collision_rate": float(environment_collisions / max(1, len(q) - 1)),
        "trajectory_length_m": float(np.linalg.norm(np.diff(tcp_positions, axis=0), axis=-1).sum()) if len(tcp_positions) > 1 else 0.0,
        "velocity_rms_rad_s": velocity_rms,
        "velocity_max_rad_s": velocity_max,
        "acceleration_rms_rad_s2": acceleration_rms,
        "acceleration_max_rad_s2": acceleration_max,
        "jerk_rms_rad_s3": jerk_rms,
        "jerk_max_rad_s3": jerk_max,
        "inference_latency_mean_ms": float(np.mean(inference_latency_ms)),
        "inference_latency_max_ms": float(np.max(inference_latency_ms)),
    }
