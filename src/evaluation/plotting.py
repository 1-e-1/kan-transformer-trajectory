"""Plots whose sole inputs are saved raw metric CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_h64_h128(left_csv, right_csv, output_png):
    left, right = _read(left_csv), _read(right_csv)
    variants = list(dict.fromkeys(row["variant"] for row in left))
    def mean(rows, variant, key):
        return float(np.mean([float(row[key]) for row in rows if row["variant"] == variant]))
    width = 0.35
    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(x - width / 2, [mean(left, variant, "tcp_position_error_m") for variant in variants], width, label="left")
    axes[0].bar(x + width / 2, [mean(right, variant, "tcp_position_error_m") for variant in variants], width, label="right")
    axes[0].set_xticks(x, variants); axes[0].set_ylabel("TCP position error [m]"); axes[0].legend()
    axes[1].bar(x - width / 2, [mean(left, variant, "inference_latency_mean_ms") for variant in variants], width, label="left")
    axes[1].bar(x + width / 2, [mean(right, variant, "inference_latency_mean_ms") for variant in variants], width, label="right")
    axes[1].set_xticks(x, variants); axes[1].set_ylabel("Inference latency [ms]")
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(output_png, dpi=160); plt.close(fig)
