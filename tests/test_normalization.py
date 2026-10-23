import torch
import numpy as np

from src.data.normalization import ObservationNormalizer, compute_normalization


def test_normalizer_keeps_padding_zero_and_rotation_untouched():
    stats = {
        "position": {"mean": [1, 2, 3], "scale": [2, 2, 2]},
        "delta_q_goal": {"mean": [0] * 8, "scale": [2] * 8},
        "delta_x_goal": {"mean": [0] * 6, "scale": [2] * 6},
        "other_arm_geometry": {"mean": [0] * 27, "scale": [2] * 27},
        "history_action": {"mean": [1] * 8, "scale": [2] * 8},
    }
    batch = {
        "x_start": torch.tensor([[1, 2, 3, 1, 0, 0, 0, 1, 0]], dtype=torch.float32),
        "x_current": torch.tensor([[1, 2, 3, 1, 0, 0, 0, 1, 0]], dtype=torch.float32),
        "x_goal": torch.tensor([[1, 2, 3, 1, 0, 0, 0, 1, 0]], dtype=torch.float32),
        "delta_q_goal": torch.ones(1, 8), "delta_x_goal": torch.ones(1, 6),
        "other_arm_geometry": torch.ones(1, 9, 3),
        "history_action": torch.ones(1, 2, 8), "history_mask": torch.tensor([[False, True]]),
    }
    output = ObservationNormalizer(stats)(batch)
    assert torch.equal(output["x_current"][0, 3:], batch["x_current"][0, 3:])
    assert torch.count_nonzero(output["history_action"][0, 0]) == 0
    assert torch.count_nonzero(output["history_action"][0, 1]) == 0


def test_statistics_use_only_supplied_train_indices():
    class Dataset:
        pass
    dataset = Dataset()
    dataset.arrays = {
        "x_start": np.array([[0] * 9, [100] * 9], dtype=np.float32),
        "x_current": np.array([[0] * 9, [100] * 9], dtype=np.float32),
        "x_goal": np.array([[0] * 9, [100] * 9], dtype=np.float32),
        "delta_q_goal": np.array([[0] * 8, [100] * 8], dtype=np.float32),
        "delta_x_goal": np.array([[0] * 6, [100] * 6], dtype=np.float32),
        "other_arm_geometry": np.array([[[0] * 3] * 9, [[100] * 3] * 9], dtype=np.float32),
        "history_action": np.array([[[0] * 8], [[100] * 8]], dtype=np.float32),
        "history_mask": np.ones((2, 1), dtype=bool),
    }
    stats = compute_normalization(dataset, [0])
    assert stats["position"]["mean"] == [0.0, 0.0, 0.0]
    assert stats["delta_q_goal"]["mean"] == [0.0] * 8
