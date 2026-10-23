import numpy as np
import torch

from src.data.dagger_buffer import ActionHistory
from src.data.dataset import ArmTrajectoryDataset, TENSOR_FIELDS
from src.training.losses import LossWeights, supervised_loss, teacher_preservation_loss


def test_action_history_private_fifo_and_padding():
    left, right = ActionHistory(4), ActionHistory(4)
    left.append(np.ones(8))
    right.append(-np.ones(8))
    actions_l, mask_l = left.padded()
    actions_r, mask_r = right.padded()
    assert mask_l.tolist() == [False, False, False, True]
    np.testing.assert_array_equal(actions_l[-1], np.ones(8))
    np.testing.assert_array_equal(actions_r[-1], -np.ones(8))
    for i in range(5):
        left.append(np.full(8, i))
    assert len(left._data) == 4


def test_supervised_losses_are_differentiable_not_bool_contacts():
    pred = torch.randn(2, 8, requires_grad=True)
    output = {"action": pred, "self_collision_logit": torch.randn(2, requires_grad=True), "inter_collision_logit": torch.randn(2, requires_grad=True)}
    batch = {
        "expert_action": torch.zeros(2, 8),
        "dt": torch.full((2,), 1 / 30),
        "history_action": torch.zeros(2, 64, 8),
        "history_mask": torch.zeros(2, 64, dtype=torch.bool),
        "self_collision_label": torch.tensor([0.0, 1.0]),
        "inter_collision_label": torch.tensor([1.0, 0.0]),
    }
    losses = supervised_loss(output, batch, LossWeights())
    losses["total_loss"].backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_dataset_can_expand_h64_to_h128(tmp_path):
    count = 2
    shapes = {
        "q_start": (8,), "q_current": (8,), "q_goal": (8,), "delta_q_goal": (8,),
        "x_start": (9,), "x_current": (9,), "x_goal": (9,), "delta_x_goal": (6,),
        "other_arm_geometry": (9, 3), "history_action": (64, 8), "history_mask": (64,),
        "expert_action": (8,), "self_collision_label": (), "inter_collision_label": (), "action_loss_mask": (), "dt": (),
        "collision_query_action": (8,),
    }
    arrays = {}
    for field in TENSOR_FIELDS:
        dtype = np.bool_ if field == "history_mask" else np.float32
        arrays[field] = np.ones((count, *shapes[field]), dtype=dtype)
    path = tmp_path / "data.npz"
    np.savez(path, **arrays)
    dataset = ArmTrajectoryDataset(path, history_length=128)
    assert dataset[0]["history_action"].shape == (128, 8)
    assert not dataset[0]["history_mask"][:64].any()
    assert dataset[0]["history_mask"][64:].all()


def test_dataset_recovers_true_older_history_from_same_original_episode(tmp_path):
    count = 66
    shapes = {
        "q_start": (8,), "q_current": (8,), "q_goal": (8,), "delta_q_goal": (8,),
        "x_start": (9,), "x_current": (9,), "x_goal": (9,), "delta_x_goal": (6,),
        "other_arm_geometry": (9, 3), "history_action": (64, 8), "history_mask": (64,),
        "expert_action": (8,), "self_collision_label": (), "inter_collision_label": (), "action_loss_mask": (), "dt": (),
        "collision_query_action": (8,),
    }
    arrays = {
        field: np.zeros((count, *shapes[field]), dtype=np.bool_ if field == "history_mask" else np.float32)
        for field in TENSOR_FIELDS
    }
    for step in range(count):
        valid = min(step, 64)
        if valid:
            actions = np.arange(step - valid + 1, step + 1, dtype=np.float32)
            arrays["history_action"][step, -valid:] = actions[:, None]
            arrays["history_mask"][step, -valid:] = True
    arrays.update(
        episode_id=np.zeros(count, dtype=np.int64),
        step_id=np.arange(count, dtype=np.int64),
        seed=np.full(count, 7, dtype=np.int64),
    )
    path = tmp_path / "sequential.npz"
    np.savez(path, **arrays)
    dataset = ArmTrajectoryDataset(path, history_length=128)
    history = dataset[65]["history_action"]
    mask = dataset[65]["history_mask"]
    assert mask.sum().item() == 65
    assert history[63, 0].item() == 1.0
    torch.testing.assert_close(history[64:, 0], torch.arange(2, 66, dtype=torch.float32))


def test_terminal_error_boost_changes_relative_sample_weight():
    pred = torch.full((2, 8), 0.1, requires_grad=True)
    batch = {
        "expert_action": torch.zeros(2, 8),
        "dt": torch.full((2,), 1 / 30),
        "history_action": torch.zeros(2, 64, 8),
        "history_mask": torch.zeros(2, 64, dtype=torch.bool),
        "self_collision_label": torch.zeros(2),
        "inter_collision_label": torch.zeros(2),
        "delta_q_goal": torch.stack((torch.zeros(8), torch.full((8,), 2.0))),
        "delta_x_goal": torch.stack((torch.zeros(6), torch.full((6,), 2.0))),
    }
    pred.data[1] = 0.2
    baseline = supervised_loss({"action": pred}, batch, LossWeights()).get("action_loss")
    boosted = supervised_loss(
        {"action": pred}, batch, LossWeights(terminal_error_boost=2.0)
    ).get("action_loss")
    assert boosted < baseline


def test_teacher_preservation_is_stronger_far_from_goal():
    student = torch.full((2, 8), 0.1, requires_grad=True)
    teacher = torch.zeros_like(student)
    batch = {
        "delta_q_goal": torch.stack((torch.zeros(8), torch.full((8,), 2.0))),
        "delta_x_goal": torch.stack((torch.zeros(6), torch.full((6,), 2.0))),
        "action_loss_mask": torch.ones(2),
    }
    both = teacher_preservation_loss(student, teacher, batch, minimum_weight=0.1)
    near_only = teacher_preservation_loss(
        student[:1], teacher[:1],
        {key: value[:1] for key, value in batch.items()},
        minimum_weight=0.1,
    )
    far_only = teacher_preservation_loss(
        student[1:], teacher[1:],
        {key: value[1:] for key, value in batch.items()},
        minimum_weight=0.1,
    )
    # Equal prediction errors have equal normalized losses in isolation, but the
    # combined loss is dominated by the far-from-goal sample's larger weight.
    torch.testing.assert_close(near_only, far_only)
    torch.testing.assert_close(both, far_only)
    both.backward()
    assert student.grad[1].abs().sum() > student.grad[0].abs().sum()


def test_teacher_preservation_rejects_invalid_minimum_weight():
    student = torch.zeros(1, 8)
    try:
        teacher_preservation_loss(student, student, {}, minimum_weight=1.1)
    except ValueError:
        return
    raise AssertionError("invalid minimum weight must be rejected")
