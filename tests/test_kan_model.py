import torch

from src.models.kan import BSplineKANLayer, JointKAN, OtherArmKAN, PoseErrorKAN, PoseKAN
from src.models.mlp import JointMLP, OtherArmMLP, PoseErrorMLP, PoseMLP
from src.models.trajectory_model import TrajectoryModel, TrajectoryModelConfig, assert_models_independent


def sample_batch(batch_size=3, history_length=64, points=9):
    return {
        "q_start": torch.randn(batch_size, 8),
        "q_current": torch.randn(batch_size, 8),
        "q_goal": torch.randn(batch_size, 8),
        "delta_q_goal": torch.randn(batch_size, 8),
        "x_start": torch.randn(batch_size, 9),
        "x_current": torch.randn(batch_size, 9),
        "x_goal": torch.randn(batch_size, 9),
        "delta_x_goal": torch.randn(batch_size, 6),
        "other_arm_geometry": torch.randn(batch_size, points, 3),
        "history_action": torch.randn(batch_size, history_length, 8),
        "history_mask": torch.cat((torch.zeros(batch_size, history_length - 5, dtype=torch.bool), torch.ones(batch_size, 5, dtype=torch.bool)), dim=1),
    }


def test_bspline_kan_forward_backward():
    layer = BSplineKANLayer(16, 64, grid_size=8, spline_order=3)
    x = torch.randn(5, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == (5, 64)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.spline_weight.grad is not None and torch.isfinite(layer.spline_weight.grad).all()


def test_required_kan_shapes():
    assert JointKAN()(torch.randn(2, 16)).shape == (2, 64)
    assert PoseKAN()(torch.randn(2, 9)).shape == (2, 64)
    assert PoseErrorKAN()(torch.randn(2, 6)).shape == (2, 64)
    assert OtherArmKAN(9)(torch.randn(2, 27)).shape == (2, 64)


def test_required_mlp_shapes():
    assert JointMLP()(torch.randn(2, 16)).shape == (2, 64)
    assert PoseMLP()(torch.randn(2, 9)).shape == (2, 64)
    assert PoseErrorMLP()(torch.randn(2, 6)).shape == (2, 64)
    assert OtherArmMLP(9)(torch.randn(2, 27)).shape == (2, 64)


def test_mlp_transformer_is_kan_free_and_bounded():
    cfg = TrajectoryModelConfig(history_length=128, semantic_encoder="mlp")
    model = TrajectoryModel(cfg)
    assert not any(isinstance(module, BSplineKANLayer) for module in model.modules())
    output = model(sample_batch(history_length=128))
    assert output["action"].shape == (3, 8)
    assert torch.max(torch.abs(output["action"])) <= cfg.action_delta_max_rad
    output["action"].square().mean().backward()


def test_kan_checkpoint_can_reinitialize_only_semantic_mlp_encoders():
    kan = TrajectoryModel(TrajectoryModelConfig(history_length=128))
    mlp = TrajectoryModel(TrajectoryModelConfig(history_length=128, semantic_encoder="mlp"))
    before = mlp.tokenizer.joint.mlp[0].weight.detach().clone()
    mlp.load_compatible_state_dict(
        kan.state_dict(), allow_semantic_encoder_reinit=True
    )
    torch.testing.assert_close(
        mlp.transformer.encoder.layers[0].self_attn.in_proj_weight,
        kan.transformer.encoder.layers[0].self_attn.in_proj_weight,
    )
    torch.testing.assert_close(mlp.tokenizer.joint.mlp[0].weight, before)


def test_transformer_output_mask_and_action_bound():
    cfg = TrajectoryModelConfig(history_length=64, other_arm_points=9)
    model = TrajectoryModel(cfg)
    output = model(sample_batch())
    assert output["action"].shape == (3, 8)
    assert output["self_collision_logit"].shape == (3,)
    assert torch.max(torch.abs(output["action"])) <= cfg.action_delta_max_rad
    output["action"].square().mean().backward()


def test_left_right_models_are_independent():
    cfg = TrajectoryModelConfig()
    left, right = TrajectoryModel(cfg), TrajectoryModel(cfg)
    assert_models_independent(left, right)


def test_collision_head_is_candidate_action_conditioned():
    torch.manual_seed(5)
    model = TrajectoryModel(TrajectoryModelConfig(history_length=8)).eval()
    batch = sample_batch(batch_size=2, history_length=8)
    batch["collision_query_action"] = torch.zeros(2, 8)
    first = model(batch)["self_collision_logit"]
    batch["collision_query_action"] = torch.full((2, 8), 0.05)
    second = model(batch)["self_collision_logit"]
    assert not torch.allclose(first, second)
