import sys
from pathlib import Path
import pytest
import torch
from src.models.full_kan import assert_no_dense,warm_start_from_dense
from src.models.trajectory_model import TrajectoryModel,TrajectoryModelConfig
from src.models.kan import BSplineKANLayer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ppo/left_only'))
from potential_model import LeftOnlyPotentialField
from potential_controller import load_potential


def observation(device):
    batch={name:torch.randn(2,dim,device=device) for name,dim in
           [('q_start',8),('q_current',8),('q_goal',8),('delta_q_goal',8),
            ('x_start',9),('x_current',9),('x_goal',9),('delta_x_goal',6)]}
    batch.update(other_arm_geometry=torch.randn(2,9,3,device=device),
                 history_action=torch.randn(2,8,8,device=device),
                 history_mask=torch.tensor([[False]*4+[True]*4]*2,device=device))
    return batch


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda:0',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_all_kan_conversion_preserves_teacher_and_trains_splines(device):
    # dropout=0; train mode avoids the legacy fused encoder's reduced-precision
    # inference kernel so this checks the algebraic conversion itself.
    teacher=TrajectoryModel(TrajectoryModelConfig(history_length=8)).to(device).train()
    model=TrajectoryModel(TrajectoryModelConfig(history_length=8,fully_kan=True)).to(device).eval()
    audit=warm_start_from_dense(model,teacher)
    assert len(audit['converted_maps'])==18 # 1 history + 12 attention/FFN + 2 action + 3 aux
    assert assert_no_dense(model)==23
    batch=observation(device)
    with torch.no_grad():
        old=teacher(batch);new=model(batch)
        for key in old:torch.testing.assert_close(new[key],old[key],atol=2e-6,rtol=2e-5)
        modified={**batch,'history_action':batch['history_action'].clone()}
        modified['history_action'][:,:4]=1e4
        torch.testing.assert_close(model(modified)['action'],new['action'],atol=2e-6,rtol=2e-5)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    model(batch)['action'].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    layer=model.transformer.encoder.layers[0].self_attn.qkv
    assert layer.spline_weight.abs().max()>0


def test_full_kan_field_checkpoint_roundtrip_and_input_gradient(tmp_path):
    teacher=LeftOnlyPotentialField(hidden_dim=16,residual_blocks=1,dropout=0).eval()
    model=LeftOnlyPotentialField(hidden_dim=16,residual_blocks=1,dropout=0,fully_kan=True).eval()
    warm_start_from_dense(model,teacher)
    q=torch.randn(6,8,requires_grad=True)
    for key,value in teacher(q).items():torch.testing.assert_close(model(q)[key],value,atol=2e-6,rtol=2e-5)
    model(q)['collision_logit_fixed'].sum().backward()
    assert torch.isfinite(q.grad).all()
    path=tmp_path/'field.pt'
    torch.save({'architecture':model.architecture(),'model_state':model.state_dict(),'thresholds':{'self':.2,'fixed':.3}},path)
    loaded,_=load_potential(path,'cpu')
    assert assert_no_dense(loaded.field)==10
    out,gradient=loaded.barriers(q.detach())
    assert torch.isfinite(gradient).all() and out['barrier'].shape==(6,2)


def test_full_kan_actor_checkpoint_roundtrip():
    config=TrajectoryModelConfig(history_length=8,fully_kan=True)
    model=TrajectoryModel(config).eval()
    loaded=TrajectoryModel(TrajectoryModelConfig(**config.__dict__)).eval()
    loaded.load_state_dict(model.state_dict(),strict=True)
    batch=observation('cpu')
    torch.testing.assert_close(model(batch)['action'],loaded(batch)['action'])


def test_single_robot_padding_pruning_preserves_action():
    model=TrajectoryModel(TrajectoryModelConfig(history_length=8,fully_kan=True)).eval()
    batch={key:value[:1] for key,value in observation('cpu').items()}
    with torch.no_grad():
        pruned=model(batch)['action']
        # Dropout is zero: train mode only disables the inference pruning.
        model.train()
        dense=model(batch)['action']
    torch.testing.assert_close(pruned,dense,atol=2e-6,rtol=2e-5)
