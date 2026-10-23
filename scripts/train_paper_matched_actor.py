"""Matched from-scratch KAN/MLP actors; deployment anchor is never overwritten.

Both families use identical episodes, normalization, loss, shuffled batches,
optimizer/schedule and update count. Architecture-matched, not parameter-count
or wall-time matched: counts and elapsed times are explicitly reported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.utils.data import Subset
from src.data.dataset import ArmTrajectoryDataset
from src.data.splits import extend_episode_group_split
from src.models.trajectory_model import TrajectoryModel,TrajectoryModelConfig
from src.models.full_kan import assert_no_dense
from src.models.kan import BSplineKANLayer
from src.training.losses import LossWeights,supervised_loss
from src.training.trainer import lock_seed,make_loader,to_device,save_checkpoint
from src.planning.benchmark_gpu_lock import BenchmarkGPULock


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=['kan','mlp'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--smoke-batches',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.output.exists():raise RuntimeError('Fresh actor experiment required')
    if min(args.epochs,args.batch_size,args.accumulation)<=0 or args.lr<=0:raise ValueError('Invalid training budget')
    if args.smoke_batches<0:raise ValueError('Invalid smoke budget')
    lock_seed(args.seed);torch.set_num_threads(4)
    begun=time.perf_counter();args.output.mkdir(parents=True)
    dataset_root=ROOT/'dataset_rebuild/astronaut_strict_v4_geometric'
    manifest_path=dataset_root/'episode_split_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    dataset=ArmTrajectoryDataset(dataset_root/'left',history_length=128)
    indices,groups=extend_episode_group_split(dataset.group_ids,manifest['groups'],seed=args.seed)
    assert all(set(groups[k])==set(manifest['groups'][k]) for k in groups)
    # Read configuration/statistics only. NO teacher weights loaded into model.
    template_path=ROOT/'checkpoints/left/left_all_kan_h128_astronaut_v1_best.pt'
    template=torch.load(template_path,map_location='cpu',weights_only=False)
    cfg={**template['config'],'fully_kan':args.family=='kan',
         'semantic_encoder':args.family,'mlp_hidden_dim':128}
    del template
    config=TrajectoryModelConfig(**cfg)
    model=TrajectoryModel(config).to(args.device)
    if args.family=='kan':assert_no_dense(model)
    else:assert not any(isinstance(m,BSplineKANLayer) for m in model.modules())
    if model.collision_head is not None:
        for parameter in model.collision_head.parameters():parameter.requires_grad_(False)
    weights=LossWeights(action=2,terminal_error_boost=1.5,velocity=2e-4,acceleration=2e-5,jerk=2e-6,self_collision=0,inter_collision=0)
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=args.lr,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.1)
    train=make_loader(Subset(dataset,indices['train']),args.batch_size,True,args.seed)
    validation=make_loader(Subset(dataset,indices['validation']),args.batch_size,False,args.seed)
    files={str(path.relative_to(ROOT)):sha(path) for path in sorted((dataset_root/'left').glob('*.npz'))}
    save(args.output/'dataset_sources.json',files)
    protocol=dict(family=args.family,seed=args.seed,initialization='from_scratch_no_teacher_weights',
                  config=cfg,template_sha256=sha(template_path),split_manifest_sha256=sha(manifest_path),
                  dataset_sources_sha256=sha(args.output/'dataset_sources.json'),
                  split_frames={k:len(v) for k,v in indices.items()},epochs=args.epochs,
                  batch_size=args.batch_size,accumulation=args.accumulation,lr=args.lr,
                  weight_decay=1e-6,gradient_clip=1.,schedule='cosine_min_lr_0.1_initial',
                  loss_weights=weights.__dict__,smoke_only=bool(args.smoke_batches),
                  selection='minimum_validation_loss_only',test_used_for_selection=False,
                  parameter_count=sum(v.numel() for v in model.parameters()),
                  trainable_parameter_count=sum(v.numel() for v in model.parameters() if v.requires_grad),
                  source_sha256=sha(Path(__file__)),
                  limitation='Matched data and updates, not parameter count, compute time or pretrained deployment-anchor history')
    save(args.output/'protocol.json',protocol)
    def validate():
        model.eval();total=0.;count=0
        with torch.no_grad():
            for i,batch in enumerate(validation):
                if args.smoke_batches and i>=args.smoke_batches:break
                batch=to_device(batch,args.device)
                loss=supervised_loss(model(batch),batch,weights)['total_loss']
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite validation loss')
                n=len(batch['q_current']);total+=float(loss)*n;count+=n
        assert count>0
        return total/count
    best=validate();chosen_epoch=0;updates=0;rows=[]
    checkpoint=args.output/'best.pt'
    save_checkpoint(checkpoint,model,optimizer,0,{'validation_loss':best},cfg)
    batches=min(len(train),args.smoke_batches) if args.smoke_batches else len(train)
    for epoch in range(1,args.epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True);total=0.;count=0
        for i,batch in enumerate(train):
            if i>=batches:break
            batch=to_device(batch,args.device);loss=supervised_loss(model(batch),batch,weights)['total_loss']
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            group_start=i//args.accumulation*args.accumulation
            divisor=min(args.accumulation,batches-group_start)
            (loss/divisor).backward()
            n=len(batch['q_current']);total+=float(loss.detach())*n;count+=n
            if (i+1)%args.accumulation==0 or i+1==batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
        value=validate()
        if value<best:
            best=value;chosen_epoch=epoch
            save_checkpoint(checkpoint,model,optimizer,epoch,{'validation_loss':best},cfg)
        row=dict(epoch=epoch,train_loss=total/count,validation_loss=value,best=best,
                 lr=optimizer.param_groups[0]['lr'],updates=updates)
        rows.append(row);scheduler.step()
        save(args.output/'status.json',dict(status='running',rows=rows,selected_epoch=chosen_epoch))
        print(json.dumps(row),flush=True)
    assert updates==args.epochs*int(np.ceil(batches/args.accumulation))
    save(args.output/'status.json',dict(status='complete',rows=rows,selected_epoch=chosen_epoch,
         checkpoint_sha256=sha(checkpoint),updates=updates,seconds=time.perf_counter()-begun,
         smoke_only=bool(args.smoke_batches),test_evaluation='not_yet_run'))


if __name__=='__main__':
    with BenchmarkGPULock(ROOT/'results/sota_simple_v1/benchmark_gpu.lock'):main()
