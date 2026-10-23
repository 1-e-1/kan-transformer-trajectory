"""Warm-convert all dense maps to residual spline KAN; fit strict expert data."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.utils.data import Subset
from src.data.dataset import ArmTrajectoryDataset
from src.data.splits import extend_episode_group_split
from src.models.trajectory_model import TrajectoryModel,TrajectoryModelConfig
from src.models.full_kan import warm_start_from_dense,assert_no_dense
from src.models.kan import BSplineKANLayer
from src.training.losses import LossWeights,supervised_loss
from src.training.trainer import lock_seed,make_loader,to_device,save_checkpoint
from torch.utils.tensorboard import SummaryWriter


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke-batches',type=int,default=0)
    args=p.parse_args();lock_seed(args.seed);started=time.time()
    if args.output.exists() or args.report.exists():raise RuntimeError('Refusing to overwrite a prior run')
    assert args.epochs>0 and args.accumulation>0
    data=ArmTrajectoryDataset(args.dataset/'left',history_length=128)
    manifest_path=args.dataset/'episode_split_manifest.json';manifest_bytes=manifest_path.read_bytes();manifest=json.loads(manifest_bytes)
    indices,groups=extend_episode_group_split(data.group_ids,manifest['groups'],seed=args.seed)
    assert all(set(groups[k])==set(manifest['groups'][k]) for k in groups)
    old=torch.load(args.teacher,map_location=args.device,weights_only=False)
    teacher=TrajectoryModel(TrajectoryModelConfig(**old['config'])).to(args.device).eval();teacher.load_state_dict(old['model'],strict=True)
    config=TrajectoryModelConfig(**{**old['config'],'fully_kan':True,'dense_kan_grid_size':4,'dense_kan_spline_order':3})
    model=TrajectoryModel(config).to(args.device);conversion=warm_start_from_dense(model,teacher)
    probe=to_device(next(iter(make_loader(Subset(data,indices['train']),16,False,args.seed))),args.device)
    model.eval()
    with torch.no_grad():
        converted_action=model(probe)['action']
        fused_difference=float((converted_action-teacher(probe)['action']).abs().max())
        fastpath=torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(False)
            difference=float((converted_action-teacher(probe)['action']).abs().max())
        finally:torch.backends.mha.set_fastpath_enabled(fastpath)
    print(json.dumps({'warm_algebra_max_difference_rad':difference,'legacy_fused_kernel_max_difference_rad':fused_difference}),flush=True)
    assert difference<2e-5,'Warm conversion changed actions beyond floating-point tolerance'
    del teacher,old,probe
    for param in model.collision_head.parameters():param.requires_grad_(False)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-6)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.1)
    weights=LossWeights(action=2,terminal_error_boost=1.5,velocity=2e-4,acceleration=2e-5,jerk=2e-6,self_collision=0,inter_collision=0)
    train=make_loader(Subset(data,indices['train']),args.batch_size,True,args.seed)
    validation=make_loader(Subset(data,indices['validation']),args.batch_size,False,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    writer=SummaryWriter(args.report.parent/'tensorboard'/args.report.stem)
    def validate():
        model.eval();total=0.;count=0
        with torch.no_grad():
            for i,batch in enumerate(validation):
                batch=to_device(batch,args.device);loss=supervised_loss(model(batch),batch,weights)['total_loss']
                total+=float(loss)*len(batch['q_current']);count+=len(batch['q_current'])
                if args.smoke_batches and i+1>=args.smoke_batches:break
        return total/count
    best=validate();selected=0
    save_checkpoint(args.output,model,optimizer,0,{'validation_loss':best},config.__dict__)
    train_batches=min(len(train),args.smoke_batches) if args.smoke_batches else len(train)
    with args.report.with_suffix('.csv').open('w',newline='',encoding='utf-8') as stream:
        csv_writer=csv.DictWriter(stream,fieldnames=['epoch','train_loss','validation_loss','best','lr']);csv_writer.writeheader()
        for epoch in range(1,args.epochs+1):
            model.train();optimizer.zero_grad(set_to_none=True);total=0.;count=0
            for i,batch in enumerate(train):
                if i>=train_batches:break
                batch=to_device(batch,args.device);loss=supervised_loss(model(batch),batch,weights)['total_loss']
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite actor loss')
                group_start=(i//args.accumulation)*args.accumulation
                divisor=min(args.accumulation,train_batches-group_start)
                (loss/divisor).backward();total+=float(loss.detach());count+=1
                if (i+1)%args.accumulation==0 or i+1==train_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
            value=validate()
            if value<best:
                best=value;selected=epoch;save_checkpoint(args.output,model,optimizer,epoch,{'validation_loss':best},config.__dict__)
            row={'epoch':epoch,'train_loss':total/count,'validation_loss':value,'best':best,'lr':optimizer.param_groups[0]['lr']}
            csv_writer.writerow(row);stream.flush()
            for key,val in row.items():
                if key!='epoch':writer.add_scalar(key,val,epoch)
            scheduler.step();print(json.dumps(row),flush=True)
    writer.close()
    chosen=torch.load(args.output,map_location=args.device,weights_only=False);model.load_state_dict(chosen['model'])
    active_splines=[name for name,module in model.named_modules() if isinstance(module,BSplineKANLayer) and module.base_activation=='identity' and module.spline_weight.abs().max().item()>0]
    report={'status':'complete','smoke_only':bool(args.smoke_batches),'epochs':args.epochs,'selected_epoch':selected,
            'validation_loss':best,'all_kan_layers':assert_no_dense(model),'trained_new_spline_maps':active_splines,
            'conversion':conversion,'initial_action_max_difference_rad':difference,'initial_legacy_fused_difference_rad':fused_difference,'seed':args.seed,
            'batch_size':args.batch_size,'gradient_accumulation':args.accumulation,'loss_weights':weights.__dict__,
            'split_frames':{k:len(v) for k,v in indices.items()},'test_used_for_training_or_selection':False,
            'normalization':'inherited strict-v4 training-only statistics','manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
            'checkpoint':str(args.output.resolve()),'checkpoint_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
            'teacher_sha256':hashlib.sha256(args.teacher.read_bytes()).hexdigest(),'seconds':time.time()-started}
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
